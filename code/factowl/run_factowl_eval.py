#!/usr/bin/env python3
"""Run FactOWL evaluation with explicit logging and progress bars.

Usage (GPU shell):
  CUDA_VISIBLE_DEVICES=0 python factowl/run_factowl_eval.py --domain rivers --sample 4

Domains: cars, rivers, disasters (disasters == bad_weather)

This uses the self-consistency 99-gen HF exports, injects wiki pages as
contexts (no Wikipedia API calls), and writes logs to
/home/moskovskiy/LongFT/factowl_eval/factowl.log.
"""

import argparse
import logging
import os
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download, login
from vllm import LLM, SamplingParams

# Local package
pkg_dir = Path(__file__).resolve().parent
if str(pkg_dir) not in sys.path:
    sys.path.append(str(pkg_dir))
from factowl.factscorer_sped_up_vllm import FactScorerSpedUpVLLM as FactScorer
from factowl.atomic_facts_sped_up_vllm import AtomicFactGeneratorSpedUpVLLM


DOMAIN_CFG = {
    "cars": {
        "en": ("cars", "s-nlp/RiDiC_cars_self_consistency_99_generations"),
        "zh": ("cars_zh", "s-nlp/RiDiC_cars_zh_self_consistency_99_generations"),
    },
    "rivers": {
        "en": ("rivers", "s-nlp/RiDiC_rivers_self_consistency_99_generations"),
        "zh": ("rivers_zh", "s-nlp/RiDiC_rivers_zh_self_consistency_99_generations"),
    },
    "disasters": {
        "en": ("disasters", "s-nlp/RiDiC_bad_weather_self_consistency_99_generations"),
        "zh": ("bad_weather_zh", "s-nlp/RiDiC_bad_weather_zh_self_consistency_99_generations"),
    },
}

ZH_META_TO_EN_BASE = {
    "rivers_zh": "rivers",
    "cars_zh": "cars",
    "bad_weather_zh": "disasters",
}

AGGREGATION_MARKERS = {
    "en": "Final Consolidated Answer:",
    "zh": "最终整合后的答案：",
}


def default_prompt_workers() -> int:
    return max(1, min(32, os.cpu_count() or 1))


def render_chat_prompts(tokenizer, messages_list, label: str):
    if not messages_list:
        return []
    workers = max(1, int(os.environ.get("FACTOWL_PROMPT_THREADS", default_prompt_workers())))
    render_start = time.perf_counter()
    if len(messages_list) <= 1 or workers <= 1:
        prompts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(messages_list))) as executor:
            prompts = list(
                executor.map(
                    lambda messages: tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                    messages_list,
                )
            )
    logging.info(
        "Rendered %s %s prompts in %.2fs with %s worker(s)",
        len(prompts),
        label,
        time.perf_counter() - render_start,
        min(workers, len(messages_list)),
    )
    return prompts


def compose_chat_messages(system_prompt: str, instructions: str, user_question: str):
    messages = []
    if system_prompt:
        messages.append({"role": "user", "content": system_prompt})
    if instructions:
        messages.append({"role": "user", "content": instructions})
    messages.append({"role": "user", "content": user_question})
    return messages


def build_aggregation_prompt(tokenizer, question: str, candidate_answers: list[str], lang: str) -> str:
    marker = AGGREGATION_MARKERS["zh"] if lang == "zh" else AGGREGATION_MARKERS["en"]
    if lang == "zh":
        system_prompt = (
            "你是一名答案聚合器。你的任务是阅读同一问题的多个候选答案，"
            "并输出一个连贯且高质量的最终答案。"
        )
        instructions = (
            "严格遵循以下规则：\n"
            "1. 找出每个候选答案中的关键信息点。\n"
            "2. 保留在大多数答案中出现的要点（≥ 一半）。\n"
            "3. 可以加入候选答案中已有且不与多数观点冲突的补充细节。\n"
            "4. 当答案不一致时，选择由多数支持的版本；若持平，选择更清晰或更详细的表述。\n"
            "5. 将内容重写为一个结构良好的单一回答。\n"
            "6. 不要提及聚合过程、投票或其他答案的存在。\n\n"
            "输出格式（且仅输出该格式）：\n"
            f"{marker}\n"
            "[你的整合答案]\n"
        )
        header = "问题"
        answers_label = "候选答案（以 ### 分隔）"
        footer = "输入结束。"
    else:
        system_prompt = (
            "You are an Answer Aggregator. Your task is to read several "
            "candidate answers to the same question and produce ONE coherent, "
            "high-quality answer."
        )
        instructions = (
            "Follow these rules exactly:\n"
            "1. Identify the main points made in each candidate answer.\n"
            "2. Keep every point that appears in a majority of the answers (>= half).\n"
            "3. Add complementary details that are already present in the existing answers and do not contradict the majority view.\n"
            "4. When answers disagree, choose the version supported by the majority. If tied, pick the clearer or more detailed wording.\n"
            "5. Rewrite everything into a single, well-structured response.\n"
            "6. Do not mention the aggregation process, voting, or the existence of other answers.\n\n"
            "Output format (nothing else):\n"
            f"{marker}\n"
            "[your aggregated answer]\n"
        )
        header = "Question"
        answers_label = "Candidate Answers (delimited by ###)"
        footer = "End of input."

    answers_block = "\n".join(
        [f"### Answer {idx + 1}\n{ans}" for idx, ans in enumerate(candidate_answers)]
    )
    user_question = (
        f"{header}:\n{question}\n\n"
        f"{answers_label}:\n{answers_block}\n\n"
        f"{footer}"
    )
    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
    )
    return messages


def extract_aggregated_answer(text: str, lang: str) -> str:
    marker = AGGREGATION_MARKERS["zh"] if lang == "zh" else AGGREGATION_MARKERS["en"]
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    lower = stripped.lower()
    marker_lower = marker.lower()
    pos = lower.find(marker_lower)
    if pos != -1:
        stripped = stripped[pos + len(marker):].strip()
    return stripped.strip().strip('"').strip("'")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=DOMAIN_CFG.keys(), required=True)
    p.add_argument("--sample", type=int, default=4, help="Subset size; -1 for full")
    p.add_argument("--lang", default="en", choices=["en", "zh"], help="Language for scoring / column selection")
    p.add_argument("--gpu", type=str, default="2", help="CUDA device(s), e.g. 0 or '0,1'")
    p.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    p.add_argument("--data-dir", default="/workspace/longft/factowl/data/")
    p.add_argument("--hf-home", default=str(Path.home() / ".cache/huggingface"))
    p.add_argument("--log", default="/workspace/longft/factowl/factowl_eval/factowl.log")
    p.add_argument("--max-tokens", type=int, default=256, help="Max new tokens for atomic fact gen")
    p.add_argument("--batch-size", type=int, default=2000, help="Number of completions per scoring chunk")
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                   help="Target fraction of total GPU memory that vLLM may reserve")
    p.add_argument("--all-completions", action="store_true",
                   help="Score every completion in all_completions (99 per question) instead of only final_answer")
    p.add_argument("--final-only", action="store_true",
                   help="Trim each completion to text after 'Final Answer:' before scoring")
    p.add_argument("--llm-aggregate", action="store_true",
                   help="Aggregate all_completions into one answer per question using the majority_vote_with_llm prompt before FactOwl scoring")
    p.add_argument("--aggregation-max-tokens", type=int, default=256,
                   help="Max new tokens for the LLM aggregation pass")
    p.add_argument("--aggregation-temperature", type=float, default=0.0,
                   help="Sampling temperature for the LLM aggregation pass")
    p.add_argument("--aggregation-batch-size", type=int, default=128,
                   help="Number of aggregation prompts to send to vLLM at once")
    p.add_argument("--aggregate-k", type=int, default=None,
                   help="Use only the first k completions when --llm-aggregate is enabled")
    p.add_argument("--prompt-workers", type=int, default=default_prompt_workers(),
                   help="CPU worker threads for large prompt rendering stages")
    p.add_argument("--hf-token", default=None,
                   help="HF token for private/rate-limited datasets; if omitted, uses HF_TOKEN env")
    return p.parse_args()


def setup_env(args):
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["VLLM_USE_RAY"] = "0"
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HOME"] = args.hf_home
    os.environ["FACTOWL_PROMPT_THREADS"] = str(max(1, args.prompt_workers))
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False, write_permission=False)


def setup_logging(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(path, mode="a"),
        ],
    )


def load_sc(meta_cfg: str, repo: str, lang: str) -> Dataset:
    fp = hf_hub_download(repo_id=repo, filename="data-00000-of-00001.arrow", repo_type="dataset")
    gens = Dataset.from_file(fp)

    if lang == "zh":
        base_cfg = ZH_META_TO_EN_BASE.get(meta_cfg, meta_cfg.replace("_zh", ""))
        meta = load_dataset("s-nlp/RiDiC", base_cfg, split="test")
        for col in ["title_en", "wikipedia_page_en", "title_zh", "wikipedia_page_zh", "popularity_part_sector"]:
            if col in meta.column_names:
                gens = gens.add_column(col, meta[col])
    else:
        meta = load_dataset("s-nlp/RiDiC", meta_cfg, split="test")
        for col in ["title_en", "wikipedia_page_en", "title_zh", "wikipedia_page_zh", "popularity_part_sector"]:
            if col in meta.column_names:
                gens = gens.add_column(col, meta[col])
    return gens


def strip_final_only(text: str) -> str:
    if not isinstance(text, str):
        return text
    markers = ["Final Answer:", "最终答案：", "最终答案:"]
    for marker in markers:
        if marker in text:
            return text.split(marker, 1)[1].strip()
    return text


def build_output_stem(args, suffix_parts: list[str]) -> str:
    return f"pred_{args.domain}_{args.lang}_k99_{'_'.join(suffix_parts)}"


def aggregate_completions_with_llm(ds: Dataset, title_col: str, topic2cxt: dict, args, llm: LLM):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling_kwargs = {
        "temperature": args.aggregation_temperature,
        "max_tokens": args.aggregation_max_tokens,
    }
    if tokenizer.eos_token:
        sampling_kwargs["stop"] = [tokenizer.eos_token]
    sampling_params = SamplingParams(**sampling_kwargs)

    rows = []
    prompt_messages = []
    fallback_answers = []
    for ex in ds:
        topic = ex.get(title_col)
        if not topic or topic not in topic2cxt:
            continue
        completions = ex.get("all_completions") or []
        if not isinstance(completions, list):
            continue
        used_completions = completions[:args.aggregate_k] if args.aggregate_k and args.aggregate_k > 0 else completions
        candidates = [strip_final_only(c) if args.final_only else c for c in used_completions]
        candidates = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
        if not candidates:
            continue

        prompt_messages.append(build_aggregation_prompt(tokenizer, ex["question"], candidates, args.lang))
        fallback = Counter(candidates).most_common(1)[0][0]
        fallback_answers.append(fallback)
        rows.append({
            "id": ex["id"],
            "question": ex["question"],
            "topic": topic,
            "popularity_part_sector": ex["popularity_part_sector"],
            "num_candidates_total": len(completions),
            "num_candidates_used": len(used_completions),
            "num_nonempty_candidates": len(candidates),
            "majority_fallback": fallback,
        })

    if not prompt_messages:
        return [], [], [], [], pd.DataFrame()
    prompts = render_chat_prompts(tokenizer, prompt_messages, "aggregation")

    aggregated_answers = []
    bs = max(1, args.aggregation_batch_size)
    for start in range(0, len(prompts), bs):
        end = min(start + bs, len(prompts))
        logging.info("aggregation batch %s:%s / %s", start, end, len(prompts))
        outputs = llm.generate(prompts[start:end], sampling_params, use_tqdm=True)
        for out, fallback in zip(outputs, fallback_answers[start:end]):
            raw = out.outputs[0].text if out.outputs else ""
            final_answer = extract_aggregated_answer(raw, args.lang)
            aggregated_answers.append(final_answer or fallback)

    for row, answer in zip(rows, aggregated_answers):
        row["aggregated_answer"] = answer

    answers_df = pd.DataFrame(rows)
    topics = answers_df["topic"].tolist()
    generations = answers_df["aggregated_answer"].tolist()
    pops = answers_df["popularity_part_sector"].tolist()
    comp_idx = [0] * len(generations)
    return topics, generations, pops, comp_idx, answers_df


def build_generations(ds: Dataset, title_col: str, topic2cxt: dict, args, llm: LLM, out_dir: Path):
    suffix_parts = []
    answers_tsv = None
    if args.llm_aggregate:
        suffix_parts.append("llmagg")
        if args.aggregate_k and args.aggregate_k > 0:
            suffix_parts.append(f"aggk{args.aggregate_k}")
        if args.final_only:
            suffix_parts.append("finalonly")
        if args.sample and args.sample > 0:
            suffix_parts.append(f"s{args.sample}")
        topics, generations, pops, comp_idx, answers_df = aggregate_completions_with_llm(
            ds=ds,
            title_col=title_col,
            topic2cxt=topic2cxt,
            args=args,
            llm=llm,
        )
        answers_tsv = out_dir / f"{build_output_stem(args, suffix_parts)}_answers.tsv"
        answers_df.to_csv(answers_tsv, sep="\t", index=False)
        logging.info("saved aggregated answers -> %s", answers_tsv)
        return topics, generations, pops, comp_idx, suffix_parts, answers_tsv

    if args.all_completions:
        suffix_parts.append("allcompletions")
        flat_gen, flat_topics, flat_pops, flat_comp_idx = [], [], [], []
        for comps, t, p in zip(ds["all_completions"], ds[title_col], ds["popularity_part_sector"]):
            if t is None or t not in topic2cxt:
                continue
            if not isinstance(comps, list):
                continue
            for j, g in enumerate(comps):
                if args.final_only:
                    g = strip_final_only(g)
                flat_gen.append(g)
                flat_topics.append(t)
                flat_pops.append(p)
                flat_comp_idx.append(j)
        generations, topics, pops, comp_idx = flat_gen, flat_topics, flat_pops, flat_comp_idx
    else:
        suffix_parts.append("final")
        topics = list(ds[title_col])
        generations = list(ds["final_answer"])
        if args.final_only:
            generations = [strip_final_only(g) for g in generations]
        pops = list(ds["popularity_part_sector"])
        comp_idx = [0] * len(generations)

    if args.final_only:
        suffix_parts.append("finalonly")
    if args.sample and args.sample > 0:
        suffix_parts.append(f"s{args.sample}")

    filt = [(g, t, p, c) for g, t, p, c in zip(generations, topics, pops, comp_idx) if t and topic2cxt.get(t)]
    if filt:
        generations, topics, pops, comp_idx = zip(*filt)
        generations, topics, pops, comp_idx = list(generations), list(topics), list(pops), list(comp_idx)
    else:
        generations, topics, pops, comp_idx = [], [], [], []

    return topics, generations, pops, comp_idx, suffix_parts, answers_tsv


def main():
    args = parse_args()
    setup_env(args)
    setup_logging(args.log)

    meta_cfg, repo = DOMAIN_CFG[args.domain][args.lang]
    logging.info(
        "domain=%s, lang=%s, meta_cfg=%s, repo=%s, sample=%s, gpu=%s, prompt_workers=%s",
        args.domain,
        args.lang,
        meta_cfg,
        repo,
        args.sample,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("FACTOWL_PROMPT_THREADS"),
    )

    ds = load_sc(meta_cfg, repo, args.lang)
    if args.sample and args.sample > 0:
        ds = ds.select(range(args.sample))

    title_col = "title_zh" if args.lang == "zh" else "title_en"
    page_col = "wikipedia_page_zh" if args.lang == "zh" else "wikipedia_page_en"
    topic2cxt = {t: [{"title": t, "text": pg}] for t, pg in zip(ds[title_col], ds[page_col]) if t and pg}

    logging.info("loading vLLM model…")
    llm = LLM(
        model=args.model,
        download_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        tensor_parallel_size=args.tp_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    out_dir = Path("/workspace/longft/experiments/factowl_eval/")
    out_dir.mkdir(parents=True, exist_ok=True)

    topics, generations, pops, comp_idx, suffix_parts, answers_tsv = build_generations(
        ds=ds,
        title_col=title_col,
        topic2cxt=topic2cxt,
        args=args,
        llm=llm,
        out_dir=out_dir,
    )

    logging.info(
        "filtered to %s examples with contexts (all_completions=%s, llm_aggregate=%s)",
        len(topics),
        args.all_completions,
        args.llm_aggregate,
    )

    fs = FactScorer(
        vllm_model=llm,
        model_name="retrieval+llama",
        data_dir=args.data_dir,
        cache_dir=args.data_dir,
        use_this_topic2content_only=topic2cxt,
        context_type="wikipedia_api",
        batched_fact_generation=True,
        batched_fact_verification=True,
        verbose=False,
        debug=False,
        lang=args.lang,
        fact_generator_max_tokens=args.max_tokens,
    )

    fs.af_generator = AtomicFactGeneratorSpedUpVLLM(
        demon_dir=os.path.join(args.data_dir, "demos"),
        vllm_model=llm,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=0.0,
        vllm_tqdm=True,
        lang=args.lang,
        debug=False,
    )

    logging.info("starting scoring in batches…")
    results = []
    total = len(topics)
    bs = max(1, args.batch_size)
    for start in range(0, total, bs):
        end = min(start + bs, total)
        logging.info("batch %s:%s / %s", start, end, total)
        batch_start = time.perf_counter()
        out = fs.get_score(
            topics[start:end],
            generations[start:end],
            gamma=10,
            knowledge_source="provided",
            all_atomic_facts=None,
            verbose=False,
            completion_labels=comp_idx[start:end],
        )
        logging.info("completed batch %s:%s in %.2fs", start, end, time.perf_counter() - batch_start)
        results.append(pd.DataFrame(out))

    if not results:
        raise ValueError("No FactOWL results were produced; check filtering and input data.")

    tsv = out_dir / f"{build_output_stem(args, suffix_parts)}.tsv"
    pd.concat(results, ignore_index=True).to_csv(tsv, sep="\t", index=False)
    logging.info("saved results -> %s", tsv)
    if answers_tsv is not None:
        logging.info("aggregation sidecar -> %s", answers_tsv)


if __name__ == "__main__":
    main()
