#!/usr/bin/env python3
"""Run FactOwl LLM-aggregation prefix-k curves.

This evaluates the LLM aggregation path on the first k completions for a list
of k values, reusing one vLLM model instance across the full sweep.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
from vllm import LLM

import run_factowl_eval as rfe
from factowl.atomic_facts_sped_up_vllm import AtomicFactGeneratorSpedUpVLLM
from factowl.factscorer_sped_up_vllm import FactScorerSpedUpVLLM as FactScorer
from factowl.factscorer_sped_up_vllm import calculate_score_from_decisions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=rfe.DOMAIN_CFG.keys(), required=True)
    p.add_argument("--lang", default="en", choices=["en", "zh"])
    p.add_argument("--sample", type=int, default=-1, help="Subset size; -1 for full")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    p.add_argument("--data-dir", default="/workspace/longft/factowl/data/")
    p.add_argument("--hf-home", default=str(Path.home() / ".cache/huggingface"))
    p.add_argument("--log", default="/workspace/longft/factowl/factowl_eval/factowl_llmagg_curve.log")
    p.add_argument("--max-tokens", type=int, default=256, help="Max new tokens for atomic fact generation")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--final-only", action="store_true")
    p.add_argument("--aggregation-max-tokens", type=int, default=256)
    p.add_argument("--aggregation-temperature", type=float, default=0.0)
    p.add_argument("--aggregation-batch-size", type=int, default=128)
    p.add_argument("--prompt-workers", type=int, default=rfe.default_prompt_workers(),
                   help="Thread count for prompt rendering, forwarded to FactOwl setup_env")
    p.add_argument("--k-list", default="1,2,4,8,16,32,64,99",
                   help="Comma-separated prefix-k values to evaluate")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--output-dir", default="/workspace/longft/experiments/factowl_eval/")
    p.add_argument("--save-per-k-details", action="store_true",
                   help="Save the full FactOwl per-fact TSV for each evaluated k")
    p.add_argument("--save-topic-details", action="store_true",
                   help="Save per-topic FactOwl scores for each evaluated k")
    return p.parse_args()


def parse_k_list(k_list: str) -> list[int]:
    ks = []
    for piece in k_list.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise ValueError(f"k must be positive, got {value}")
        ks.append(value)
    if not ks:
        raise ValueError("No valid k values were provided")
    return sorted(dict.fromkeys(ks))


def build_curve_stem(args) -> str:
    suffix_parts = ["llmaggcurve"]
    if args.final_only:
        suffix_parts.append("finalonly")
    if args.sample and args.sample > 0:
        suffix_parts.append(f"s{args.sample}")
    return rfe.build_output_stem(args, suffix_parts)


def make_fs(args, llm, topic2cxt):
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
    return fs


def plot_curve(df: pd.DataFrame, out_png: Path, title: str):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(df["k"], df["score"], label="score", linewidth=2)
    ax[0].plot(df["k"], df["init_score"], label="init_score", linewidth=2)
    ax[0].set_xlabel("k (first k samples)")
    ax[0].set_ylabel("Score")
    ax[0].set_title(title)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].plot(df["k"], df["respond_ratio"], label="respond_ratio", linewidth=2)
    ax[1].plot(df["k"], df["num_facts_per_response"], label="facts/response", linewidth=2)
    ax[1].set_xlabel("k (first k samples)")
    ax[1].set_ylabel("Metric")
    ax[1].set_title("Support Metrics")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)


def build_topic_detail_df(topics: list[str], out: dict, k: int) -> pd.DataFrame:
    grouped = {topic: [] for topic in topics}
    for decision in out.get("decisions", []):
        grouped.setdefault(decision["topic"], []).append(decision)

    rows = []
    for topic in topics:
        decisions = grouped.get(topic, [])
        atomic_facts = [d["atom"] for d in decisions] if decisions else None
        topic_out = calculate_score_from_decisions(
            all_atomic_facts=[atomic_facts],
            decisions=[decisions],
            gamma=10,
        )
        rows.append(
            {
                "topic": topic,
                "k": k,
                "score": topic_out["score"],
                "init_score": topic_out.get("init_score", topic_out["score"]),
                "respond_ratio": topic_out["respond_ratio"],
                "num_facts_per_response": topic_out["num_facts_per_response"],
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    ks = parse_k_list(args.k_list)
    rfe.setup_env(args)
    rfe.setup_logging(args.log)

    meta_cfg, repo = rfe.DOMAIN_CFG[args.domain][args.lang]
    logging.info("domain=%s lang=%s repo=%s ks=%s", args.domain, args.lang, repo, ks)

    ds = rfe.load_sc(meta_cfg, repo, args.lang)
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
    fs = make_fs(args, llm, topic2cxt)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_stem = build_curve_stem(args)
    rows = []
    topic_detail_frames = []

    for k in ks:
        logging.info("running llm aggregation curve point k=%s", k)
        agg_args = SimpleNamespace(**vars(args))
        agg_args.aggregate_k = k
        agg_args.llm_aggregate = True

        topics, generations, _, comp_idx, answers_df = rfe.aggregate_completions_with_llm(
            ds=ds,
            title_col=title_col,
            topic2cxt=topic2cxt,
            args=agg_args,
            llm=llm,
        )
        if not topics:
            raise ValueError(f"No aggregated generations were produced for k={k}")

        answers_path = out_dir / f"{rfe.build_output_stem(agg_args, ['llmagg', f'aggk{k}'] + (['finalonly'] if args.final_only else []) + ([f's{args.sample}'] if args.sample and args.sample > 0 else []))}_answers.tsv"
        answers_df.to_csv(answers_path, sep="\t", index=False)

        out = fs.get_score(
            topics,
            generations,
            gamma=10,
            knowledge_source="provided",
            all_atomic_facts=None,
            verbose=False,
            completion_labels=comp_idx,
        )

        rows.append({
            "k": k,
            "n_examples": len(topics),
            "score": out["score"],
            "init_score": out.get("init_score"),
            "respond_ratio": out["respond_ratio"],
            "num_facts_per_response": out["num_facts_per_response"],
            "answers_tsv": str(answers_path),
        })

        if args.save_per_k_details:
            detail_path = out_dir / f"{rfe.build_output_stem(agg_args, ['llmagg', f'aggk{k}'] + (['finalonly'] if args.final_only else []) + ([f's{args.sample}'] if args.sample and args.sample > 0 else []))}.tsv"
            pd.DataFrame(out).to_csv(detail_path, sep="\t", index=False)
            rows[-1]["details_tsv"] = str(detail_path)
        if args.save_topic_details:
            topic_detail_frames.append(build_topic_detail_df(topics, out, k))

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / f"{curve_stem}_summary.tsv"
    plot_path = out_dir / f"{curve_stem}.png"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    plot_curve(summary_df, plot_path, title=f"LLM Aggregation Curve: {args.domain}/{args.lang}")
    if args.save_topic_details and topic_detail_frames:
        topic_detail_path = out_dir / f"{curve_stem}_details.tsv"
        pd.concat(topic_detail_frames, ignore_index=True).to_csv(topic_detail_path, sep="\t", index=False)
        print(f"saved_topic_details={topic_detail_path}")

    print(summary_df.to_string(index=False))
    print(f"saved_summary={summary_path}")
    print(f"saved_plot={plot_path}")


if __name__ == "__main__":
    main()
