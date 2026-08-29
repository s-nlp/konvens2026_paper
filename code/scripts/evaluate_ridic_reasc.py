#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from ReASC.reasc_core import CalibrationManager, ReASCMetrics, ReASCStopper


FACTOWL_TSVS = {
    ("rivers", "en"): "experiments/factowl_eval/pred_rivers_k99_allcompletions_finalonly.tsv",
    ("cars", "en"): "experiments/factowl_eval/pred_cars_k99_allcompletions_finalonly.tsv",
    ("disasters", "en"): "experiments/factowl_eval/pred_disasters_k99_allcompletions_finalonly.tsv",
    ("rivers", "zh"): "experiments/factowl_eval/pred_rivers_zh_k99_allcompletions_finalonly.tsv",
    ("cars", "zh"): "experiments/factowl_eval/pred_cars_zh_k99_allcompletions_finalonly.tsv",
    ("disasters", "zh"): "experiments/factowl_eval/pred_disasters_zh_k99_allcompletions_finalonly.tsv",
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ReASC-style adaptive routing on RiDiC with FactOwl scores")
    p.add_argument("--input", required=True, help="Parquet with all_completions + all_logprobs from score_ridic_logprobs.py")
    p.add_argument("--domain", choices=["rivers", "cars", "disasters"], required=True)
    p.add_argument("--lang", choices=["en", "zh"], required=True)
    p.add_argument("--factowl-tsv", default=None, help="FactOwl all-completions TSV with per-completion scores")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-budget", type=int, default=99)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--target-accuracy", type=float, default=0.95)
    p.add_argument("--p-target", type=float, default=0.90)
    p.add_argument("--lambda-param", type=float, default=0.7)
    p.add_argument("--gate-threshold", type=float, default=None)
    p.add_argument("--verbose-rows", type=int, default=0, help="Log selection diagnostics for the first N rows")
    return p.parse_args()


def setup_logger(output_dir: Path, stem: str) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{stem}_{ts}.log"
    logger = logging.getLogger(stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info("log_path=%s", log_path)
    return logger


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def extract_topic_from_question(question: str, lang: str) -> str:
    if not isinstance(question, str):
        return ""
    q = question.strip()
    if lang == "zh":
        prefix = "请用一个段落告诉我你对"
        suffix = "的了解。"
        if q.startswith(prefix) and q.endswith(suffix):
            return q[len(prefix) : -len(suffix)].strip()
        return q
    prefix = "In a paragraph, could you tell me what you know about "
    suffix = "?"
    if q.startswith(prefix) and q.endswith(suffix):
        return q[len(prefix) : -len(suffix)].strip()
    m = re.search(r"about\s+(.+?)\?$", q)
    if m:
        return m.group(1).strip()
    return q


def extract_final_segment(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for marker in ["Final Answer:", "Answer:", "最终答案：", "最终答案:"]:
        if marker.lower() in text.lower():
            pos = text.lower().find(marker.lower())
            return text[pos + len(marker):].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def extract_completion_answer(text: str, lang: str) -> str:
    if not isinstance(text, str):
        return ""
    if lang == "zh":
        for marker in ["最终答案：", "最终答案:"]:
            if marker in text:
                return text.split(marker, 1)[1].strip()
    else:
        for marker in ["Final Answer:"]:
            if marker in text:
                return text.split(marker, 1)[1].strip()
    return extract_final_segment(text).strip()


def find_factowl_tsv(domain: str, lang: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return Path(FACTOWL_TSVS[(domain, lang)])


def load_factowl_completion_scores(path: Path) -> dict[tuple[str, int], float]:
    df = pd.read_csv(path, sep="\t")
    if "decisions" not in df.columns:
        raise ValueError(f"Missing required columns in {path}")
    grouped: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for _, row in df.iterrows():
        try:
            dec = ast.literal_eval(row["decisions"])
        except Exception:
            continue
        if isinstance(dec, dict):
            topic = dec.get("topic")
            comp_idx = dec.get("completion_idx")
            if topic is None or comp_idx is None:
                continue
            grouped[(str(topic), int(comp_idx))].append(bool(dec.get("is_supported")))

    score_map: dict[tuple[str, int], float] = {}
    gamma = 10
    for key, supports in grouped.items():
        total = len(supports)
        supported = sum(supports)
        support_rate = supported / total if total else 0.0
        if total == 0:
            penalty = 0.0
        elif total > gamma:
            penalty = 1.0
        else:
            penalty = float(np.exp(1 - gamma / total))
        score_map[key] = float(penalty * support_rate)
    return score_map


def simulate_reasc_row(row: pd.Series, stopper: ReASCStopper, conf_stats: tuple[float, float], gate_threshold: float, max_budget: int):
    extracted = row["extracted_answers"][:max_budget]
    logprobs = row["all_logprobs"][:max_budget]
    confidences = [ReASCMetrics.compute_confidence(lp, window_size=row["_window_size"]) for lp in logprobs]
    usable = min(len(extracted), len(confidences))
    extracted = extracted[:usable]
    confidences = confidences[:usable]
    if usable <= 1:
        raise ValueError(f"ReASC requires more than one candidate, got usable={usable} for topic={row.get('topic', '')}")

    if confidences and confidences[0] >= gate_threshold:
        return {
            "final_answer": extracted[0],
            "samples_used": 1,
            "stage": 1,
            "exhausted": False,
            "confidences": confidences,
            "stop_prob": None,
        }

    weighted_votes = defaultdict(float)
    for idx, ans in enumerate(extracted):
        weight = stopper.compute_weight(confidences[idx], conf_stats)
        weighted_votes[ans] += weight
        if idx + 1 < 2:
            continue
        ranked = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)
        v1 = ranked[0][1]
        v2 = ranked[1][1] if len(ranked) > 1 else 0.0
        stop_prob = stopper.get_beta_confidence(v1, v2)
        if stop_prob >= stopper.threshold:
            return {
                "final_answer": ranked[0][0],
                "samples_used": idx + 1,
                "stage": 2,
                "exhausted": False,
                "stop_prob": float(stop_prob),
                "confidences": confidences,
            }

    best = max(weighted_votes.items(), key=lambda x: x[1])[0] if weighted_votes else ""
    return {
        "final_answer": best,
        "samples_used": min(max_budget, len(extracted)),
        "stage": 2,
        "exhausted": True,
        "confidences": confidences,
        "stop_prob": None,
    }


def completion_index_for_answer(extracted_answers: list[str], answer: str) -> int:
    norm_target = normalize_text(answer)
    for idx, cand in enumerate(extracted_answers):
        if normalize_text(cand) == norm_target:
            return idx
    return 0


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir, "evaluate_ridic_reasc")
    logger.info("input=%s domain=%s lang=%s", args.input, args.domain, args.lang)

    df = pd.read_parquet(args.input).copy()
    if "question" not in df.columns:
        raise ValueError("Input parquet must contain question column")

    df["topic"] = [extract_topic_from_question(q, args.lang) for q in df["question"]]
    df["extracted_answers"] = [
        [extract_completion_answer(c, args.lang) for c in comps[: args.max_budget]]
        for comps in df["all_completions"]
    ]
    df["_window_size"] = args.window_size

    factowl_tsv = find_factowl_tsv(args.domain, args.lang, args.factowl_tsv)
    score_map = load_factowl_completion_scores(factowl_tsv)
    logger.info("loaded_factowl_scores=%s from %s", len(score_map), factowl_tsv)

    first_confs = []
    all_confs = []
    for lps in df["all_logprobs"]:
        seq_confs = [ReASCMetrics.compute_confidence(lp, window_size=args.window_size) for lp in lps[: args.max_budget]]
        all_confs.extend(seq_confs)
        if seq_confs:
            first_confs.append(seq_confs[0])

    conf_stats = (float(np.mean(all_confs)), float(np.std(all_confs))) if all_confs else (0.0, 1.0)
    if args.gate_threshold is None:
        if len(first_confs) >= 2:
            calibrator = CalibrationManager()
            gate_threshold = float(calibrator.online_calibration(first_confs, p_target=args.p_target))
        elif first_confs:
            gate_threshold = float(first_confs[0])
        else:
            gate_threshold = -1.0
    else:
        gate_threshold = float(args.gate_threshold)
    stopper = ReASCStopper(target_accuracy=args.target_accuracy, lambda_param=args.lambda_param)

    logger.info("confidence_stats mean=%.6f std=%.6f", conf_stats[0], conf_stats[1])
    logger.info("gate_threshold=%.6f target_accuracy=%.4f lambda=%.4f", gate_threshold, args.target_accuracy, args.lambda_param)

    rows_out = []
    for row_idx, row in df.iterrows():
        result = simulate_reasc_row(row, stopper, conf_stats, gate_threshold, args.max_budget)
        topic = row["topic"]
        extracted = row["extracted_answers"][: args.max_budget]
        single_idx = 0 if extracted else -1
        mv_ans = Counter(extracted).most_common(1)[0][0] if extracted else ""
        mv_idx = completion_index_for_answer(extracted, mv_ans) if extracted else -1
        reasc_idx = completion_index_for_answer(extracted, result["final_answer"]) if extracted else -1
        selected_equals_first = bool(extracted) and normalize_text(result["final_answer"]) == normalize_text(extracted[0])
        n_candidates = len(extracted)

        rows_out.append(
            {
                "topic": topic,
                "question": row["question"],
                "source_row_idx": row.get("source_row_idx", row_idx),
                "n_candidates": n_candidates,
                "single_completion_idx": single_idx,
                "mv_completion_idx": mv_idx,
                "reasc_completion_idx": reasc_idx,
                "single_score": score_map.get((topic, single_idx), np.nan),
                "mv_score": score_map.get((topic, mv_idx), np.nan),
                "reasc_score": score_map.get((topic, reasc_idx), np.nan),
                "samples_used": result["samples_used"],
                "stage": result["stage"],
                "exhausted": result["exhausted"],
                "stop_prob": result.get("stop_prob"),
                "gate_threshold": gate_threshold,
                "selected_equals_first_text": selected_equals_first,
            }
        )

        if row_idx < args.verbose_rows:
            logger.info(
                "trace row=%s topic=%s candidates=%s single_idx=%s reasc_idx=%s same_as_first=%s samples_used=%s stage=%s stop_prob=%s first_conf=%.6f selected_conf=%.6f",
                row_idx,
                topic,
                n_candidates,
                single_idx,
                reasc_idx,
                selected_equals_first,
                result["samples_used"],
                result["stage"],
                result.get("stop_prob"),
                result["confidences"][0],
                result["confidences"][reasc_idx] if 0 <= reasc_idx < len(result["confidences"]) else float("nan"),
            )

    out_df = pd.DataFrame(rows_out)
    summary = {
        "domain": args.domain,
        "lang": args.lang,
        "rows": int(len(out_df)),
        "scored_rows": int(out_df["reasc_score"].notna().sum()),
        "score_coverage": float(out_df["reasc_score"].notna().mean()),
        "single_score": float(out_df["single_score"].mean()),
        "mv_score": float(out_df["mv_score"].mean()),
        "reasc_score": float(out_df["reasc_score"].mean()),
        "avg_samples_used": float(out_df["samples_used"].mean()),
        "stage1_rate": float((out_df["stage"] == 1).mean()),
        "exhausted_rate": float(out_df["exhausted"].mean()),
        "selected_equals_first_rate": float(out_df["selected_equals_first_text"].mean()),
        "gate_threshold": gate_threshold,
        "confidence_mean": conf_stats[0],
        "confidence_std": conf_stats[1],
        "factowl_tsv": str(factowl_tsv),
    }

    pred_path = output_dir / "reasc_predictions.tsv"
    summary_path = output_dir / "reasc_summary.json"
    out_df.to_csv(pred_path, sep="\t", index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("summary=%s", json.dumps(summary, ensure_ascii=False))
    logger.info("saved_predictions=%s", pred_path)
    logger.info("saved_summary=%s", summary_path)


if __name__ == "__main__":
    main()
