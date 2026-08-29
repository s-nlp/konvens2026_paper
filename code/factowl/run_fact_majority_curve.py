#!/usr/bin/env python3
"""Build fact-level majority-vote SC curves from existing FactOwl decisions.

This reuses the saved atomic-fact extraction and support labels from an
`allcompletions` FactOwl TSV. For each prefix-k, it votes over atomic facts
across the first k completions for each topic, then scores the selected fact
set directly without regenerating text or re-running verification.
"""

import argparse
import ast
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from factowl.factscorer_sped_up_vllm import calculate_score_from_decisions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="FactOwl TSV with a decisions column")
    p.add_argument("--output-dir", default="/workspace/longft/experiments/factowl_eval/")
    p.add_argument("--output-stem", default=None, help="Override output stem")
    p.add_argument("--max-k", type=int, default=99)
    p.add_argument(
        "--k-list",
        default=None,
        help="Optional comma-separated prefix-k values. Defaults to 1..max-k.",
    )
    p.add_argument(
        "--min-votes",
        default="majority",
        help="Vote threshold: 'majority', 'all', or an integer count",
    )
    p.add_argument(
        "--expected-topics",
        type=int,
        default=None,
        help="Optional total number of topics; pads missing zero-fact topics as zero-score examples",
    )
    return p.parse_args()


def parse_k_list(k_list: str | None, max_k: int) -> list[int]:
    if not k_list:
        return list(range(1, max_k + 1))
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
    return sorted(dict.fromkeys(v for v in ks if v <= max_k))


def infer_output_stem(path: Path) -> str:
    stem = path.stem
    if stem.startswith("pred_") and stem.endswith("_allcompletions_finalonly"):
        return stem.replace("_allcompletions_finalonly", "_factmajoritycurve_finalonly")
    return f"{stem}_factmajoritycurve"


def normalize_atom(atom: str) -> str:
    if not isinstance(atom, str):
        return ""
    atom = atom.strip().lower()
    atom = re.sub(r"\s+", " ", atom)
    atom = re.sub(r"[^\w\s\u4e00-\u9fff]", "", atom)
    return atom.strip()


def threshold_for_k(spec: str, k: int) -> int:
    if spec == "majority":
        return max(1, math.ceil(k / 2))
    if spec == "all":
        return k
    value = int(spec)
    if value <= 0:
        raise ValueError(f"min-votes must be positive, got {value}")
    return value


def load_records(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "decisions" not in df.columns:
        raise ValueError(f"`decisions` column not found in {path}")
    rec = pd.DataFrame(ast.literal_eval(s) for s in df["decisions"])
    required = {"topic", "completion_idx", "atom", "is_supported"}
    missing = required - set(rec.columns)
    if missing:
        raise ValueError(f"Missing required decision fields: {sorted(missing)}")
    rec["norm_atom"] = rec["atom"].map(normalize_atom)
    rec = rec[rec["norm_atom"] != ""].copy()
    return rec


def build_topic_completion_map(rec: pd.DataFrame, max_k: int):
    topic_map = {}
    grouped = rec.groupby(["topic", "completion_idx"], sort=False)
    for (topic, completion_idx), group in grouped:
        if completion_idx >= max_k:
            continue
        fact_map = {}
        for row in group.itertuples(index=False):
            entry = fact_map.setdefault(
                row.norm_atom,
                {
                    "atom": row.atom,
                    "support_sum": 0,
                    "support_count": 0,
                },
            )
            entry["support_sum"] += int(bool(row.is_supported))
            entry["support_count"] += 1
        topic_map.setdefault(topic, {})[int(completion_idx)] = fact_map
    return topic_map


def aggregate_topic(topic: str, completion_facts: dict, k: int, min_votes: int):
    votes = {}
    for completion_idx in range(k):
        for norm_atom, payload in completion_facts.get(completion_idx, {}).items():
            entry = votes.setdefault(
                norm_atom,
                {
                    "topic": topic,
                    "atom": payload["atom"],
                    "vote_count": 0,
                    "support_sum": 0,
                    "support_count": 0,
                },
            )
            entry["vote_count"] += 1
            entry["support_sum"] += payload["support_sum"]
            entry["support_count"] += payload["support_count"]

    selected = []
    for payload in votes.values():
        if payload["vote_count"] < min_votes:
            continue
        is_supported = payload["support_sum"] / payload["support_count"] >= 0.5
        selected.append(
            {
                "topic": topic,
                "completion_idx": f"factmajority_k{k}",
                "atom": payload["atom"],
                "is_supported": bool(is_supported),
                "vote_count": payload["vote_count"],
            }
        )
    selected.sort(key=lambda item: (-item["vote_count"], item["atom"]))
    return selected


def calculate_topic_score(selected_decisions: list[dict], gamma: int = 10):
    atomic_facts = [d["atom"] for d in selected_decisions] if selected_decisions else None
    out = calculate_score_from_decisions(
        all_atomic_facts=[atomic_facts],
        decisions=[selected_decisions],
        gamma=gamma,
    )
    return {
        "score": out["score"],
        "init_score": out.get("init_score", out["score"]),
        "respond_ratio": out["respond_ratio"],
        "num_facts_per_response": out["num_facts_per_response"],
    }


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


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = parse_k_list(args.k_list, args.max_k)
    stem = args.output_stem or infer_output_stem(input_path)
    summary_path = out_dir / f"{stem}_summary.tsv"
    details_path = out_dir / f"{stem}_details.tsv"
    plot_path = out_dir / f"{stem}.png"

    rec = load_records(input_path)
    topic_map = build_topic_completion_map(rec, max_k=max(ks))
    topics = sorted(topic_map.keys())
    if args.expected_topics is not None and args.expected_topics < len(topics):
        raise ValueError(
            f"expected_topics={args.expected_topics} is smaller than observed topics={len(topics)}"
        )
    if args.expected_topics is not None and args.expected_topics > len(topics):
        missing = args.expected_topics - len(topics)
        for idx in range(missing):
            topic_map[f"__missing_topic_{idx}__"] = {}
        topics = sorted(topic_map.keys())

    summary_rows = []
    detail_rows = []
    for k in ks:
        min_votes = threshold_for_k(args.min_votes, k)
        all_atomic_facts = []
        all_decisions = []
        for topic in topics:
            selected = aggregate_topic(topic, topic_map[topic], k=k, min_votes=min_votes)
            topic_out = calculate_topic_score(selected, gamma=10)
            detail_rows.append(
                {
                    "topic": topic,
                    "k": k,
                    "min_votes": min_votes,
                    "score": topic_out["score"],
                    "init_score": topic_out["init_score"],
                    "respond_ratio": topic_out["respond_ratio"],
                    "num_facts_per_response": topic_out["num_facts_per_response"],
                }
            )
            all_atomic_facts.append([d["atom"] for d in selected] if selected else None)
            all_decisions.append(selected)

        out = calculate_score_from_decisions(
            all_atomic_facts=all_atomic_facts,
            decisions=all_decisions,
            gamma=10,
        )
        summary_rows.append(
            {
                "k": k,
                "n_examples": len(topics),
                "min_votes": min_votes,
                "score": out["score"],
                "init_score": out.get("init_score"),
                "respond_ratio": out["respond_ratio"],
                "num_facts_per_response": out["num_facts_per_response"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.DataFrame(detail_rows)
    summary_df.to_csv(summary_path, sep="\t", index=False)
    details_df.to_csv(details_path, sep="\t", index=False)
    plot_curve(summary_df, plot_path, title=f"Fact Majority Curve: {input_path.stem}")

    print(summary_df.to_string(index=False))
    print(f"saved_summary={summary_path}")
    print(f"saved_details={details_path}")
    print(f"saved_plot={plot_path}")


if __name__ == "__main__":
    main()
