#!/usr/bin/env python3
"""Analyze self-consistency curves from FactOWL TSV outputs.

Expected input format:
- TSV with a `decisions` column where each cell is a dict-like string:
  {'topic': ..., 'completion_idx': ..., 'atom': ..., 'is_supported': ...}
"""

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to FactOWL TSV")
    ap.add_argument("--output", required=True, help="Output PNG path")
    ap.add_argument("--max-k", type=int, default=99, help="Maximum k to evaluate")
    ap.add_argument("--gamma", type=int, default=10, help="FactOwl gamma penalty for long/short responses")
    ap.add_argument(
        "--expected-topics",
        type=int,
        default=None,
        help="Optional total number of topics; pads missing zero-fact topics as zero-score examples",
    )
    ap.add_argument(
        "--llmagg-summary",
        default=None,
        help="Optional LLM aggregation curve summary TSV from run_llmagg_curve.py",
    )
    ap.add_argument(
        "--llmagg-details",
        default=None,
        help="Optional LLM aggregation per-topic detail TSV for threshold coverage curves",
    )
    ap.add_argument(
        "--fact-majority-summary",
        default=None,
        help="Optional fact-majority curve summary TSV from run_fact_majority_curve.py",
    )
    ap.add_argument(
        "--fact-majority-details",
        default=None,
        help="Optional fact-majority per-topic detail TSV from run_fact_majority_curve.py",
    )
    ap.add_argument(
        "--seqprob-summary",
        default=None,
        help="Optional seqprob selector curve CSV with columns including k and logprob_selector",
    )
    return ap.parse_args()


def load_decisions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "decisions" not in df.columns:
        raise ValueError(f"`decisions` column not found in {path}")
    rec = pd.DataFrame(ast.literal_eval(s) for s in df["decisions"])
    required = {"topic", "is_supported"}
    missing = required - set(rec.columns)
    if missing:
        raise ValueError(f"Missing required decision fields: {sorted(missing)}")
    if "completion_idx" not in rec.columns:
        raise ValueError(
            "No `completion_idx` found. Re-run FactOWL with the updated runner that emits completion indices."
        )
    return rec


def build_matrix(rec: pd.DataFrame, max_k: int, gamma: int) -> pd.DataFrame:
    comp = (
        rec.groupby(["topic", "completion_idx"])["is_supported"]
        .agg(["mean", "count"])
        .reset_index()
    )
    if gamma:
        comp["penalty"] = comp["count"].apply(
            lambda n: 1.0 if n > gamma else np.exp(1 - gamma / n) if n > 0 else 0.0
        )
        comp["score"] = comp["mean"] * comp["penalty"]
    else:
        comp["score"] = comp["mean"]
    mat = comp.pivot(index="topic", columns="completion_idx", values="score")
    mat = mat.reindex(columns=range(max_k))
    return mat


def load_llmagg_summary(path: str, max_k: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"k", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required llmagg summary fields: {sorted(missing)}")
    df = df.sort_values("k").drop_duplicates(subset=["k"], keep="last")
    df = df[df["k"] <= max_k].copy()
    if df.empty:
        raise ValueError(f"No llmagg curve points with k <= {max_k} found in {path}")
    return df


def load_llmagg_details(path: str, max_k: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"topic", "k", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required llmagg detail fields: {sorted(missing)}")
    df = df[df["k"] <= max_k].copy()
    if df.empty:
        raise ValueError(f"No llmagg detail rows with k <= {max_k} found in {path}")
    return df


def load_fact_majority_summary(path: str, max_k: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"k", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required fact-majority summary fields: {sorted(missing)}")
    df = df.sort_values("k").drop_duplicates(subset=["k"], keep="last")
    df = df[df["k"] <= max_k].copy()
    if df.empty:
        raise ValueError(f"No fact-majority curve points with k <= {max_k} found in {path}")
    return df


def load_fact_majority_details(path: str, max_k: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"topic", "k", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required fact-majority detail fields: {sorted(missing)}")
    df = df[df["k"] <= max_k].copy()
    if df.empty:
        raise ValueError(f"No fact-majority detail rows with k <= {max_k} found in {path}")
    return df


def load_seqprob_summary(path: str, max_k: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"k", "logprob_selector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required seqprob summary fields: {sorted(missing)}")
    df = df.sort_values("k").drop_duplicates(subset=["k"], keep="last")
    df = df[df["k"] <= max_k].copy()
    if df.empty:
        raise ValueError(f"No seqprob curve points with k <= {max_k} found in {path}")
    return df


def infer_expected_topics(expected_topics, llmagg_df, fact_majority_df):
    value = expected_topics
    for df in (llmagg_df, fact_majority_df):
        if df is not None and "n_examples" in df.columns:
            current = int(df["n_examples"].max())
            value = max(value or 0, current)
    return value


def pad_missing_topics(mat: pd.DataFrame, expected_topics: int | None) -> pd.DataFrame:
    if expected_topics is not None and expected_topics < mat.shape[0]:
        raise ValueError(
            f"expected_topics={expected_topics} is smaller than observed topics={mat.shape[0]}"
        )
    if expected_topics is not None and expected_topics > mat.shape[0]:
        missing = expected_topics - mat.shape[0]
        filler = pd.DataFrame(
            0.0,
            index=[f"__missing_topic_{i}__" for i in range(missing)],
            columns=mat.columns,
        )
        mat = pd.concat([mat, filler], axis=0)
    return mat


def compute_baseline_curves(rec: pd.DataFrame, max_k: int, gamma: int, expected_topics: int | None = None) -> dict:
    mat = build_matrix(rec, max_k, gamma)
    mat = pad_missing_topics(mat, expected_topics)
    missing_cells = int(mat.isna().sum().sum())
    m = mat.fillna(0.0).values
    ks = np.arange(1, max_k + 1)
    prefix_best = np.maximum.accumulate(m, axis=1).mean(axis=0)
    prefix_mean = (np.cumsum(m, axis=1) / ks).mean(axis=0)
    majority = ((np.cumsum((m > 0.5).astype(int), axis=1) / ks) >= 0.5).astype(float).mean(axis=0)
    thr_curves = {}
    for th in (0.6, 0.7, 0.8, 0.9):
        thr_curves[th] = (np.maximum.accumulate(m, axis=1) >= th).mean(axis=0)
    return {
        "mat": mat,
        "missing_cells": missing_cells,
        "ks": ks,
        "prefix_best": prefix_best,
        "prefix_mean": prefix_mean,
        "majority": majority,
        "thr_curves": thr_curves,
    }


def main():
    args = parse_args()
    rec = load_decisions(args.input)
    llmagg_df = load_llmagg_summary(args.llmagg_summary, args.max_k) if args.llmagg_summary else None
    llmagg_details = load_llmagg_details(args.llmagg_details, args.max_k) if args.llmagg_details else None
    fact_majority_df = (
        load_fact_majority_summary(args.fact_majority_summary, args.max_k)
        if args.fact_majority_summary
        else None
    )
    fact_majority_details = (
        load_fact_majority_details(args.fact_majority_details, args.max_k)
        if args.fact_majority_details
        else None
    )
    seqprob_df = load_seqprob_summary(args.seqprob_summary, args.max_k) if args.seqprob_summary else None
    expected_topics = infer_expected_topics(args.expected_topics, llmagg_df, fact_majority_df)
    curves = compute_baseline_curves(rec, args.max_k, args.gamma, expected_topics)
    mat = curves["mat"]
    missing_cells = curves["missing_cells"]
    ks = curves["ks"]
    prefix_best = curves["prefix_best"]
    prefix_mean = curves["prefix_mean"]
    majority = curves["majority"]
    thr_curves = curves["thr_curves"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, prefix_best, label="Best single completion in prefix-k", linewidth=2, color="purple")
    ax[0].plot(ks, majority, label="Majority over completion success", linewidth=2, color="red")
    ax[0].axhline(prefix_mean[0], label="Single prediction", linewidth=2, linestyle="--", color="black")
    if fact_majority_df is not None:
        ax[0].plot(
            fact_majority_df["k"],
            fact_majority_df["score"],
            label="Fact majority @ k",
            linewidth=2,
            marker="s",
            color="green",
        )
    if llmagg_df is not None:
        ax[0].plot(
            llmagg_df["k"],
            llmagg_df["score"],
            label="LLM aggregation @ k",
            linewidth=2,
            marker="o",
            color="blue",
        )
    if seqprob_df is not None:
        ax[0].plot(
            seqprob_df["k"],
            seqprob_df["logprob_selector"],
            label="Seqprob selector",
            linewidth=2,
            marker="^",
            color="orange",
        )
    ax[0].set_xlabel("k (first k samples)")
    ax[0].set_ylabel("Score")
    ax[0].set_title("SC Curves")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    for th, c in thr_curves.items():
        ax[1].plot(ks, c, label=f">={th}", linewidth=2)
    if fact_majority_details is not None:
        for th in (0.6, 0.7, 0.8, 0.9):
            sub = (
                fact_majority_details.assign(hit=lambda df: (df["score"] >= th).astype(float))
                .groupby("k", as_index=False)["hit"]
                .mean()
            )
            ax[1].plot(
                sub["k"],
                sub["hit"],
                linewidth=2,
                linestyle="--",
                alpha=0.8,
                label=f"Fact maj >= {th}",
            )
    if llmagg_details is not None:
        for th in (0.6, 0.7, 0.8, 0.9):
            sub = (
                llmagg_details.assign(hit=lambda df: (df["score"] >= th).astype(float))
                .groupby("k", as_index=False)["hit"]
                .mean()
            )
            ax[1].plot(
                sub["k"],
                sub["hit"],
                linewidth=2,
                linestyle=":",
                alpha=0.8,
                label=f"LLM agg >= {th}",
            )
    ax[1].set_xlabel("k (first k samples)")
    ax[1].set_ylabel("Coverage over topics")
    ax[1].set_title("Coverage Threshold Curves")
    ax[1].grid(alpha=0.3)
    ax[1].legend(title="Best score threshold", fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=220)

    print(f"topics={mat.shape[0]} k={args.max_k} missing_cells={missing_cells}")
    print(f"best@1={prefix_best[0]:.4f} best@{args.max_k}={prefix_best[-1]:.4f}")
    print(f"mean@1={prefix_mean[0]:.4f} mean@{args.max_k}={prefix_mean[-1]:.4f}")
    print(f"majority@1={majority[0]:.4f} majority@{args.max_k}={majority[-1]:.4f}")
    print(f"single_prediction={prefix_mean[0]:.4f}")
    if fact_majority_df is not None:
        print(f"fact_majority@{int(fact_majority_df['k'].iloc[0])}={fact_majority_df['score'].iloc[0]:.4f}")
        print(f"fact_majority@{int(fact_majority_df['k'].iloc[-1])}={fact_majority_df['score'].iloc[-1]:.4f}")
    if llmagg_df is not None:
        print(f"llmagg@{int(llmagg_df['k'].iloc[0])}={llmagg_df['score'].iloc[0]:.4f}")
        print(f"llmagg@{int(llmagg_df['k'].iloc[-1])}={llmagg_df['score'].iloc[-1]:.4f}")
    if seqprob_df is not None:
        print(f"seqprob@{int(seqprob_df['k'].iloc[0])}={seqprob_df['logprob_selector'].iloc[0]:.4f}")
        print(f"seqprob@{int(seqprob_df['k'].iloc[-1])}={seqprob_df['logprob_selector'].iloc[-1]:.4f}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
