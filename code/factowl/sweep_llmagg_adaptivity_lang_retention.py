#!/usr/bin/env python3
"""Sweep retention targets for language-level llmagg adaptivity models."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import train_llmagg_adaptivity_lang as tll


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", default="en,zh")
    p.add_argument("--detail-dir", default="/workspace/longft/experiments/factowl_eval")
    p.add_argument("--output-dir", default="/workspace/longft/experiments/factowl_eval/adaptivity_lang")
    p.add_argument("--retentions", default="0.90,0.92,0.95,0.97,0.99")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def parse_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def parse_retentions(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def evaluate_lang(lang: str, detail_dir: Path, retention: float, n_splits: int, test_size: float, seed: int) -> pd.DataFrame:
    df, detail_df, feature_df = tll.build_dataset(lang, detail_dir, retention)
    strat_labels = tll.make_strat_labels(df)
    splitter = tll.StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    all_preds = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(feature_df.values, strat_labels), start=1):
        _, pred_df, _ = tll.evaluate_split(df, detail_df, feature_df, train_idx, test_idx, seed=seed + fold_id)
        pred_df["fold"] = fold_id
        all_preds.append(pred_df)
    pred_df = pd.concat(all_preds, ignore_index=True)
    curve_df = (
        pred_df[pred_df["baseline_k"] == 99]
        .groupby("domain", as_index=False)
        .agg(
            avg_compute_savings=("compute_savings", "mean"),
            avg_score_retention=("score_retention", "mean"),
            coverage_retention=("score_retention", lambda s: float((s >= retention).mean())),
            n_eval=("topic", "count"),
        )
    )
    curve_df["lang"] = lang
    curve_df["retention_target"] = retention
    return curve_df


def plot_lang(df: pd.DataFrame, out_path: Path, lang: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"rivers": "tab:blue", "cars": "tab:green", "disasters": "tab:red"}
    for domain in tll.DOMAINS:
        sub = df[df["domain"] == domain].sort_values("retention_target")
        ax.plot(
            sub["retention_target"] * 100.0,
            sub["avg_compute_savings"] * 100.0,
            marker="o",
            linewidth=2,
            label=domain,
            color=colors[domain],
        )
    ax.set_xlabel("Retention target (%)")
    ax.set_ylabel("Realized compute savings (%)")
    ax.set_title(f"{lang.upper()} adaptivity: savings vs retention target")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)


def main():
    args = parse_args()
    langs = parse_list(args.langs)
    retentions = parse_retentions(args.retentions)
    detail_dir = Path(args.detail_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for lang in langs:
        for retention in retentions:
            all_rows.append(evaluate_lang(lang, detail_dir, retention, args.n_splits, args.test_size, args.seed))
    summary_df = pd.concat(all_rows, ignore_index=True)
    summary_path = out_dir / "llmagg_adaptivity_lang_retention_sweep.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(summary_df.to_string(index=False))
    print(f"saved_summary={summary_path}")
    for lang in langs:
        plot_path = out_dir / f"{lang}_llmagg_adaptivity_retention_sweep.png"
        plot_lang(summary_df[summary_df["lang"] == lang], plot_path, lang)
        print(f"saved_plot={plot_path}")


if __name__ == "__main__":
    main()
