#!/usr/bin/env python3
"""Build paper-ready figures for the RiDiC / LLMAgg paper.

Outputs:
  - prefix_scaling_en_zh.pdf / .png
  - adaptive_tradeoff_scatter.pdf / .png
  - adaptive_feature_importance.pdf / .png

Data sources are restricted to the TSVs listed in the task description.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


EXPECTED_KS = [1, 2, 4, 8, 16, 20]
DOMAINS = ["rivers", "cars", "disasters"]
LANGS = ["en", "zh"]
RETENTION_TARGETS = [0.95, 0.99]
DOMAIN_STYLES = {
    "rivers": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "cars": {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
    "disasters": {"color": "#2ca02c", "linestyle": ":", "marker": "^"},
}
TARGET_STYLES = {
    0.95: {"color": "#1f77b4", "marker": "o"},
    0.99: {"color": "#d62728", "marker": "s"},
}
LANG_FILL = {"en": "full", "zh": "none"}
LANG_EDGE = {"en": "#111111", "zh": "#111111"}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "savefig.transparent": False,
        }
    )


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def readable_feature_name(name: str) -> str:
    mapping = {
        "q_len_chars": "len chars",
        "q_len_words": "len words",
        "q_avg_word_length": "avg word len",
        "q_num_digits": "num digits",
        "q_num_punct": "num punct",
        "q_has_year": "has year",
        "q_is_chinese": "is Chinese",
        "starts_with_what": "starts w/ what",
        "starts_with_where": "starts w/ where",
        "starts_with_when": "starts w/ when",
        "starts_with_who": "starts w/ who",
        "starts_with_which": "starts w/ which",
        "contains_parentheses": "has parens",
        "contains_comma": "has comma",
        "contains_hyphen": "has hyphen",
        "domain_is_rivers": "domain=rivers",
        "domain_is_cars": "domain=cars",
        "domain_is_disasters": "domain=disasters",
        "baseline_k": "baseline k",
        "baseline_k_log2": "log2(k)",
        "lang_is_zh": "lang=zh",
        "lang_is_en": "lang=en",
    }
    return mapping.get(name, name)


def ensure_prefix_sanity(df: pd.DataFrame) -> None:
    ks = sorted(df["k"].astype(int).unique().tolist())
    if ks != EXPECTED_KS:
        raise ValueError(f"Unexpected k values: {ks}")
    combos = set(zip(df["lang"], df["domain"]))
    expected = {(lang, domain) for lang in LANGS for domain in DOMAINS}
    if combos != expected:
        raise ValueError(f"Unexpected lang/domain coverage: {sorted(combos)}")


def ensure_scatter_sanity(df: pd.DataFrame) -> None:
    if len(df) != 12:
        raise ValueError(f"Expected 12 adaptive points, got {len(df)}")
    combos = set(zip(df["lang"], df["domain"], df["retention_target"]))
    expected = {(lang, domain, ret) for lang in LANGS for domain in DOMAINS for ret in RETENTION_TARGETS}
    if combos != expected:
        raise ValueError(f"Unexpected adaptive coverage: {sorted(combos)}")


def plot_prefix_scaling(input_dir: Path, output_dir: Path) -> None:
    prefix = read_tsv(input_dir / "llmagg_prefix_curve_k1_20.tsv")
    boot = read_tsv(input_dir / "bootstrap_llmagg_prefix_curve.tsv")
    ensure_prefix_sanity(prefix)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    for ax, lang, title in zip(axes, LANGS, ["English", "Chinese"]):
        for domain in DOMAINS:
            sub = prefix[(prefix["lang"] == lang) & (prefix["domain"] == domain)].copy()
            sub["k"] = sub["k"].astype(int)
            sub = sub.sort_values("k")
            boot_sub = boot[(boot["lang"] == lang) & (boot["domain"] == domain)].copy()
            boot_sub["k"] = boot_sub["k"].astype(int)
            boot_sub = boot_sub.sort_values("k")
            if list(sub["k"]) != EXPECTED_KS or list(boot_sub["k"]) != EXPECTED_KS:
                raise ValueError(f"Prefix k mismatch for {lang}/{domain}")

            y = sub["score"].to_numpy(dtype=float)
            lo = boot_sub["lower"].to_numpy(dtype=float)
            hi = boot_sub["upper"].to_numpy(dtype=float)
            yerr = np.vstack([y - lo, hi - y])
            st = DOMAIN_STYLES[domain]
            ax.errorbar(
                sub["k"],
                y,
                yerr=yerr,
                fmt=st["marker"],
                linestyle=st["linestyle"],
                color=st["color"],
                capsize=2.0,
                elinewidth=0.9,
                linewidth=2.0,
                markersize=4.6,
                markerfacecolor="white" if domain != "rivers" else "white",
                markeredgewidth=0.9,
                label=domain,
            )

        ax.set_title(title, pad=4)
        ax.set_xlabel(r"Number of samples $k$")
        ax.set_xticks(EXPECTED_KS)
        ax.set_xlim(0.8, 20.4)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.45)
        ax.grid(False, axis="x")

    axes[0].set_ylabel("Factual precision")

    legend_domains = ["cars", "disasters", "rivers"]
    handles = [
        Line2D([], [], color=DOMAIN_STYLES[d]["color"], linestyle=DOMAIN_STYLES[d]["linestyle"], marker=DOMAIN_STYLES[d]["marker"], markersize=5.0, markerfacecolor="white", markeredgewidth=0.9, label=d)
        for d in legend_domains
    ]
    axes[0].legend(
        handles,
        legend_domains,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.85",
        handlelength=2.6,
        borderpad=0.6,
        labelspacing=0.5,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.98))

    fig.savefig(output_dir / "prefix_scaling_en_zh.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "prefix_scaling_en_zh.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_adaptive_tradeoff(input_dir: Path, output_dir: Path) -> None:
    comp = read_tsv(input_dir / "adaptive_retention_sweep_compact.tsv")
    boot = read_tsv(input_dir / "bootstrap_adaptive_tradeoff.tsv")
    ensure_scatter_sanity(comp)

    comp = comp.copy()
    comp = comp.rename(
        columns={
            "retention_target": "retention",
            "avg_compute_savings": "compute_savings",
            "avg_score_retention": "score_retention",
            "coverage_retention": "coverage",
        }
    )

    # One row per point, with bootstrap error bars for the same metric names.
    merged = comp.merge(
        boot[boot["metric"].isin(["compute_savings", "score_retention"])],
        left_on=["lang", "domain", "retention"],
        right_on=["lang", "domain", "retention_target"],
        how="left",
        suffixes=("", "_boot"),
    )
    if merged.empty:
        raise ValueError("Adaptive merge produced no rows")

    categories = [
        ("en", "cars"),
        ("en", "disasters"),
        ("en", "rivers"),
        ("zh", "cars"),
        ("zh", "disasters"),
        ("zh", "rivers"),
    ]
    y_positions = np.arange(len(categories))[::-1]
    y_map = {cat: y for cat, y in zip(categories, y_positions)}
    y_offsets = {0.95: 0.16, 0.99: -0.16}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharey=True)
    panel_specs = [
        ("compute_savings", "Compute savings", (0.0, 0.33), None),
        ("score_retention", "Score retention", (0.98, 1.03), 1.0),
    ]

    for ax, (metric, xlabel, xlim, vline) in zip(axes, panel_specs):
        for lang, domain in categories:
            for ret in RETENTION_TARGETS:
                row = merged[
                    (merged["lang"] == lang)
                    & (merged["domain"] == domain)
                    & (merged["retention"] == ret)
                ].iloc[0]
                boot_rows = boot[
                    (boot["lang"] == lang)
                    & (boot["domain"] == domain)
                    & (boot["retention_target"] == ret)
                ]
                if metric == "compute_savings":
                    boot_row = boot_rows[boot_rows["metric"] == "compute_savings"].iloc[0]
                    x = float(row["compute_savings"])
                    xerr = np.array([[x - float(boot_row["lower"])], [float(boot_row["upper"]) - x]])
                else:
                    boot_row = boot_rows[boot_rows["metric"] == "score_retention"].iloc[0]
                    x = float(row["score_retention"])
                    xerr = np.array([[x - float(boot_row["lower"])], [float(boot_row["upper"]) - x]])
                y = y_map[(lang, domain)] + y_offsets[ret]
                st = TARGET_STYLES[ret]
                ax.errorbar(
                    x,
                    y,
                    xerr=xerr,
                    yerr=None,
                    fmt=st["marker"],
                    linestyle="none",
                    color=st["color"],
                    ecolor="#444444",
                    elinewidth=0.9,
                    capsize=2.0,
                    markersize=6.0,
                    markerfacecolor=st["color"] if LANG_FILL[lang] == "full" else "white",
                    markeredgewidth=1.0,
                    markeredgecolor=LANG_EDGE[lang],
                    alpha=0.98,
                )

        if vline is not None:
            ax.axvline(vline, color="0.25", linestyle="--", linewidth=1.0, alpha=0.85)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.35)
        ax.grid(False, axis="y")

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(["EN-cars", "EN-disasters", "EN-rivers", "ZH-cars", "ZH-disasters", "ZH-rivers"])

    target_handles = [
        Line2D([], [], marker="o", color=TARGET_STYLES[0.95]["color"], linestyle="none", markerfacecolor=TARGET_STYLES[0.95]["color"], markeredgecolor="#111111", markersize=6, label="Target 0.95"),
        Line2D([], [], marker="s", color=TARGET_STYLES[0.99]["color"], linestyle="none", markerfacecolor=TARGET_STYLES[0.99]["color"], markeredgecolor="#111111", markersize=6, label="Target 0.99"),
    ]
    lang_handles = [
        Line2D([], [], marker="o", color="#111111", linestyle="none", markerfacecolor="#111111", markersize=6, label="EN"),
        Line2D([], [], marker="o", color="#111111", linestyle="none", markerfacecolor="white", markeredgecolor="#111111", markersize=6, label="ZH"),
    ]
    legend1 = axes[1].legend(handles=target_handles, loc="lower right", frameon=False)
    axes[1].add_artist(legend1)
    axes[1].legend(handles=lang_handles, loc="lower left", frameon=False, ncol=1)
    fig.tight_layout()

    fig.savefig(output_dir / "adaptive_tradeoff_scatter.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "adaptive_tradeoff_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(input_dir: Path, output_dir: Path) -> None:
    df = read_tsv(input_dir / "adaptive_feature_importance.tsv")
    targets = RETENTION_TARGETS
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.9), sharex=True)

    for ax, target in zip(axes, targets):
        sub = df[df["retention_target"] == target].copy()
        sub = sub.sort_values("mean_importance", ascending=False).head(10).iloc[::-1]
        y = np.arange(len(sub))
        labels = [readable_feature_name(v) for v in sub["feature"].tolist()]
        x = sub["mean_importance"].to_numpy(dtype=float)
        xerr = sub["std_importance"].to_numpy(dtype=float)
        color = TARGET_STYLES[target]["color"]

        ax.barh(
            y,
            x,
            xerr=xerr,
            color=color,
            edgecolor="#111111",
            linewidth=0.8,
            hatch="//" if target == 0.95 else "\\\\",
            error_kw={"elinewidth": 0.9, "ecolor": "#111111", "capsize": 2.0},
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(f"Retention {target:.2f}")
        ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.4)
        ax.grid(False, axis="y")
        ax.set_xlim(0.0, max(0.35, float((x + xerr).max()) * 1.1))

    axes[0].set_xlabel("Mean feature importance")
    axes[1].set_xlabel("Mean feature importance")
    fig.supylabel("Feature")
    fig.tight_layout()

    fig.savefig(output_dir / "adaptive_feature_importance.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "adaptive_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


from _root import find_experiment_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper figures from paper_prep TSVs.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=find_experiment_root() / "paper_prep" / "tables",
        help="Directory containing the TSV inputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=find_experiment_root() / "paper_prep" / "figures",
        help="Directory to write PDF/PNG outputs.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_prefix_scaling(args.input_dir, args.output_dir)
    plot_adaptive_tradeoff(args.input_dir, args.output_dir)
    plot_feature_importance(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
