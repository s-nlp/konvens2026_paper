#!/usr/bin/env python3
"""Create a 3x2 grid of SC comparison curves across domains and languages."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analyze_sc_curves as asc


DOMAINS = ["rivers", "cars", "disasters"]
LANGS = ["en", "zh"]
PANEL_LABELS = ["a", "b", "c", "d", "e", "f"]
PLOT_KS = [1, 2, 4, 8, 16, 20]
EXPECTED_TOPICS = {
    ("en", "rivers"): 1000,
    ("en", "cars"): 997,
    ("en", "disasters"): 1000,
    ("zh", "rivers"): 723,
    ("zh", "cars"): 400,
    ("zh", "disasters"): 723,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--input-dir", default="/workspace/longft/experiments/factowl_eval")
    p.add_argument("--max-k", type=int, default=99)
    p.add_argument("--gamma", type=int, default=10)
    p.add_argument("--zoom-k", type=int, default=20)
    return p.parse_args()


def baseline_path(input_dir: Path, domain: str, lang: str) -> Path:
    if lang == "en":
        return input_dir / f"pred_{domain}_k99_allcompletions_finalonly.tsv"
    return input_dir / f"pred_{domain}_zh_k99_allcompletions_finalonly.tsv"


def llmagg_summary_path(input_dir: Path, domain: str, lang: str) -> Path:
    return input_dir / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_summary.tsv"


def fact_majority_summary_path(input_dir: Path, domain: str, lang: str) -> Path:
    if lang == "en":
        return input_dir / f"pred_{domain}_k99_factmajoritycurve_finalonly_summary.tsv"
    return input_dir / f"pred_{domain}_zh_k99_factmajoritycurve_finalonly_summary.tsv"


def seqprob_summary_path(input_dir: Path, domain: str) -> Path:
    return input_dir / "seqprob_selection" / f"seqprob_selector_{domain}_k_curve.csv"


def load_panel(input_dir: Path, domain: str, lang: str, max_k: int, gamma: int):
    rec = asc.load_decisions(str(baseline_path(input_dir, domain, lang)))
    llmagg_df = asc.load_llmagg_summary(str(llmagg_summary_path(input_dir, domain, lang)), max_k)
    fact_majority_df = asc.load_fact_majority_summary(str(fact_majority_summary_path(input_dir, domain, lang)), max_k)
    expected = asc.infer_expected_topics(EXPECTED_TOPICS[(lang, domain)], llmagg_df, fact_majority_df)
    curves = asc.compute_baseline_curves(rec, max_k=max_k, gamma=gamma, expected_topics=expected)
    return curves, llmagg_df, fact_majority_df


def style_axes(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(axis="both", color="#d7d7d7", alpha=0.55, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9a9a9a")
    ax.spines["bottom"].set_color("#9a9a9a")
    ax.tick_params(colors="#2f2f2f", labelsize=10)


def add_panel_label(ax, label: str):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="semibold",
        color="#222222",
    )


def order_averaged_completion_majority(curves, n_perm: int = 64, seed: int = 13):
    mat = curves["mat"].fillna(0.0).values
    if mat.size == 0:
        return np.zeros(0, dtype=float)
    ks = np.arange(1, mat.shape[1] + 1)
    rng = np.random.default_rng(seed)
    acc = np.zeros(mat.shape[1], dtype=float)
    for _ in range(n_perm):
        perm = rng.permutation(mat.shape[1])
        success = (mat[:, perm] > 0.5).astype(int)
        acc += ((np.cumsum(success, axis=1) / ks) >= 0.5).mean(axis=0)
    return acc / float(n_perm)


def smooth_visual_curve(x: np.ndarray, y: np.ndarray, dense_step: float = 0.1, window: int = 7):
    if len(x) == 0:
        return x, y
    dense_x = np.arange(float(x.min()), float(x.max()) + dense_step, dense_step)
    dense_y = np.interp(dense_x, x, y)
    if window > 1 and dense_y.size >= window:
        kernel = np.ones(window, dtype=float) / float(window)
        pad = window // 2
        padded = np.pad(dense_y, (pad, pad), mode="edge")
        dense_y = np.convolve(padded, kernel, mode="valid")
    return dense_x, dense_y


def plot_panel(ax, curves, llmagg_df, fact_majority_df, domain: str, lang: str, zoom_k: int, panel_label: str):
    ks = curves["ks"]
    mask = ks <= zoom_k
    best_color = "#6f3ff5"
    majority_color = "#d44d3a"
    llmagg_color = "#1971c2"
    majority = order_averaged_completion_majority(curves, n_perm=64, seed=17)

    ax.plot(
        ks[mask],
        curves["prefix_best"][mask],
        label="Best single completion",
        linewidth=2.6,
        color=best_color,
        solid_capstyle="round",
    )
    ax.axhline(
        curves["prefix_mean"][0],
        label="Single prediction",
        linewidth=1.8,
        linestyle=(0, (6, 4)),
        color="#2b2b2b",
        alpha=0.9,
    )
    ax.plot(
        ks[mask],
        majority[mask],
        label="Completion majority (order-avg)",
        linewidth=1.2,
        color=majority_color,
        alpha=0.35,
        linestyle=(0, (4, 2)),
        solid_capstyle="round",
    )
    mx, my = smooth_visual_curve(ks[mask], majority[mask], dense_step=0.1, window=7)
    ax.plot(mx, my, linewidth=2.0, color=majority_color, alpha=0.9, solid_capstyle="round")
    llmagg_df = llmagg_df[llmagg_df["k"].isin(PLOT_KS)]
    ax.plot(
        llmagg_df["k"],
        llmagg_df["score"],
        label="LLM aggregation",
        linewidth=2.8,
        marker="o",
        markersize=4,
        color=llmagg_color,
        markerfacecolor="#ffffff",
        markeredgewidth=1.2,
    )
    style_axes(ax)
    ax.set_title(f"{domain.title()} / {lang.upper()}", fontsize=12, fontweight="semibold", color="#202020")
    ax.set_xlabel("Prefix k", fontsize=10, color="#202020")
    ax.set_ylabel("FactOwl score", fontsize=10, color="#202020")
    ax.set_xlim(0, zoom_k)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_xticklabels(["0", "4", "8", "12", "16", "20"])
    add_panel_label(ax, panel_label)


def plot_aux_panel(ax, curves, fact_majority_df, domain: str, lang: str, zoom_k: int, panel_label: str):
    ks = curves["ks"]
    mask = ks <= zoom_k
    majority_color = "#d44d3a"
    factmaj_color = "#178f57"
    majority = order_averaged_completion_majority(curves, n_perm=64, seed=17)

    ax.plot(
        ks[mask],
        majority[mask],
        label="Completion majority (order-avg)",
        linewidth=1.2,
        color=majority_color,
        alpha=0.35,
        linestyle=(0, (4, 2)),
        solid_capstyle="round",
    )
    mx, my = smooth_visual_curve(ks[mask], majority[mask], dense_step=0.1, window=7)
    ax.plot(mx, my, linewidth=2.0, color=majority_color, alpha=0.95, solid_capstyle="round")
    fact_majority_df = fact_majority_df[fact_majority_df["k"].isin(PLOT_KS)]
    ax.scatter(
        fact_majority_df["k"],
        fact_majority_df["score"],
        label="Fact majority",
        s=30,
        color=factmaj_color,
        edgecolors="#ffffff",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_facecolor("#ffffff")
    ax.grid(axis="both", color="#d7d7d7", alpha=0.55, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9a9a9a")
    ax.spines["bottom"].set_color("#9a9a9a")
    ax.tick_params(colors="#2f2f2f", labelsize=10)
    ax.set_title(f"{domain.title()} / {lang.upper()}", fontsize=12, fontweight="semibold", color="#202020")
    ax.set_xlabel("Prefix k", fontsize=10, color="#202020")
    ax.set_ylabel("FactOwl score", fontsize=10, color="#202020")
    ax.set_xlim(0, zoom_k)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_xticklabels(["0", "4", "8", "12", "16", "20"])
    add_panel_label(ax, panel_label)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.8, 8.6),
        sharex=True,
        sharey=False,
        gridspec_kw={"wspace": 0.18, "hspace": 0.22},
    )
    legend_handles = None
    legend_labels = None

    for row, lang in enumerate(LANGS):
        for col, domain in enumerate(DOMAINS):
            ax = axes[row, col]
            panel_label = PANEL_LABELS[row * len(DOMAINS) + col]
            curves, llmagg_df, fact_majority_df = load_panel(
                input_dir=input_dir,
                domain=domain,
                lang=lang,
                max_k=args.max_k,
                gamma=args.gamma,
            )
            plot_panel(ax, curves, llmagg_df, fact_majority_df, domain, lang, args.zoom_k, panel_label)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.supxlabel("Prefix k", fontsize=11, y=0.055)
    fig.supylabel("FactOwl score", fontsize=11, x=0.02)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.965, bottom=0.14, wspace=0.18, hspace=0.24)
    fig.savefig(out, dpi=220)
    print(f"saved={out}")

    aux_out = out.with_name(out.stem + "_aux" + out.suffix)
    fig2, axes2 = plt.subplots(
        2,
        3,
        figsize=(16.8, 8.6),
        sharex=True,
        sharey=False,
        gridspec_kw={"wspace": 0.18, "hspace": 0.22},
    )
    legend_handles = None
    legend_labels = None
    for row, lang in enumerate(LANGS):
        for col, domain in enumerate(DOMAINS):
            ax = axes2[row, col]
            panel_label = PANEL_LABELS[row * len(DOMAINS) + col]
            curves, llmagg_df, fact_majority_df = load_panel(
                input_dir=input_dir,
                domain=domain,
                lang=lang,
                max_k=args.max_k,
                gamma=args.gamma,
            )
            plot_aux_panel(ax, curves, fact_majority_df, domain, lang, args.zoom_k, panel_label)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig2.supxlabel("Prefix k", fontsize=11, y=0.055)
    fig2.supylabel("FactOwl score", fontsize=11, x=0.02)
    fig2.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig2.subplots_adjust(left=0.065, right=0.99, top=0.965, bottom=0.14, wspace=0.18, hspace=0.24)
    fig2.savefig(aux_out, dpi=220)
    print(f"saved={aux_out}")


if __name__ == "__main__":
    main()
