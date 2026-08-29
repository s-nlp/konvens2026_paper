#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


from _root import find_experiment_root

ROOT = find_experiment_root()
TABLES = ROOT / "paper_prep" / "tables"
BOOTSTRAP_RESAMPLES = 1000
SEED = 13
K20 = 20.0
K99 = 99.0
C_AGG = 1.0


SOURCES = {
    ("en", 0.95): ROOT / "paper_prep" / "tmp_adapt_ci" / "en_095" / "en_llmagg_adaptivity_lang_95_predictions.tsv",
    ("zh", 0.95): ROOT / "paper_prep" / "tmp_adapt_ci" / "zh_095" / "zh_llmagg_adaptivity_lang_95_predictions.tsv",
    ("en", 0.99): ROOT / "experiments" / "factowl_eval" / "adaptivity_lang" / "en_llmagg_adaptivity_lang_99_predictions.tsv",
    ("zh", 0.99): ROOT / "experiments" / "factowl_eval" / "adaptivity_lang" / "zh_llmagg_adaptivity_lang_99_predictions.tsv",
}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        v = float(values[0])
        return v, v, v
    n = values.size
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values[idx].mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def summarize_group(df: pd.DataFrame, lang: str, retention: float, domain: str, rng: np.random.Generator) -> dict:
    sub = df[(df["baseline_k"] == 99) & (df["domain"] == domain)].copy()
    if sub.empty:
        raise ValueError(f"Missing baseline_k=99 rows for {lang} {retention} {domain}")
    pred_k = sub["pred_k"].to_numpy(dtype=float)
    score_ret = sub["score_retention"].to_numpy(dtype=float)
    coverage = (score_ret >= retention).astype(float)

    savings_99_gen = 1.0 - (pred_k / K99)
    savings_99_plusagg = 1.0 - ((pred_k + C_AGG) / (K99 + C_AGG))
    savings_20_gen = 1.0 - (pred_k / K20)
    savings_20_plusagg = 1.0 - ((pred_k + C_AGG) / (K20 + C_AGG))
    ratio_20_gen = pred_k / K20
    ratio_20_plusagg = (pred_k + C_AGG) / (K20 + C_AGG)

    out = {
        "lang": lang,
        "domain": domain,
        "retention_target": retention,
        "n_eval": int(len(sub)),
        "avg_pred_k": float(pred_k.mean()),
        "score_retention": float(score_ret.mean()),
        "coverage": float(coverage.mean()),
    }

    metrics = {
        "savings_vs_k99_gen": savings_99_gen,
        "savings_vs_k99_plusagg": savings_99_plusagg,
        "savings_vs_k20_gen": savings_20_gen,
        "savings_vs_k20_plusagg": savings_20_plusagg,
        "compute_ratio_vs_k20_gen": ratio_20_gen,
        "compute_ratio_vs_k20_plusagg": ratio_20_plusagg,
        "score_retention": score_ret,
        "coverage": coverage,
    }
    for name, vals in metrics.items():
        mean, lo, hi = bootstrap_ci(vals, rng)
        out[name] = mean
        out[f"{name}_lower"] = lo
        out[f"{name}_upper"] = hi
    return out


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for (lang, retention), path in SOURCES.items():
        df = pd.read_csv(path, sep="\t")
        for domain in ["cars", "disasters", "rivers"]:
            rows.append(summarize_group(df, lang, retention, domain, rng))
    out_df = pd.DataFrame(rows).sort_values(["lang", "domain", "retention_target"]).reset_index(drop=True)
    out_df.to_csv(TABLES / "adaptive_cost_accounting.tsv", sep="\t", index=False)

    compact_cols = [
        "lang",
        "domain",
        "retention_target",
        "avg_pred_k",
        "score_retention",
        "coverage",
        "savings_vs_k99_gen",
        "savings_vs_k99_plusagg",
        "savings_vs_k20_gen",
        "savings_vs_k20_plusagg",
        "compute_ratio_vs_k20_gen",
        "compute_ratio_vs_k20_plusagg",
    ]
    out_df[compact_cols].to_csv(TABLES / "adaptive_cost_accounting_compact.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
