#!/usr/bin/env python3
"""Paper robustness analysis for RiDiC / LLMAgg.

This script builds bootstrap confidence intervals and gap tables from the
archived FactOwl outputs currently in the workspace.

Outputs are written to paper_prep/tables/.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


from _root import find_experiment_root

REPO_ROOT = find_experiment_root()
FACTOWL_DIR = Path(__file__).resolve().parents[1] / "factowl"
if str(FACTOWL_DIR) not in sys.path:
    sys.path.insert(0, str(FACTOWL_DIR))

import run_factowl_eval as rfe  # type: ignore
import train_llmagg_adaptivity as tla  # type: ignore
import train_llmagg_adaptivity_lang as tll  # type: ignore


DOMAINS = ["rivers", "cars", "disasters"]
LANGS = ["en", "zh"]
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 13
TARGET_K = 20
FULL_K = 99
ADAPT_RETENTIONS = [0.95, 0.99]


@dataclass(frozen=True)
class MetricCI:
    mean: float
    lower: float
    upper: float


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_resamples: int = BOOTSTRAP_RESAMPLES) -> MetricCI:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return MetricCI(np.nan, np.nan, np.nan)
    if values.size == 1:
        v = float(values[0])
        return MetricCI(v, v, v)
    means = np.empty(n_resamples, dtype=float)
    n = values.size
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values[idx].mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return MetricCI(float(values.mean()), float(lo), float(hi))


def load_topic_completion_matrix(path: Path, max_k: int = FULL_K) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=["score", "decisions"])
    rec = pd.DataFrame([ast.literal_eval(s) for s in df["decisions"]])
    if "topic" not in rec.columns or "completion_idx" not in rec.columns or "is_supported" not in rec.columns:
        raise ValueError(f"Missing required decision fields in {path}")
    comp = (
        rec.groupby(["topic", "completion_idx"])["is_supported"]
        .agg(["mean", "count"])
        .reset_index()
    )
    comp["penalty"] = comp["count"].apply(
        lambda n: 1.0 if n > 10 else np.exp(1 - 10 / n) if n > 0 else 0.0
    )
    comp["score"] = comp["mean"] * comp["penalty"]
    mat = comp.pivot(index="topic", columns="completion_idx", values="score")
    mat = mat.reindex(columns=range(max_k))
    return mat


def load_topic_score_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"topic", "k", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return df.drop_duplicates(subset=["topic", "k"], keep="last")


def load_llmagg_curve(path: Path, k_max: int = TARGET_K) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df[df["k"] <= k_max].copy()
    return df.drop_duplicates(subset=["topic", "k"], keep="last")


def topic_tier_map(domain: str, lang: str) -> dict[str, str]:
    domain_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "s-nlp___ri_di_c" / domain / "0.0.0"
    arrows = sorted(domain_dir.glob("*/ri_di_c-test.arrow"))
    if not arrows:
        domain_dir = REPO_ROOT / "cache" / "huggingface" / "datasets" / "s-nlp___ri_di_c" / domain / "0.0.0"
        arrows = sorted(domain_dir.glob("*/ri_di_c-test.arrow"))
    if not arrows:
        raise FileNotFoundError(f"Could not locate cached RiDiC arrow files for {domain}")
    ds = Dataset.from_file(str(arrows[-1]))
    title_col = "title_zh" if lang == "zh" else "title_en"
    if "popularity_part_sector" not in ds.column_names:
        raise ValueError("Dataset is missing popularity_part_sector")
    tier_label = {"0": "head", "1": "torso", "2": "tail", 0: "head", 1: "torso", 2: "tail"}
    return {
        title: tier_label.get(tier, tier)
        for title, tier in zip(ds[title_col], ds["popularity_part_sector"])
        if title and tier
    }


def weighted_ci(grouped: pd.Series, rng: np.random.Generator) -> MetricCI:
    return bootstrap_ci(grouped.values.astype(float), rng)


def load_question_source(lang: str) -> pd.DataFrame:
    source = REPO_ROOT / "experiments" / "factowl_eval" / "adaptivity_lang" / f"{lang}_llmagg_adaptivity_lang_99_predictions.tsv"
    if not source.exists():
        raise FileNotFoundError(f"Missing question source TSV: {source}")
    return pd.read_csv(source, sep="\t")[["topic", "question", "domain"]].drop_duplicates()


def build_adaptivity_predictions(lang: str, retention: float, detail_dir: Path) -> pd.DataFrame:
    """Rebuild the question-only adaptive controller locally.

    We reuse the archived 99-retention source TSV for the question text / domain
    fields and the per-retention llmagg detail TSVs for labels.
    """
    question_df = load_question_source(lang)
    detail_frames = []
    for domain in DOMAINS:
        path = detail_dir / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_details.tsv"
        detail_frames.append(pd.read_csv(path, sep="\t").assign(domain=domain))
    detail_df = pd.concat(detail_frames, ignore_index=True)
    ex_df = tll.derive_examples(detail_df, retention)
    train_df = question_df.merge(ex_df, on=["domain", "topic"], how="inner")
    feature_rows = [tla.featurize(q, lang) for q in train_df["question"].fillna("")]
    feature_df = pd.DataFrame(feature_rows)
    strat_labels = tll.make_strat_labels(train_df)
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=13)
    all_preds = []
    allowed_ks = sorted(detail_df["k"].astype(int).unique().tolist())
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(feature_df.values, strat_labels), start=1):
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=13 + fold_id,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
        )
        X = feature_df.values
        y = train_df["optimized_k"].astype(int).values
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        test_df = train_df.iloc[test_idx][["topic", "question", "domain", "baseline_k", "baseline_score", "optimized_k"]].copy()
        test_df["pred_k"] = [tla.nearest_allowed(int(v), allowed_ks) for v in pred]
        test_df["pred_k"] = np.minimum(test_df["pred_k"], test_df["baseline_k"])
        score_lookup = detail_df[["domain", "topic", "k", "score"]].copy()
        test_df = test_df.merge(
            score_lookup.rename(columns={"k": "pred_k", "score": "pred_score"}),
            on=["domain", "topic", "pred_k"],
            how="left",
        )
        test_df["score_retention"] = test_df["pred_score"] / test_df["baseline_score"].replace(0, np.nan)
        test_df["score_retention"] = test_df["score_retention"].fillna(1.0)
        test_df["compute_savings"] = 1.0 - (test_df["pred_k"] / test_df["baseline_k"])
        test_df["fold"] = fold_id
        all_preds.append(test_df)
    return pd.concat(all_preds, ignore_index=True)


def main():
    out_dir = REPO_ROOT / "paper_prep" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    main_rows = []
    prefix_rows = []
    tier_rows = []
    gap_rows = []
    adaptive_rows = []

    for lang in LANGS:
        for domain in DOMAINS:
            if lang == "en":
                allcomp_path = REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_k99_allcompletions_finalonly.tsv"
                llmagg_sparse_detail_path = REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_details.tsv"
                llmagg_k20_detail_path = REPO_ROOT / "paper_prep" / "experiments" / "llmagg_k20" / f"en_{domain}" / f"pred_{domain}_en_k99_llmaggcurve_finalonly_details.tsv"
                factmaj_detail_path = REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_k99_factmajoritycurve_finalonly_details.tsv"
                llmagg_summary_path = REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_summary.tsv"
                factmaj_summary_path = REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_k99_factmajoritycurve_finalonly_summary.tsv"
            else:
                allcomp_path = REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_zh_k99_allcompletions_finalonly.tsv"
                llmagg_sparse_detail_path = REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_details.tsv"
                llmagg_k20_detail_path = REPO_ROOT / "paper_prep" / "experiments" / "llmagg_k20" / f"zh_{domain}" / f"pred_{domain}_zh_k99_llmaggcurve_finalonly_details.tsv"
                factmaj_detail_path = REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_zh_k99_factmajoritycurve_finalonly_details.tsv"
                llmagg_summary_path = REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_summary.tsv"
                factmaj_summary_path = REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_zh_k99_factmajoritycurve_finalonly_summary.tsv"

            # Main comparison.
            mat = load_topic_completion_matrix(allcomp_path, FULL_K).fillna(0.0)
            comp_topics = mat.index.to_numpy()
            comp_scores = mat.to_numpy(dtype=float)
            # Topic-level scores.
            single_vals = comp_scores[:, 0]
            best20_vals = np.max(comp_scores[:, :TARGET_K], axis=1)
            maj20_vals = (((np.cumsum((comp_scores[:, :TARGET_K] > 0.5).astype(int), axis=1) / np.arange(1, TARGET_K + 1)) >= 0.5).astype(float))[:, -1]

            llmagg_sparse_detail = load_llmagg_curve(llmagg_sparse_detail_path, TARGET_K)
            llmagg20_detail = pd.read_csv(llmagg_k20_detail_path, sep="\t").drop_duplicates(subset=["topic", "k"], keep="last")
            llmagg20 = llmagg20_detail[llmagg20_detail["k"] == TARGET_K].set_index("topic")["score"]
            factmaj_detail = load_topic_score_curve(factmaj_detail_path)
            factmaj20 = factmaj_detail[factmaj_detail["k"] == TARGET_K].set_index("topic")["score"]
            llmagg_summary = pd.read_csv(llmagg_summary_path, sep="\t")
            factmaj_summary = pd.read_csv(factmaj_summary_path, sep="\t")
            seqprob_path = REPO_ROOT / "experiments" / "factowl_eval" / "seqprob_selection" / f"seqprob_selector_{domain}_k_curve.csv"
            seqprob_summary = pd.read_csv(seqprob_path)
            seqprob20 = float(seqprob_summary.loc[seqprob_summary["k"] == TARGET_K, "logprob_selector"].iloc[0])

            # Bootstrap topic-resampled CIs.
            ci_single = bootstrap_ci(single_vals, rng)
            ci_best20 = bootstrap_ci(best20_vals, rng)
            ci_maj20 = bootstrap_ci(maj20_vals, rng)
            ci_fact20 = bootstrap_ci(factmaj20.reindex(comp_topics).to_numpy(dtype=float), rng)
            ci_llm20 = bootstrap_ci(llmagg20.reindex(comp_topics).to_numpy(dtype=float), rng)

            main_rows.append(
                {
                    "lang": lang,
                    "domain": domain,
                    "method": "Single",
                    "mean": ci_single.mean,
                    "lower": ci_single.lower,
                    "upper": ci_single.upper,
                }
            )
            main_rows.append(
                {
                    "lang": lang,
                    "domain": domain,
                    "method": "Best_single",
                    "mean": ci_best20.mean,
                    "lower": ci_best20.lower,
                    "upper": ci_best20.upper,
                }
            )
            main_rows.append(
                {
                    "lang": lang,
                    "domain": domain,
                    "method": "Majority_completion",
                    "mean": ci_maj20.mean,
                    "lower": ci_maj20.lower,
                    "upper": ci_maj20.upper,
                }
            )
            main_rows.append(
                {
                    "lang": lang,
                    "domain": domain,
                    "method": "Fact_majority",
                    "mean": ci_fact20.mean,
                    "lower": ci_fact20.lower,
                    "upper": ci_fact20.upper,
                }
            )
            main_rows.append(
                {
                    "lang": lang,
                    "domain": domain,
                    "method": "Seqprob",
                    "mean": seqprob20,
                    "lower": np.nan,
                    "upper": np.nan,
                    "note": "summary-only; no topic-level trace available",
                }
            )
            main_rows.append(
                {
                    "lang": lang,
                    "domain": domain,
                    "method": "LLMAgg",
                    "mean": ci_llm20.mean,
                    "lower": ci_llm20.lower,
                    "upper": ci_llm20.upper,
                }
            )

            # Prefix curve CIs for LLMAgg.
            llm_topic = llmagg_sparse_detail.pivot(index="topic", columns="k", values="score").reindex(index=comp_topics)
            llm_topic.columns = llm_topic.columns.astype(int)
            for k in sorted([int(k) for k in llm_topic.columns if int(k) <= 16]):
                vals = llm_topic[k].to_numpy(dtype=float)
                ci = bootstrap_ci(vals, rng)
                prefix_rows.append(
                    {
                        "lang": lang,
                        "domain": domain,
                        "k": k,
                        "mean": ci.mean,
                        "lower": ci.lower,
                        "upper": ci.upper,
                    }
                )
            llm20_vals = llmagg20_detail.set_index("topic")["score"].reindex(comp_topics).to_numpy(dtype=float)
            ci20 = bootstrap_ci(llm20_vals, rng)
            prefix_rows.append(
                {
                    "lang": lang,
                    "domain": domain,
                    "k": 20,
                    "mean": ci20.mean,
                    "lower": ci20.lower,
                    "upper": ci20.upper,
                }
            )

            # Tier comparison CIs using dataset-derived popularity tiers.
            tiers = topic_tier_map(domain, lang)
            tier_topic_scores = pd.DataFrame(
                {
                    "topic": comp_topics,
                    "tier": [tiers.get(t) for t in comp_topics],
                    "single": single_vals,
                    "llmagg_k20": llmagg20.reindex(comp_topics).to_numpy(dtype=float),
                }
            ).dropna(subset=["tier"])
            for tier, tier_df in tier_topic_scores.groupby("tier", sort=False):
                s_ci = bootstrap_ci(tier_df["single"].to_numpy(dtype=float), rng)
                l_ci = bootstrap_ci(tier_df["llmagg_k20"].to_numpy(dtype=float), rng)
                tier_rows.append(
                    {
                        "lang": lang,
                        "domain": domain,
                        "tier": tier,
                        "n_topics": int(len(tier_df)),
                        "single_mean": s_ci.mean,
                        "single_lower": s_ci.lower,
                        "single_upper": s_ci.upper,
                        "llmagg_mean": l_ci.mean,
                        "llmagg_lower": l_ci.lower,
                        "llmagg_upper": l_ci.upper,
                        "delta_mean": l_ci.mean - s_ci.mean,
                        "delta_lower": l_ci.lower - s_ci.upper,
                        "delta_upper": l_ci.upper - s_ci.lower,
                    }
                )

            # K20 vs K99 gap.
            if lang == "en":
                k20_detail_path = REPO_ROOT / "paper_prep" / "experiments" / "llmagg_k20" / f"en_{domain}" / f"pred_{domain}_en_k99_llmaggcurve_finalonly_details.tsv"
            else:
                k20_detail_path = REPO_ROOT / "paper_prep" / "experiments" / "llmagg_k20" / f"zh_{domain}" / f"pred_{domain}_zh_k99_llmaggcurve_finalonly_details.tsv"
            k20_detail = pd.read_csv(k20_detail_path, sep="\t").drop_duplicates(subset=["topic", "k"], keep="last")
            k20_vals = k20_detail.set_index("topic")["score"].reindex(comp_topics).to_numpy(dtype=float)
            k20 = bootstrap_ci(k20_vals, rng)
            k32 = float(llmagg_summary[llmagg_summary["k"] == 32]["score"].iloc[0]) if "score" in llmagg_summary.columns else np.nan
            k64 = float(llmagg_summary[llmagg_summary["k"] == 64]["score"].iloc[0]) if "score" in llmagg_summary.columns else np.nan
            k99 = float(llmagg_summary[llmagg_summary["k"] == 99]["score"].iloc[0]) if "score" in llmagg_summary.columns else np.nan
            gap_rows.append(
                {
                    "lang": lang,
                    "domain": domain,
                    "K20": k20.mean,
                    "K20_lower": k20.lower,
                    "K20_upper": k20.upper,
                    "K32": k32,
                    "K64": k64,
                    "K99": k99,
                    "Delta_99_minus_20": k99 - k20.mean,
                    "Rel_compute_32x_vs_20x": 32 / 20,
                    "Rel_compute_64x_vs_20x": 64 / 20,
                    "Rel_compute_99x_vs_20x": 99 / 20,
                    "Rel_compute_plus_agg_32x_vs_20x": (32 + 1) / (20 + 1),
                    "Rel_compute_plus_agg_64x_vs_20x": (64 + 1) / (20 + 1),
                    "Rel_compute_plus_agg_99x_vs_20x": (99 + 1) / (20 + 1),
                }
            )

    # Adaptive tradeoff. We reuse the existing 0.99 predictions and can regenerate 0.95 if present.
    adapt_detail_dir = REPO_ROOT / "experiments" / "factowl_eval"
    for lang in LANGS:
        for retention in ADAPT_RETENTIONS:
            df = build_adaptivity_predictions(lang, retention, adapt_detail_dir).copy()
            # Each row is one evaluation instance.
            for domain, dom_df in df.groupby("domain", sort=False):
                gen_only = dom_df["compute_savings"].to_numpy(dtype=float)
                gen_plus_agg = 1.0 - ((dom_df["pred_k"].to_numpy(dtype=float) + 1.0) / (FULL_K + 1.0))
                score_ret = dom_df["score_retention"].to_numpy(dtype=float)
                coverage = (score_ret >= retention).astype(float)
                ci_gen = bootstrap_ci(gen_only, rng)
                ci_genagg = bootstrap_ci(gen_plus_agg, rng)
                ci_score = bootstrap_ci(score_ret, rng)
                ci_cov = bootstrap_ci(coverage, rng)
                adaptive_rows.append(
                    {
                        "lang": lang,
                        "domain": domain,
                        "retention_target": retention,
                        "metric": "gen_only",
                        "mean": ci_gen.mean,
                        "lower": ci_gen.lower,
                        "upper": ci_gen.upper,
                    }
                )
                adaptive_rows.append(
                    {
                        "lang": lang,
                        "domain": domain,
                        "retention_target": retention,
                        "metric": "gen_plus_agg",
                        "mean": ci_genagg.mean,
                        "lower": ci_genagg.lower,
                        "upper": ci_genagg.upper,
                    }
                )
                adaptive_rows.append(
                    {
                        "lang": lang,
                        "domain": domain,
                        "retention_target": retention,
                        "metric": "score_retention",
                        "mean": ci_score.mean,
                        "lower": ci_score.lower,
                        "upper": ci_score.upper,
                    }
                )
                adaptive_rows.append(
                    {
                        "lang": lang,
                        "domain": domain,
                        "retention_target": retention,
                        "metric": "coverage",
                        "mean": ci_cov.mean,
                        "lower": ci_cov.lower,
                        "upper": ci_cov.upper,
                    }
                )

    pd.DataFrame(main_rows).to_csv(out_dir / "bootstrap_main_comparison.tsv", sep="\t", index=False)
    pd.DataFrame(prefix_rows).to_csv(out_dir / "bootstrap_llmagg_prefix_curve.tsv", sep="\t", index=False)
    pd.DataFrame(tier_rows).to_csv(out_dir / "bootstrap_tier_comparison.tsv", sep="\t", index=False)
    pd.DataFrame(gap_rows).to_csv(out_dir / "llmagg_k20_k99_gap.tsv", sep="\t", index=False)
    pd.DataFrame(adaptive_rows).to_csv(out_dir / "bootstrap_adaptive_tradeoff.tsv", sep="\t", index=False)

    print("wrote:")
    for name in [
        "bootstrap_main_comparison.tsv",
        "bootstrap_llmagg_prefix_curve.tsv",
        "bootstrap_tier_comparison.tsv",
        "llmagg_k20_k99_gap.tsv",
        "bootstrap_adaptive_tradeoff.tsv",
    ]:
        print(out_dir / name)


if __name__ == "__main__":
    main()
