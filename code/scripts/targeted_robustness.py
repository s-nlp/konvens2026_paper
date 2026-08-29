#!/usr/bin/env python3
"""Targeted robustness analyses for the RiDiC / LLMAgg paper.

This script focuses on the remaining reviewer-facing gaps:
- adaptive feature importance and ablations
- adaptive failure / low-coverage profiling
- domain difficulty profiling for rivers vs cars vs disasters

It only uses local artifacts already present in the workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit


from _root import find_experiment_root

REPO_ROOT = find_experiment_root()
FACTOWL_DIR = Path(__file__).resolve().parents[1] / "factowl"
if str(FACTOWL_DIR) not in sys.path:
    sys.path.insert(0, str(FACTOWL_DIR))

import train_llmagg_adaptivity as tla  # type: ignore
import train_llmagg_adaptivity_lang as tll  # type: ignore


DOMAINS = ["rivers", "cars", "disasters"]
LANGS = ["en", "zh"]
RETENTIONS = [0.95, 0.99]
N_SPLITS = 5
TEST_SIZE = 0.3
SEED = 13
BOOTSTRAP_RESAMPLES = 1000


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    compute_savings: float
    score_retention: float
    coverage: float
    n_eval: int


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        v = float(values[0])
        return v, v, v
    means = np.empty(n_resamples, dtype=float)
    n = values.size
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values[idx].mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def local_question_len(question: str) -> dict[str, float]:
    q = question or ""
    lowered = q.lower()
    return {
        "q_len_chars": float(len(q)),
        "q_len_words": float(len(q.split())),
        "q_avg_word_length": float(len(q) / max(len(q.split()), 1)),
        "q_num_digits": float(sum(ch.isdigit() for ch in q)),
        "q_num_punct": float(sum(ch in ".,!?;:()[]{}-_/\\'\"`、，。！？；：" for ch in q)),
        "q_has_year": float(int(any(tok.isdigit() and len(tok) == 4 for tok in q.split()))),
        "q_is_chinese": float(int("请用一个段落告诉我你对" in q)),
        "starts_with_what": float(int(lowered.startswith("what"))),
        "starts_with_where": float(int(lowered.startswith("where"))),
        "starts_with_when": float(int(lowered.startswith("when"))),
        "starts_with_who": float(int(lowered.startswith("who"))),
        "starts_with_which": float(int(lowered.startswith("which"))),
        "contains_parentheses": float(int("(" in q or ")" in q)),
        "contains_comma": float(int("," in q)),
        "contains_hyphen": float(int("-" in q)),
    }


def load_detail_df(lang: str, domain: str) -> pd.DataFrame:
    path = REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_details.tsv"
    df = pd.read_csv(path, sep="\t")
    df["lang"] = lang
    df["domain"] = domain
    return df


def build_pooled_dataset(retention: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    detail_frames = []
    feature_frames = []
    for lang in LANGS:
        for domain in DOMAINS:
            q_df = tll.load_question_df(domain, lang).copy()
            q_df["lang"] = lang
            detail_df = load_detail_df(lang, domain)
            ex_df = tll.derive_examples(detail_df, retention)
            merged = q_df.merge(ex_df, on=["domain", "topic"], how="inner")
            merged["lang"] = lang
            feature_rows = [
                {
                    **tll.featurize_row(q, lang, d, int(k)),
                    "lang_is_zh": float(int(lang == "zh")),
                    "lang_is_en": float(int(lang == "en")),
                }
                for q, d, k in zip(merged["question"].fillna(""), merged["domain"], merged["baseline_k"])
            ]
            feat_df = pd.DataFrame(feature_rows)
            frames.append(merged)
            detail_frames.append(detail_df)
            feature_frames.append(feat_df)
    df = pd.concat(frames, ignore_index=True)
    detail_df = pd.concat(detail_frames, ignore_index=True)
    feature_df = pd.concat(feature_frames, ignore_index=True)
    return df, detail_df, feature_df


def train_eval(df: pd.DataFrame, detail_df: pd.DataFrame, feature_df: pd.DataFrame, feature_cols: list[str], retention: float) -> tuple[EvalResult, list[np.ndarray], float]:
    strat_labels = (
        df["lang"].astype(str)
        + "|"
        + df["domain"].astype(str)
        + "|"
        + df["baseline_k"].astype(str)
        + "|"
        + df["optimized_k"].astype(str)
    )
    counts = strat_labels.value_counts()
    rare = counts[counts < 2].index
    strat_labels = strat_labels.where(~strat_labels.isin(rare), other="other")

    splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)
    allowed_ks = sorted(detail_df["k"].astype(int).unique().tolist())
    X = feature_df[feature_cols].values
    y = df["optimized_k"].astype(int).values

    all_metrics = []
    importances = []
    fold_accs = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, strat_labels), start=1):
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=SEED + fold_id,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        fold_accs.append(float((pred == y[test_idx]).mean()))
        test_df = df.iloc[test_idx][["lang", "domain", "topic", "question", "baseline_k", "baseline_score"]].copy()
        test_df["pred_k"] = [tla.nearest_allowed(int(v), allowed_ks) for v in pred]
        test_df["pred_k"] = np.minimum(test_df["pred_k"], test_df["baseline_k"])
        score_lookup = detail_df[["lang", "domain", "topic", "k", "score"]].copy()
        test_df = test_df.merge(
            score_lookup.rename(columns={"k": "pred_k", "score": "pred_score"}),
            on=["lang", "domain", "topic", "pred_k"],
            how="left",
        )
        test_df["score_retention"] = test_df["pred_score"] / test_df["baseline_score"].replace(0, np.nan)
        test_df["score_retention"] = test_df["score_retention"].fillna(1.0)
        test_df["compute_savings"] = 1.0 - (test_df["pred_k"] / test_df["baseline_k"])
        test_df["coverage"] = test_df["score_retention"]
        all_metrics.append(test_df)
        importances.append(clf.feature_importances_)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    result = EvalResult(
        accuracy=float(np.mean(fold_accs)),
        compute_savings=float(metrics_df["compute_savings"].mean()),
        score_retention=float(metrics_df["score_retention"].mean()),
        coverage=float((metrics_df["score_retention"] >= retention).mean()),
        n_eval=int(len(metrics_df)),
    )
    return result, importances, float(np.mean(fold_accs))


def train_ablation_suite(retention: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, detail_df, feature_df = build_pooled_dataset(retention)
    mask = df["baseline_k"] == 99
    df = df.loc[mask].reset_index(drop=True)
    feature_df = feature_df.loc[mask].reset_index(drop=True)

    # Group definitions over the pooled feature space.
    full_cols = list(feature_df.columns)
    domain_cols = [c for c in full_cols if c.startswith("domain_is_")]
    length_cols = ["q_len_chars", "q_len_words", "q_avg_word_length", "q_num_digits", "q_has_year"]
    lang_cols = ["q_is_chinese", "lang_is_zh", "lang_is_en"]
    punct_cols = ["q_num_punct", "starts_with_what", "starts_with_where", "starts_with_when", "starts_with_who", "starts_with_which", "contains_parentheses", "contains_comma", "contains_hyphen"]
    budget_cols = ["baseline_k", "baseline_k_log2"]

    feature_sets = {
        "full": full_cols,
        "length_only": sorted(set(length_cols + budget_cols)),
        "domain_only": sorted(set(domain_cols + budget_cols)),
        "language_only": sorted(set(lang_cols + budget_cols)),
        "punct_wh_only": sorted(set(punct_cols + budget_cols)),
        "baseline_only": budget_cols,
        "no_domain": [c for c in full_cols if c not in domain_cols],
        "no_length": [c for c in full_cols if c not in length_cols],
        "no_language": [c for c in full_cols if c not in lang_cols],
        "no_punct_wh": [c for c in full_cols if c not in punct_cols],
    }

    ablation_rows = []
    importance_rows = []
    for name, cols in feature_sets.items():
        result, importances, acc = train_eval(df, detail_df, feature_df, cols, retention)
        ablation_rows.append(
            {
                "retention_target": retention,
                "feature_set": name,
                "n_features": len(cols),
                "accuracy": result.accuracy,
                "compute_savings": result.compute_savings,
                "score_retention": result.score_retention,
                "coverage": result.coverage,
                "n_eval": result.n_eval,
            }
        )
        if name == "full":
            imp_arr = np.vstack(importances)
            for idx, feat in enumerate(cols):
                importance_rows.append(
                    {
                        "retention_target": retention,
                        "feature": feat,
                        "mean_importance": float(imp_arr[:, idx].mean()),
                        "std_importance": float(imp_arr[:, idx].std(ddof=1) if imp_arr.shape[0] > 1 else 0.0),
                    }
                )

    return pd.DataFrame(ablation_rows), pd.DataFrame(importance_rows)


def make_failure_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(REPO_ROOT / "paper_prep" / "tables" / "main_comparison_k20.tsv", sep="\t")
    main = main.set_index(["lang", "domain"])
    tier_map = {}
    for lang in LANGS:
        for domain in DOMAINS:
            tier_map.update(_tier_map(domain, lang))

    profile_rows = []
    tier_rows = []
    for lang in LANGS:
        for retention in RETENTIONS:
            pred_path = REPO_ROOT / "paper_prep" / "tmp_adapt_ci" / f"{lang}_{int(retention*100):03d}" / f"{lang}_llmagg_adaptivity_lang_{int(retention*100)}_predictions.tsv"
            pred = pd.read_csv(pred_path, sep="\t")
            pred = pred[pred["baseline_k"] == 99].copy()
            pred["tier"] = pred["topic"].map(tier_map)
            pred["q_len_words"] = pred["question"].fillna("").str.split().str.len()
            pred["q_len_chars"] = pred["question"].fillna("").str.len()
            pred["q_num_punct"] = pred["question"].fillna("").str.count(r"[\\.,!?;:()\\[\\]{}\\-/\\\\'\"`、，。！？；：]")
            pred["q_has_year"] = pred["question"].fillna("").str.contains(r"\\b\\d{4}\\b", regex=True)
            pred["status"] = np.where(pred["score_retention"] >= retention, "success", "failure")

            topic_level = (
                pred.groupby(["domain", "topic"], as_index=False)
                .agg(
                    question=("question", "first"),
                    tier=("tier", "first"),
                    baseline_score=("baseline_score", "mean"),
                    pred_score=("pred_score", "mean"),
                    score_retention=("score_retention", "mean"),
                    compute_savings=("compute_savings", "mean"),
                    pred_k=("pred_k", "mean"),
                    q_len_words=("q_len_words", "mean"),
                    q_len_chars=("q_len_chars", "mean"),
                    q_num_punct=("q_num_punct", "mean"),
                    q_has_year=("q_has_year", "mean"),
                )
            )
            topic_level["status"] = np.where(topic_level["score_retention"] >= retention, "success", "failure")
            topic_level["lang"] = lang
            topic_level["retention_target"] = retention
            topic_level["single_domain_score"] = topic_level["domain"].map(lambda d: float(main.loc[(lang, d), "Single"]))
            topic_level["llmagg_domain_score"] = topic_level["domain"].map(lambda d: float(main.loc[(lang, d), "LLMAgg"]))

            for status, sub in topic_level.groupby("status"):
                profile_rows.append(
                    {
                        "lang": lang,
                        "retention_target": retention,
                        "status": status,
                        "n_topics": int(len(sub)),
                        "compute_savings": float(sub["compute_savings"].mean()),
                        "score_retention": float(sub["score_retention"].mean()),
                        "baseline_score": float(sub["baseline_score"].mean()),
                        "single_domain_score": float(sub["single_domain_score"].mean()),
                        "llmagg_domain_score": float(sub["llmagg_domain_score"].mean()),
                        "q_len_words": float(sub["q_len_words"].mean()),
                        "q_len_chars": float(sub["q_len_chars"].mean()),
                        "q_num_punct": float(sub["q_num_punct"].mean()),
                        "q_has_year": float(sub["q_has_year"].mean()),
                    }
                )

            for (domain, tier), sub in topic_level.groupby(["domain", "tier"]):
                tier_rows.append(
                    {
                        "lang": lang,
                        "retention_target": retention,
                        "domain": domain,
                        "tier": tier,
                        "n_topics": int(len(sub)),
                        "n_success": int((sub["status"] == "success").sum()),
                        "n_failure": int((sub["status"] == "failure").sum()),
                        "success_rate": float((sub["status"] == "success").mean()),
                        "compute_savings": float(sub["compute_savings"].mean()),
                        "score_retention": float(sub["score_retention"].mean()),
                        "baseline_score": float(sub["baseline_score"].mean()),
                    }
                )

    return pd.DataFrame(profile_rows), pd.DataFrame(tier_rows)


def _tier_map(domain: str, lang: str) -> dict[str, str]:
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "s-nlp___ri_di_c" / domain / "0.0.0"
    arrows = sorted(cache_root.glob("*/ri_di_c-test.arrow"))
    if not arrows:
        return {}
    from datasets import Dataset

    ds = Dataset.from_file(str(arrows[-1]))
    title_col = "title_zh" if lang == "zh" else "title_en"
    tier_label = {"0": "head", "1": "torso", "2": "tail", 0: "head", 1: "torso", 2: "tail"}
    return {
        title: tier_label.get(tier, tier)
        for title, tier in zip(ds[title_col], ds["popularity_part_sector"])
        if title and tier is not None
    }


def domain_difficulty_profile() -> pd.DataFrame:
    rows = []
    main = pd.read_csv(REPO_ROOT / "paper_prep" / "tables" / "main_comparison_k20.tsv", sep="\t")
    prefix = pd.read_csv(REPO_ROOT / "paper_prep" / "tables" / "llmagg_prefix_curve_k1_20.tsv", sep="\t")
    for lang in LANGS:
        for domain in DOMAINS:
            detail_k20 = pd.read_csv(
                REPO_ROOT / "paper_prep" / "experiments" / "llmagg_k20" / f"{lang}_{domain}" / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_details.tsv",
                sep="\t",
            )
            curve = pd.read_csv(
                REPO_ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_details.tsv",
                sep="\t",
            )
            curve = curve[curve["k"] <= 20].copy()
            psub = prefix[(prefix["lang"] == lang) & (prefix["domain"] == domain)].sort_values("k")
            per_topic = curve.pivot_table(index="topic", columns="k", values="score")
            best_k = per_topic.idxmax(axis=1)
            rows.append(
                {
                    "lang": lang,
                    "domain": domain,
                    "single_score": float(main[(main["lang"] == lang) & (main["domain"] == domain)]["Single"].iloc[0]),
                    "llmagg_k20_score": float(main[(main["lang"] == lang) & (main["domain"] == domain)]["LLMAgg"].iloc[0]),
                    "gain_k20_minus_k1": float(psub[psub["k"] == 20]["score"].iloc[0] - psub[psub["k"] == 1]["score"].iloc[0]),
                    "mean_num_facts_k20": float(detail_k20["num_facts_per_response"].mean()),
                    "mean_llmagg_std_k1_20": float(per_topic.std(axis=1).mean()),
                    "mean_best_k": float(best_k.mean()),
                    "p_best_k_ge16": float((best_k >= 16).mean()),
                }
            )
    return pd.DataFrame(rows)


def main():
    out_dir = REPO_ROOT / "paper_prep" / "tables"

    # 1) Feature importance + ablation
    ablation_frames = []
    importance_frames = []
    for retention in RETENTIONS:
        ablation_df, importance_df = train_ablation_suite(retention)
        ablation_frames.append(ablation_df)
        importance_frames.append(importance_df)
    pd.concat(ablation_frames, ignore_index=True).to_csv(out_dir / "adaptive_feature_ablation.tsv", sep="\t", index=False)
    pd.concat(importance_frames, ignore_index=True).to_csv(out_dir / "adaptive_feature_importance.tsv", sep="\t", index=False)

    # 2) Failure / low-coverage profiling
    failure_profile_df, failure_tier_df = make_failure_profiles()
    failure_profile_df.to_csv(out_dir / "adaptive_failure_profile.tsv", sep="\t", index=False)
    failure_tier_df.to_csv(out_dir / "adaptive_failure_by_tier.tsv", sep="\t", index=False)

    # 3) Domain difficulty
    domain_df = domain_difficulty_profile()
    domain_df.to_csv(out_dir / "domain_difficulty_profile.tsv", sep="\t", index=False)

    print("wrote:")
    for name in [
        "adaptive_feature_importance.tsv",
        "adaptive_feature_ablation.tsv",
        "adaptive_failure_profile.tsv",
        "adaptive_failure_by_tier.tsv",
        "domain_difficulty_profile.tsv",
    ]:
        print(out_dir / name)


if __name__ == "__main__":
    main()
