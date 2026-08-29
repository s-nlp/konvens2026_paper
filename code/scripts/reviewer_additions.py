#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.ensemble import RandomForestClassifier


from _root import find_experiment_root

ROOT = find_experiment_root()

from evaluate_ridic_reasc import load_factowl_completion_scores  # type: ignore


DOMAINS = ["rivers", "cars", "disasters"]
LANGS = ["en", "zh"]
RETENTIONS = [0.95, 0.99]

TABLES = ROOT / "paper_prep" / "tables"
APPENDIX = ROOT / "paper_prep" / "appendix"

REASC_SCORED = ROOT / "outputs" / "reasc_posthoc"
LLMAGG_K20 = ROOT / "paper_prep" / "experiments" / "llmagg_k20"
ADAPT_095 = ROOT / "paper_prep" / "tmp_adapt_ci"
ADAPT_099 = ROOT / "experiments" / "factowl_eval" / "adaptivity_lang"
ADAPT_COMPACT = TABLES / "adaptive_retention_sweep_compact.tsv"

FACTOWL_TSVS = {
    ("rivers", "en"): ROOT / "experiments" / "factowl_eval" / "pred_rivers_k99_allcompletions_finalonly.tsv",
    ("cars", "en"): ROOT / "experiments" / "factowl_eval" / "pred_cars_k99_allcompletions_finalonly.tsv",
    ("disasters", "en"): ROOT / "experiments" / "factowl_eval" / "pred_disasters_k99_allcompletions_finalonly.tsv",
    ("rivers", "zh"): ROOT / "experiments" / "factowl_eval" / "pred_rivers_zh_k99_allcompletions_finalonly.tsv",
    ("cars", "zh"): ROOT / "experiments" / "factowl_eval" / "pred_cars_zh_k99_allcompletions_finalonly.tsv",
    ("disasters", "zh"): ROOT / "experiments" / "factowl_eval" / "pred_disasters_zh_k99_allcompletions_finalonly.tsv",
}


def llmagg_k20_detail_path(domain: str, lang: str) -> Path:
    return LLMAGG_K20 / f"{lang}_{domain}" / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_details.tsv"


def scored_parquet_path(domain: str, lang: str) -> Path:
    return REASC_SCORED / f"ridic_{domain}_{lang}_k20" / "scored.parquet"


def adapt_prediction_path(lang: str, retention: float) -> Path:
    if retention == 0.95:
        return ADAPT_095 / f"{lang}_095" / f"{lang}_llmagg_adaptivity_lang_95_predictions.tsv"
    return ADAPT_099 / f"{lang}_llmagg_adaptivity_lang_99_predictions.tsv"


def load_topic_score_maps(domain: str, lang: str) -> tuple[dict[tuple[str, int], float], dict[str, float]]:
    comp_scores = load_factowl_completion_scores(FACTOWL_TSVS[(domain, lang)])
    llmagg_df = pd.read_csv(llmagg_k20_detail_path(domain, lang), sep="\t")
    llmagg_df = llmagg_df[llmagg_df["k"] == 20].drop_duplicates(subset=["topic"], keep="last")
    llmagg_scores = dict(zip(llmagg_df["topic"], llmagg_df["score"]))
    return comp_scores, llmagg_scores


def completion_logprob_value(lp: np.ndarray) -> float:
    arr = np.asarray(lp, dtype=float)
    if arr.size == 0:
        return float("-inf")
    return float(arr.mean())


def load_question_df(domain: str, lang: str) -> pd.DataFrame:
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "s-nlp___ri_di_c" / domain / "0.0.0"
    arrows = sorted(cache_root.glob("*/ri_di_c-test.arrow"))
    if not arrows:
        cache_root = ROOT / "cache" / "huggingface" / "datasets" / "s-nlp___ri_di_c" / domain / "0.0.0"
        arrows = sorted(cache_root.glob("*/ri_di_c-test.arrow"))
    if not arrows:
        raise FileNotFoundError(f"Could not locate cached RiDiC arrow files for {domain}")
    dset = Dataset.from_file(str(arrows[-1]))
    title_col = "title_zh" if lang == "zh" else "title_en"
    prefix = "请用一个段落告诉我你对" if lang == "zh" else "In a paragraph, could you tell me what you know about "
    suffix = "的了解。" if lang == "zh" else "?"
    questions = [f"{prefix}{title}{suffix}" for title in dset[title_col]]
    return pd.DataFrame({"topic": dset[title_col], "question": questions, "domain": domain})


def load_detail_df(domain: str, lang: str) -> pd.DataFrame:
    path = ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_details.tsv"
    df = pd.read_csv(path, sep="\t")
    df["domain"] = domain
    return df


def derive_examples(detail_df: pd.DataFrame, retention: float) -> pd.DataFrame:
    rows = []
    for domain, dom_df in detail_df.groupby("domain"):
        piv = (
            dom_df[["topic", "k", "score"]]
            .drop_duplicates(subset=["topic", "k"], keep="last")
            .pivot(index="topic", columns="k", values="score")
        )
        allowed_ks = sorted(int(k) for k in piv.columns.tolist())
        for topic, rec in piv.iterrows():
            for baseline_k in allowed_ks:
                baseline_score = rec[baseline_k]
                target = retention * baseline_score
                eligible = [k for k in allowed_ks if k <= baseline_k and rec[k] >= target]
                optimized_k = min(eligible) if eligible else baseline_k
                rows.append(
                    {
                        "domain": domain,
                        "topic": topic,
                        "baseline_k": baseline_k,
                        "baseline_score": baseline_score,
                        "optimized_k": optimized_k,
                    }
                )
    return pd.DataFrame(rows)


def featurize(question: str, lang: str, domain: str, baseline_k: int) -> dict[str, float]:
    q = question or ""
    chars = len(q)
    words = len(q.split())
    lowered = q.lower()
    feat = {
        "q_len_chars": float(chars),
        "q_len_words": float(words),
        "q_avg_word_length": float(chars / max(words, 1)),
        "q_num_digits": float(sum(ch.isdigit() for ch in q)),
        "q_num_punct": float(sum(ch in ".,!?;:()[]{}-_/\\'\"`、，。！？；：" for ch in q)),
        "q_has_year": float(int(any(tok.isdigit() and len(tok) == 4 for tok in q.split()))),
        "q_is_chinese": float(int(lang == "zh")),
        "starts_with_what": float(int(lowered.startswith("what"))),
        "starts_with_where": float(int(lowered.startswith("where"))),
        "starts_with_when": float(int(lowered.startswith("when"))),
        "starts_with_who": float(int(lowered.startswith("who"))),
        "starts_with_which": float(int(lowered.startswith("which"))),
        "contains_parentheses": float(int("(" in q or ")" in q)),
        "contains_comma": float(int("," in q)),
        "contains_hyphen": float(int("-" in q)),
        "baseline_k": float(baseline_k),
        "baseline_k_log2": float(np.log2(baseline_k)),
    }
    for d in DOMAINS:
        feat[f"domain_is_{d}"] = float(int(domain == d))
    return feat


def nearest_allowed(value: int, allowed: list[int]) -> int:
    return min(allowed, key=lambda k: (abs(k - value), k))


def make_bestofk_table() -> pd.DataFrame:
    rows = []
    for lang in LANGS:
        for domain in DOMAINS:
            comp_scores, llmagg_scores = load_topic_score_maps(domain, lang)
            df = pd.read_parquet(scored_parquet_path(domain, lang), columns=["topic", "all_logprobs"])
            eval_rows = []
            for row in df.itertuples(index=False):
                topic = row.topic
                if topic not in llmagg_scores:
                    continue
                logprobs = list(row.all_logprobs)
                usable = min(20, len(logprobs))
                if usable == 0:
                    continue
                lp_values = [completion_logprob_value(logprobs[idx]) for idx in range(usable)]
                selected_idx = int(np.argmax(lp_values))
                single = comp_scores.get((topic, 0))
                selected = comp_scores.get((topic, selected_idx))
                oracle = max(
                    (comp_scores.get((topic, idx), float("-inf")) for idx in range(usable)),
                    default=float("-inf"),
                )
                if single is None or selected is None or not np.isfinite(oracle):
                    continue
                eval_rows.append(
                    {
                        "topic": topic,
                        "single": float(single),
                        "bestofk_logprob": float(selected),
                        "oracle_best_single": float(oracle),
                        "llmagg_k20": float(llmagg_scores[topic]),
                        "selected_idx": selected_idx,
                    }
                )
            eval_df = pd.DataFrame(eval_rows)
            rows.append(
                {
                    "lang": lang,
                    "domain": domain,
                    "n_eval": int(len(eval_df)),
                    "single": float(eval_df["single"].mean()),
                    "bestofk_logprob": float(eval_df["bestofk_logprob"].mean()),
                    "llmagg_k20": float(eval_df["llmagg_k20"].mean()),
                    "oracle_best_single": float(eval_df["oracle_best_single"].mean()),
                    "selected_equals_first_rate": float((eval_df["selected_idx"] == 0).mean()),
                    "gain_vs_single": float((eval_df["bestofk_logprob"] - eval_df["single"]).mean()),
                    "gap_to_llmagg": float((eval_df["llmagg_k20"] - eval_df["bestofk_logprob"]).mean()),
                    "score_coverage": float(len(eval_df) / len(df)) if len(df) else np.nan,
                    "scoring_function": "avg_token_logprob",
                }
            )
    return pd.DataFrame(rows).sort_values(["lang", "domain"]).reset_index(drop=True)


def build_cross_domain_dataset(lang: str, retention: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    detail_frames = []
    feature_frames = []
    for domain in DOMAINS:
        q_df = load_question_df(domain, lang).copy()
        q_df["lang"] = lang
        detail_df = load_detail_df(domain, lang)
        ex_df = derive_examples(detail_df, retention)
        merged = q_df.merge(ex_df, on=["domain", "topic"], how="inner")
        merged["lang"] = lang
        feature_rows = [
            featurize(q, lang, d, int(k))
            for q, d, k in zip(merged["question"].fillna(""), merged["domain"], merged["baseline_k"])
        ]
        frames.append(merged)
        detail_frames.append(detail_df)
        feature_frames.append(pd.DataFrame(feature_rows))
    df = pd.concat(frames, ignore_index=True)
    detail_df = pd.concat(detail_frames, ignore_index=True)
    feature_df = pd.concat(feature_frames, ignore_index=True)
    mask = df["baseline_k"] == 99
    return df.loc[mask].reset_index(drop=True), detail_df, feature_df.loc[mask].reset_index(drop=True)


def compute_eval_metrics(test_df: pd.DataFrame) -> dict[str, float]:
    return {
        "n_eval": int(len(test_df)),
        "accuracy": float((test_df["pred_k"] == test_df["optimized_k"]).mean()),
        "avg_pred_k": float(test_df["pred_k"].mean()),
        "compute_savings_vs_k99": float(test_df["compute_savings"].mean()),
        "score_retention": float(test_df["score_retention"].mean()),
        "coverage": float((test_df["score_retention"] >= test_df["retention_target"]).mean()),
    }


def make_cross_domain_table() -> pd.DataFrame:
    compact = pd.read_csv(ADAPT_COMPACT, sep="\t")
    rows = []
    for lang in LANGS:
        for retention in RETENTIONS:
            df, detail_df, feature_df = build_cross_domain_dataset(lang, retention)
            allowed_ks = sorted(detail_df["k"].astype(int).unique().tolist())
            score_lookup = detail_df[["domain", "topic", "k", "score"]].copy()
            for heldout in DOMAINS:
                train_mask = df["domain"] != heldout
                test_mask = ~train_mask
                X_train = feature_df.loc[train_mask].values
                y_train = df.loc[train_mask, "optimized_k"].astype(int).values
                X_test = feature_df.loc[test_mask].values
                y_test = df.loc[test_mask, "optimized_k"].astype(int).values
                clf = RandomForestClassifier(
                    n_estimators=300,
                    random_state=13,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                )
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                test_df = df.loc[test_mask, ["lang", "domain", "topic", "question", "baseline_k", "baseline_score", "optimized_k"]].copy()
                test_df["pred_k"] = [nearest_allowed(int(v), allowed_ks) for v in pred]
                test_df["pred_k"] = np.minimum(test_df["pred_k"], test_df["baseline_k"])
                test_df = test_df.merge(
                    score_lookup.rename(columns={"k": "pred_k", "score": "pred_score"}),
                    on=["domain", "topic", "pred_k"],
                    how="left",
                )
                test_df["score_retention"] = test_df["pred_score"] / test_df["baseline_score"].replace(0, np.nan)
                test_df["score_retention"] = test_df["score_retention"].fillna(1.0)
                test_df["compute_savings"] = 1.0 - (test_df["pred_k"] / test_df["baseline_k"])
                test_df["retention_target"] = retention
                metrics = compute_eval_metrics(test_df)

                ref = compact[
                    (compact["lang"] == lang)
                    & (compact["domain"] == heldout)
                    & (compact["retention_target"] == retention)
                ].iloc[0]
                rows.append(
                    {
                        "lang": lang,
                        "retention_target": retention,
                        "heldout_domain": heldout,
                        **metrics,
                        "in_domain_compute_savings_vs_k99": float(ref["avg_compute_savings"]),
                        "in_domain_score_retention": float(ref["avg_score_retention"]),
                        "in_domain_coverage": float(ref["coverage_retention"]),
                        "delta_compute_savings_vs_in_domain": float(metrics["compute_savings_vs_k99"] - ref["avg_compute_savings"]),
                        "delta_score_retention_vs_in_domain": float(metrics["score_retention"] - ref["avg_score_retention"]),
                        "delta_coverage_vs_in_domain": float(metrics["coverage"] - ref["coverage_retention"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["lang", "retention_target", "heldout_domain"]).reset_index(drop=True)


def load_adapt_predictions(lang: str, retention: float) -> pd.DataFrame:
    df = pd.read_csv(adapt_prediction_path(lang, retention), sep="\t")
    df = df[df["baseline_k"] == 99].copy()
    df["lang"] = lang
    return df


def make_catastrophic_failure_table() -> pd.DataFrame:
    rows = []
    for lang in LANGS:
        score_maps = {}
        llmagg_maps = {}
        for domain in DOMAINS:
            comp_scores, llmagg_scores = load_topic_score_maps(domain, lang)
            score_maps[domain] = comp_scores
            llmagg_maps[domain] = llmagg_scores

        per_topic_rows = []
        for domain in DOMAINS:
            topics = set(llmagg_maps[domain].keys())
            for topic in topics:
                single = score_maps[domain].get((topic, 0))
                llmagg = llmagg_maps[domain].get(topic)
                if single is None or llmagg is None:
                    continue
                delta = float(llmagg - single)
                if delta <= -0.5:
                    group = "catastrophic_loss"
                elif delta < 0:
                    group = "loss"
                else:
                    group = "win_or_tie"
                per_topic_rows.append(
                    {
                        "lang": lang,
                        "domain": domain,
                        "topic": topic,
                        "single_score": float(single),
                        "llmagg_k20_score": float(llmagg),
                        "delta": delta,
                        "failure_group": group,
                    }
                )
        topic_df = pd.DataFrame(per_topic_rows)

        for retention in RETENTIONS:
            pred_df = load_adapt_predictions(lang, retention)
            merged = pred_df.merge(topic_df, on=["lang", "domain", "topic"], how="inner")
            for domain in ["ALL", *DOMAINS]:
                scope = merged if domain == "ALL" else merged[merged["domain"] == domain]
                for group in ["catastrophic_loss", "loss", "win_or_tie"]:
                    sub = scope[scope["failure_group"] == group]
                    if sub.empty:
                        continue
                    rows.append(
                        {
                            "lang": lang,
                            "retention_target": retention,
                            "domain": domain,
                            "failure_group": group,
                            "n_topics": int(len(sub)),
                            "avg_pred_k": float(sub["pred_k"].mean()),
                            "median_pred_k": float(sub["pred_k"].median()),
                            "pct_pred_le_20": float((sub["pred_k"] <= 20).mean()),
                            "pct_pred_ge_64": float((sub["pred_k"] >= 64).mean()),
                            "avg_compute_savings_vs_k99": float(sub["compute_savings"].mean()),
                            "avg_adaptive_score_retention": float(sub["score_retention"].mean()),
                            "avg_single_score": float(sub["single_score"].mean()),
                            "avg_llmagg_k20_score": float(sub["llmagg_k20_score"].mean()),
                            "avg_delta": float(sub["delta"].mean()),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["lang", "retention_target", "domain", "failure_group"]).reset_index(drop=True)


def write_repro_statement() -> None:
    text = """# Reproducibility Statement

We will release the code used for data preparation, FactOwl evaluation, LLMAgg aggregation, adaptive-routing analysis, and table/figure generation for this paper.

Planned release contents:

- experiment scripts and configuration files
- prompt templates for Single, LLMAgg, and auxiliary baselines
- derived paper tables and plotting scripts
- evaluation wrappers for RiDiC and FactOwl
- notes describing any post-hoc corrections, including the audited ReASC evaluator fix

For data and model outputs:

- RiDiC itself will be referenced through its public release
- derived per-topic result tables and non-proprietary metadata will be released
- any model outputs or caches that are restricted by upstream licenses or API terms will be documented with reproduction instructions rather than redistributed directly

This is intended to make every reported table and figure in the paper reproducible from the public benchmark plus the released code and documented environment.
"""
    APPENDIX.mkdir(parents=True, exist_ok=True)
    (APPENDIX / "reproducibility_statement.md").write_text(text)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    bestofk = make_bestofk_table()
    cross_domain = make_cross_domain_table()
    catastrophic = make_catastrophic_failure_table()

    bestofk.to_csv(TABLES / "bestofk_logprob_k20.tsv", sep="\t", index=False)
    cross_domain.to_csv(TABLES / "adaptive_cross_domain.tsv", sep="\t", index=False)
    catastrophic.to_csv(TABLES / "adaptive_catastrophic_failure_analysis.tsv", sep="\t", index=False)

    write_repro_statement()


if __name__ == "__main__":
    main()
