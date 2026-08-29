#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit


from _root import find_experiment_root

ROOT = find_experiment_root()

DOMAINS = ["rivers", "cars", "disasters"]
LANGS = ["en", "zh"]
RETENTIONS = [0.95, 0.99]
ALLOWED_K = [1, 2, 4, 8, 16, 20]
N_SPLITS = 5
TEST_SIZE = 0.3
SEED = 13


def load_detail_df(domain: str, lang: str) -> pd.DataFrame:
    path = ROOT / "experiments" / "factowl_eval" / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_details.tsv"
    df = pd.read_csv(path, sep="\t")
    df = df[df["k"].isin([1, 2, 4, 8, 16])].copy()
    k20_path = ROOT / "paper_prep" / "experiments" / "llmagg_k20" / f"{lang}_{domain}" / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_details.tsv"
    k20_df = pd.read_csv(k20_path, sep="\t")
    k20_df = k20_df[k20_df["k"] == 20].copy()
    df = pd.concat([df, k20_df], ignore_index=True)
    df["domain"] = domain
    return df


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


def derive_k20_examples(detail_df: pd.DataFrame, retention: float) -> pd.DataFrame:
    rows = []
    for domain, dom_df in detail_df.groupby("domain"):
        piv = (
            dom_df[["topic", "k", "score"]]
            .drop_duplicates(subset=["topic", "k"], keep="last")
            .pivot(index="topic", columns="k", values="score")
        )
        for topic, rec in piv.iterrows():
            if 20 not in rec.index:
                continue
            baseline_score = rec[20]
            target = retention * baseline_score
            eligible = [k for k in ALLOWED_K if k in rec.index and rec[k] >= target]
            optimized_k = min(eligible) if eligible else 20
            rows.append(
                {
                    "domain": domain,
                    "topic": topic,
                    "baseline_k": 20,
                    "baseline_score": baseline_score,
                    "optimized_k": optimized_k,
                }
            )
    return pd.DataFrame(rows)


def build_dataset(lang: str, retention: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qdfs = [load_question_df(domain, lang) for domain in DOMAINS]
    ddfs = [load_detail_df(domain, lang) for domain in DOMAINS]
    question_df = pd.concat(qdfs, ignore_index=True)
    detail_df = pd.concat(ddfs, ignore_index=True)
    ex_df = derive_k20_examples(detail_df, retention)
    df = question_df.merge(ex_df, on=["domain", "topic"], how="inner")
    feature_rows = [
        featurize(q, lang, d, 20)
        for q, d in zip(df["question"].fillna(""), df["domain"])
    ]
    feature_df = pd.DataFrame(feature_rows)
    return df, detail_df, feature_df


def evaluate_split(df: pd.DataFrame, detail_df: pd.DataFrame, feature_df: pd.DataFrame, train_idx, test_idx, retention: float, seed: int):
    X = feature_df.values
    y = df["optimized_k"].astype(int).values
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_df = df.iloc[test_idx].copy()

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    test_df["pred_k"] = [nearest_allowed(int(v), ALLOWED_K) for v in pred]
    score_lookup = detail_df[["domain", "topic", "k", "score"]].copy()
    test_df = test_df.merge(
        score_lookup.rename(columns={"k": "pred_k", "score": "pred_score"}),
        on=["domain", "topic", "pred_k"],
        how="left",
    )
    test_df["score_retention"] = test_df["pred_score"] / test_df["baseline_score"].replace(0, np.nan)
    test_df["score_retention"] = test_df["score_retention"].fillna(1.0)
    test_df["compute_savings_vs_k20"] = 1.0 - (test_df["pred_k"] / 20.0)
    test_df["compute_savings_vs_k20_plusagg"] = 1.0 - ((test_df["pred_k"] + 1.0) / 21.0)
    test_df["coverage"] = (test_df["score_retention"] >= retention).astype(float)
    return clf, test_df, float((pred == y_test).mean())


def main():
    out_dir = ROOT / "paper_prep" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    pred_rows = []
    for lang in LANGS:
        for retention in RETENTIONS:
            df, detail_df, feature_df = build_dataset(lang, retention)
            strat_labels = (
                df["domain"].astype(str)
                + "|"
                + df["optimized_k"].astype(str)
            )
            counts = strat_labels.value_counts()
            rare = counts[counts < 2].index
            strat_labels = strat_labels.where(~strat_labels.isin(rare), other="other")
            splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)

            fold_preds = []
            fold_accs = []
            for fold_id, (train_idx, test_idx) in enumerate(splitter.split(feature_df.values, strat_labels), start=1):
                _, pred_df, acc = evaluate_split(df, detail_df, feature_df, train_idx, test_idx, retention, SEED + fold_id)
                pred_df["fold"] = fold_id
                pred_df["lang"] = lang
                pred_df["retention_target"] = retention
                fold_preds.append(pred_df)
                fold_accs.append(acc)
            pred_df = pd.concat(fold_preds, ignore_index=True)
            pred_rows.append(pred_df)
            for domain, sub in pred_df.groupby("domain"):
                summary_rows.append(
                    {
                        "lang": lang,
                        "retention_target": retention,
                        "domain": domain,
                        "n_eval": int(len(sub)),
                        "accuracy": float(np.mean(fold_accs)),
                        "avg_pred_k": float(sub["pred_k"].mean()),
                        "compute_savings_vs_k20": float(sub["compute_savings_vs_k20"].mean()),
                        "compute_savings_vs_k20_plusagg": float(sub["compute_savings_vs_k20_plusagg"].mean()),
                        "score_retention": float(sub["score_retention"].mean()),
                        "coverage": float(sub["coverage"].mean()),
                    }
                )
            summary_rows.append(
                {
                    "lang": lang,
                    "retention_target": retention,
                    "domain": "ALL",
                    "n_eval": int(len(pred_df)),
                    "accuracy": float(np.mean(fold_accs)),
                    "avg_pred_k": float(pred_df["pred_k"].mean()),
                    "compute_savings_vs_k20": float(pred_df["compute_savings_vs_k20"].mean()),
                    "compute_savings_vs_k20_plusagg": float(pred_df["compute_savings_vs_k20_plusagg"].mean()),
                    "score_retention": float(pred_df["score_retention"].mean()),
                    "coverage": float(pred_df["coverage"].mean()),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["lang", "retention_target", "domain"]).reset_index(drop=True)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    summary_path = out_dir / "adaptive_k20_controller.tsv"
    pred_path = out_dir / "adaptive_k20_controller_predictions.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    pred_df.to_csv(pred_path, sep="\t", index=False)
    print(summary_df.to_string(index=False))
    print(f"saved_summary={summary_path}")
    print(f"saved_predictions={pred_path}")


if __name__ == "__main__":
    main()
