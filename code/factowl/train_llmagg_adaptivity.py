#!/usr/bin/env python3
"""Train a minimal question-only adaptivity policy for llmagg."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import run_factowl_eval as rfe


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=rfe.DOMAIN_CFG.keys(), required=True)
    p.add_argument("--lang", default="en", choices=["en", "zh"])
    p.add_argument("--detail-tsv", default=None, help="Per-topic llmagg detail TSV from run_llmagg_curve.py")
    p.add_argument("--output-dir", default="/workspace/longft/experiments/factowl_eval/adaptivity")
    p.add_argument("--sample", type=int, default=-1)
    p.add_argument("--target-frac", type=float, default=0.95, help="Target fraction of llmagg@max_k score")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def default_detail_tsv(args) -> Path:
    return Path(args.output_dir).parent / f"pred_{args.domain}_{args.lang}_k99_llmaggcurve_finalonly_details.tsv"


def featurize(question: str, lang: str) -> dict:
    q = question or ""
    chars = len(q)
    words = len(q.split())
    lowered = q.lower()
    return {
        "q_len_chars": chars,
        "q_len_words": words,
        "q_avg_word_length": chars / max(words, 1),
        "q_num_digits": sum(ch.isdigit() for ch in q),
        "q_num_punct": sum(ch in ".,!?;:()[]{}-_/\\'\"`、，。！？；：" for ch in q),
        "q_has_year": int(any(tok.isdigit() and len(tok) == 4 for tok in q.split())),
        "q_is_chinese": int(lang == "zh"),
        "starts_with_what": int(lowered.startswith("what")),
        "starts_with_where": int(lowered.startswith("where")),
        "starts_with_when": int(lowered.startswith("when")),
        "starts_with_who": int(lowered.startswith("who")),
        "starts_with_which": int(lowered.startswith("which")),
        "contains_parentheses": int("(" in q or ")" in q),
        "contains_comma": int("," in q),
        "contains_hyphen": int("-" in q),
    }


def load_question_df(args) -> pd.DataFrame:
    meta_cfg, repo = rfe.DOMAIN_CFG[args.domain][args.lang]
    ds = rfe.load_sc(meta_cfg, repo, args.lang)
    if args.sample and args.sample > 0:
        ds = ds.select(range(args.sample))
    title_col = "title_zh" if args.lang == "zh" else "title_en"
    return pd.DataFrame({"topic": ds[title_col], "question": ds["question"]})


def derive_targets(detail_df: pd.DataFrame, target_frac: float) -> pd.DataFrame:
    max_k = int(detail_df["k"].max())
    final_scores = detail_df[detail_df["k"] == max_k][["topic", "score"]].rename(columns={"score": "score_max"})
    merged = detail_df.merge(final_scores, on="topic", how="left")
    merged["target_score"] = merged["score_max"] * target_frac
    merged["hit_target"] = merged["score"] >= merged["target_score"]
    targets = (
        merged[merged["hit_target"]]
        .sort_values(["topic", "k"])
        .groupby("topic", as_index=False)
        .first()[["topic", "k", "score_max", "target_score"]]
        .rename(columns={"k": "target_k"})
    )
    missing_topics = sorted(set(final_scores["topic"]) - set(targets["topic"]))
    if missing_topics:
        fallback = final_scores[final_scores["topic"].isin(missing_topics)].copy()
        fallback["target_score"] = fallback["score"]
        fallback["target_k"] = max_k
        targets = pd.concat([targets, fallback[["topic", "target_k", "score_max", "target_score"]]], ignore_index=True)
    return targets


def nearest_allowed(value: int, allowed: list[int]) -> int:
    return min(allowed, key=lambda k: (abs(k - value), k))


def main():
    args = parse_args()
    detail_path = Path(args.detail_tsv) if args.detail_tsv else default_detail_tsv(args)
    if not detail_path.exists():
        raise FileNotFoundError(f"Detail TSV not found: {detail_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_df = pd.read_csv(detail_path, sep="\t")
    required = {"topic", "k", "score"}
    missing = required - set(detail_df.columns)
    if missing:
        raise ValueError(f"Missing required detail columns: {sorted(missing)}")

    question_df = load_question_df(args)
    target_df = derive_targets(detail_df, args.target_frac)
    train_df = question_df.merge(target_df, on="topic", how="inner")
    if train_df.empty:
        raise ValueError("No overlap between llmagg topic details and question dataset")

    feature_rows = [featurize(q, args.lang) for q in train_df["question"].fillna("")]
    feature_df = pd.DataFrame(feature_rows)
    feature_cols = list(feature_df.columns)
    X = feature_df.values
    y = train_df["target_k"].astype(int).values

    unique_classes, class_counts = np.unique(y, return_counts=True)
    test_n = int(np.ceil(len(y) * args.test_size)) if args.test_size < 1 else int(args.test_size)
    stratify = y if (
        len(unique_classes) > 1
        and class_counts.min() >= 2
        and test_n >= len(unique_classes)
    ) else None
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, train_df, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=args.seed,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    allowed_ks = sorted(detail_df["k"].astype(int).unique().tolist())
    pred_df = df_test[["topic", "question"]].copy()
    pred_df["target_k"] = y_test
    pred_df["pred_k"] = [nearest_allowed(int(v), allowed_ks) for v in y_pred]

    score_lookup = detail_df[["topic", "k", "score"]].copy()
    pred_df = pred_df.merge(score_lookup.rename(columns={"k": "pred_k", "score": "pred_score"}), on=["topic", "pred_k"], how="left")
    max_k = max(allowed_ks)
    pred_df = pred_df.merge(score_lookup[score_lookup["k"] == max_k][["topic", "score"]].rename(columns={"score": "full_score"}), on="topic", how="left")
    pred_df["score_gap"] = pred_df["full_score"] - pred_df["pred_score"]
    pred_df["compute_savings"] = 1.0 - (pred_df["pred_k"] / max_k)

    summary = {
        "domain": args.domain,
        "lang": args.lang,
        "detail_tsv": str(detail_path),
        "n_examples": int(len(train_df)),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "target_frac": float(args.target_frac),
        "max_k": int(max_k),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "avg_target_k_test": float(np.mean(y_test)),
        "avg_pred_k_test": float(pred_df["pred_k"].mean()),
        "avg_pred_score_test": float(pred_df["pred_score"].mean()),
        "avg_full_score_test": float(pred_df["full_score"].mean()),
        "avg_score_gap_test": float(pred_df["score_gap"].mean()),
        "avg_compute_savings_test": float(pred_df["compute_savings"].mean()),
        "feature_columns": feature_cols,
    }

    stem = f"{args.domain}_{args.lang}_llmagg_adaptivity_qonly"
    pred_path = out_dir / f"{stem}_predictions.tsv"
    summary_path = out_dir / f"{stem}_summary.json"
    report_path = out_dir / f"{stem}_report.txt"
    model_path = out_dir / f"{stem}.joblib"

    pred_df.to_csv(pred_path, sep="\t", index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred))
    joblib.dump({"model": clf, "feature_columns": feature_cols, "allowed_ks": allowed_ks}, model_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_predictions={pred_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_report={report_path}")
    print(f"saved_model={model_path}")


if __name__ == "__main__":
    main()
