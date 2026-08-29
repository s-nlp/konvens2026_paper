#!/usr/bin/env python3
"""Train one question-only llmagg adaptivity model per language.

Each training example is (question, domain, baseline_k) -> optimized_k, where
optimized_k is the smallest j <= baseline_k that preserves a target fraction of
the llmagg score achieved at baseline_k.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

import run_factowl_eval as rfe
import train_llmagg_adaptivity as tla


DOMAINS = ["rivers", "cars", "disasters"]


def _load_local_question_df(domain: str, lang: str) -> pd.DataFrame | None:
    """Load RiDiC questions from the local HF Arrow cache if available."""
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "s-nlp___ri_di_c" / domain / "0.0.0"
    arrows = sorted(cache_root.glob("*/ri_di_c-test.arrow"))
    if not arrows:
        return None
    from datasets import Dataset
    dset = Dataset.from_file(str(arrows[-1]))
    title_col = "title_zh" if lang == "zh" else "title_en"
    prefix = "请用一个段落告诉我你对" if lang == "zh" else "In a paragraph, could you tell me what you know about "
    suffix = "的了解。" if lang == "zh" else "?"
    questions = [f"{prefix}{title}{suffix}" for title in dset[title_col]]
    return pd.DataFrame({"topic": dset[title_col], "question": questions, "domain": domain})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="en", choices=["en", "zh"])
    p.add_argument("--detail-dir", default="/workspace/longft/experiments/factowl_eval")
    p.add_argument("--output-dir", default="/workspace/longft/experiments/factowl_eval/adaptivity_lang")
    p.add_argument("--retention", type=float, default=0.99)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def load_question_df(domain: str, lang: str) -> pd.DataFrame:
    local_df = _load_local_question_df(domain, lang)
    if local_df is not None:
        return local_df
    meta_cfg, repo = rfe.DOMAIN_CFG[domain][lang]
    ds = rfe.load_sc(meta_cfg, repo, lang)
    title_col = "title_zh" if lang == "zh" else "title_en"
    return pd.DataFrame({"topic": ds[title_col], "question": ds["question"], "domain": domain})


def load_detail_df(detail_dir: Path, domain: str, lang: str) -> pd.DataFrame:
    path = detail_dir / f"pred_{domain}_{lang}_k99_llmaggcurve_finalonly_details.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing detail TSV: {path}")
    df = pd.read_csv(path, sep="\t")
    df["domain"] = domain
    return df


def derive_examples(detail_df: pd.DataFrame, retention: float) -> pd.DataFrame:
    rows = []
    for domain, dom_df in detail_df.groupby("domain"):
        max_k = int(dom_df["k"].max())
        score_lookup = dom_df[["topic", "k", "score"]].copy()
        piv = score_lookup.drop_duplicates(subset=["topic", "k"], keep="last").pivot(index="topic", columns="k", values="score")
        allowed_ks = sorted([int(k) for k in piv.columns.tolist()])
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


def featurize_row(question: str, lang: str, domain: str, baseline_k: int) -> dict:
    feat = tla.featurize(question, lang)
    for d in DOMAINS:
        feat[f"domain_is_{d}"] = int(domain == d)
    feat["baseline_k"] = baseline_k
    feat["baseline_k_log2"] = float(np.log2(baseline_k))
    return feat


def build_dataset(lang: str, detail_dir: Path, retention: float):
    qdfs = [load_question_df(domain, lang) for domain in DOMAINS]
    ddfs = [load_detail_df(detail_dir, domain, lang) for domain in DOMAINS]
    question_df = pd.concat(qdfs, ignore_index=True)
    detail_df = pd.concat(ddfs, ignore_index=True)
    ex_df = derive_examples(detail_df, retention)
    df = question_df.merge(ex_df, on=["domain", "topic"], how="inner")
    feature_rows = [
        featurize_row(q, lang, d, int(k))
        for q, d, k in zip(df["question"].fillna(""), df["domain"], df["baseline_k"])
    ]
    feature_df = pd.DataFrame(feature_rows)
    return df, detail_df, feature_df


def evaluate_split(df: pd.DataFrame, detail_df: pd.DataFrame, feature_df: pd.DataFrame, train_idx, test_idx, seed: int):
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

    allowed_ks = sorted(detail_df["k"].astype(int).unique().tolist())
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
    return clf, test_df, accuracy_score(y_test, pred)


def make_strat_labels(df: pd.DataFrame) -> pd.Series:
    labels = df["baseline_k"].astype(str) + "|" + df["optimized_k"].astype(str)
    counts = labels.value_counts()
    rare = counts[counts < 2].index
    labels = labels.where(~labels.isin(rare), other="other")
    return labels


def plot_curve(summary_df: pd.DataFrame, out_path: Path, lang: str, retention: float):
    fig, ax = plt.subplots(figsize=(7, 5))
    for domain in DOMAINS:
        sub = summary_df[summary_df["domain"] == domain].sort_values("baseline_k")
        ax.plot(sub["baseline_k"], sub["avg_compute_savings"], marker="o", linewidth=2, label=domain)
    ax.set_xlabel("Baseline k")
    ax.set_ylabel("Realized compute savings")
    ax.set_title(f"{lang.upper()} adaptivity savings at {int(retention*100)}% retention")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = Path(args.detail_dir)

    df, detail_df, feature_df = build_dataset(args.lang, detail_dir, args.retention)
    strat_labels = make_strat_labels(df)
    splitter = StratifiedShuffleSplit(n_splits=args.n_splits, test_size=args.test_size, random_state=args.seed)

    all_preds = []
    fold_summaries = []
    models = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(feature_df.values, strat_labels), start=1):
        model, pred_df, acc = evaluate_split(df, detail_df, feature_df, train_idx, test_idx, seed=args.seed + fold_id)
        pred_df["fold"] = fold_id
        all_preds.append(pred_df)
        models.append(model)
        fold_summaries.append({"fold": fold_id, "accuracy": float(acc), "n_test": int(len(pred_df))})

    pred_df = pd.concat(all_preds, ignore_index=True)
    curve_df = (
        pred_df.groupby(["domain", "baseline_k"], as_index=False)
        .agg(
            avg_compute_savings=("compute_savings", "mean"),
            avg_score_retention=("score_retention", "mean"),
            coverage_retention=("score_retention", lambda s: float((s >= args.retention).mean())),
            n_eval=("topic", "count"),
        )
    )

    stem = f"{args.lang}_llmagg_adaptivity_lang_{int(args.retention*100)}"
    pred_path = out_dir / f"{stem}_predictions.tsv"
    curve_path = out_dir / f"{stem}_curve.tsv"
    summary_path = out_dir / f"{stem}_summary.json"
    plot_path = out_dir / f"{stem}_savings_curve.png"
    model_path = out_dir / f"{stem}.joblib"

    pred_df.to_csv(pred_path, sep="\t", index=False)
    curve_df.to_csv(curve_path, sep="\t", index=False)
    plot_curve(curve_df, plot_path, args.lang, args.retention)
    summary = {
        "lang": args.lang,
        "retention": args.retention,
        "n_examples": int(len(df)),
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "folds": fold_summaries,
        "feature_columns": list(feature_df.columns),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    joblib_payload = {
        "models": models,
        "feature_columns": list(feature_df.columns),
        "domains": DOMAINS,
        "retention": args.retention,
    }
    import joblib
    joblib.dump(joblib_payload, model_path)

    print(curve_df.to_string(index=False))
    print(f"saved_predictions={pred_path}")
    print(f"saved_curve={curve_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_plot={plot_path}")
    print(f"saved_model={model_path}")


if __name__ == "__main__":
    main()
