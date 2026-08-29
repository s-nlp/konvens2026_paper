#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset


from _root import find_experiment_root

ROOT = find_experiment_root()
OUTPUTS = ROOT / "outputs" / "reasc_posthoc"
PAPER_PREP = ROOT / "paper_prep"
TABLES = PAPER_PREP / "tables"
APPENDIX = PAPER_PREP / "appendix"
FACTOWL = ROOT / "experiments" / "factowl_eval"


@dataclass(frozen=True)
class Pair:
    domain: str
    lang: str
    llmagg: float


PAIRS = [
    Pair("cars", "en", 0.654924),
    Pair("cars", "zh", 0.631724),
    Pair("disasters", "en", 0.612313),
    Pair("disasters", "zh", 0.615923),
    Pair("rivers", "en", 0.463175),
    Pair("rivers", "zh", 0.441869),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def make_reasc_table() -> pd.DataFrame:
    rows = []
    for pair in PAIRS:
        old_path = OUTPUTS / f"ridic_{pair.domain}_{pair.lang}_k20_eval" / "reasc_summary.json"
        new_path = OUTPUTS / f"ridic_{pair.domain}_{pair.lang}_k20_eval_v2" / "reasc_summary.json"
        old = load_json(old_path) if old_path.exists() else {}
        new = load_json(new_path)
        rows.append(
            {
                "lang": pair.lang,
                "domain": pair.domain,
                "Single": round(new["single_score"], 6),
                "LLMAgg": round(pair.llmagg, 6),
                "ReASC": round(new["reasc_score"], 6),
                "old_ReASC": round(old.get("reasc_score"), 6) if old else None,
                "delta_vs_old": round(new["reasc_score"] - old.get("reasc_score", new["reasc_score"]), 6),
                "delta_vs_single": round(new["reasc_score"] - new["single_score"], 6),
                "avg_samples_used": round(new["avg_samples_used"], 3),
                "stage1_rate": round(new["stage1_rate"], 3),
                "exhausted_rate": round(1.0 - new["stage1_rate"], 3),
                "selected_equals_first_rate": round(new["selected_equals_first_rate"], 3),
                "score_coverage": round(new["score_coverage"], 3),
                "audit_note": "partial_factowl_coverage" if new["score_coverage"] < 0.95 else "full_factowl_coverage",
            }
        )
    return pd.DataFrame(rows)


def text_snippet(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def make_provenance_table() -> pd.DataFrame:
    pred = pd.read_csv(OUTPUTS / "ridic_cars_en_k20_eval_v2" / "reasc_predictions.tsv", sep="\t")
    scored = pd.read_parquet(OUTPUTS / "ridic_cars_en_k20" / "scored.parquet")
    llmagg = pd.read_csv(FACTOWL / "pred_cars_en_k99_llmagg_aggk16_finalonly_answers.tsv", sep="\t")
    ridic = load_dataset("s-nlp/RiDiC", "LLM_generations_cars", split="test").to_pandas()

    chosen = (
        pred[(pred["single_completion_idx"] != pred["reasc_completion_idx"]) & (pred["reasc_score"] > pred["single_score"])]
        .assign(delta=lambda x: x["reasc_score"] - x["single_score"])
        .sort_values("delta", ascending=False)
        .head(5)
        .copy()
    )

    scored = scored[["topic", "question", "all_completions"]].copy()
    llmagg = llmagg[["topic", "aggregated_answer"]].drop_duplicates("topic")
    ridic = ridic[["title_en", "llama3.1:8b", "qwen2.5:7b"]].rename(columns={"title_en": "topic"})

    merged = (
        chosen.merge(scored, on=["topic", "question"], how="left")
        .merge(llmagg, on="topic", how="left")
        .merge(ridic, on="topic", how="left")
        .copy()
    )

    merged["local_single_text"] = merged.apply(
        lambda r: r["all_completions"][int(r["single_completion_idx"])], axis=1
    )
    merged["local_reasc_text"] = merged.apply(
        lambda r: r["all_completions"][int(r["reasc_completion_idx"])], axis=1
    )

    out = merged[
        [
            "topic",
            "question",
            "single_completion_idx",
            "reasc_completion_idx",
            "single_score",
            "reasc_score",
            "delta",
            "local_single_text",
            "local_reasc_text",
            "aggregated_answer",
            "llama3.1:8b",
            "qwen2.5:7b",
        ]
    ].rename(
        columns={
            "aggregated_answer": "local_llmagg_k16_text",
            "llama3.1:8b": "released_ridic_llama31_8b_text",
            "qwen2.5:7b": "released_ridic_qwen25_7b_text",
        }
    )
    return out


def make_note(reasc_df: pd.DataFrame, provenance_df: pd.DataFrame) -> str:
    en_cars = reasc_df[(reasc_df["lang"] == "en") & (reasc_df["domain"] == "cars")].iloc[0]
    zh_cars = reasc_df[(reasc_df["lang"] == "zh") & (reasc_df["domain"] == "cars")].iloc[0]
    examples = []
    for row in provenance_df.itertuples(index=False):
        examples.append(
            f"- `{row.topic}`: local Single used idx `{row.single_completion_idx}`, ReASC used idx `{row.reasc_completion_idx}`, "
            f"FactOwl score moved `{row.single_score:.3f} -> {row.reasc_score:.3f}`."
        )

    return "\n".join(
        [
            "# ReASC Investigation Note",
            "",
            "## Provenance",
            "",
            "The paper's ReASC numbers come from the local post-hoc ReASC path under `outputs/reasc_posthoc/ridic_*_k20_eval/`, "
            "which is built from the local multi-sample traces and the local FactOwl all-completions files in `experiments/factowl_eval/`.",
            "They do not come from the released `s-nlp/RiDiC` single-completion columns.",
            "",
            "## Root Cause",
            "",
            "The original evaluator joined ReASC selections to the repeated topic-level `score` column in the FactOwl TSVs, "
            "instead of recomputing per-completion scores from atomic-fact decisions. That made ReASC look numerically identical "
            "or nearly identical to Single even when it selected different completion indices.",
            "",
            "The corrected evaluator now recomputes completion scores from fact decisions and asserts that each ReASC example sees more than one candidate.",
            "",
            "## Key Findings",
            "",
            f"- EN cars changed from old ReASC `{en_cars.old_ReASC:.6f}` to corrected `{en_cars.ReASC:.6f}`, versus Single `{en_cars.Single:.6f}`.",
            f"- ZH cars remains close numerically (`{zh_cars.Single:.6f}` vs `{zh_cars.ReASC:.6f}`), but not because ReASC always selects the first sample. "
            f"`selected_equals_first_rate={zh_cars.selected_equals_first_rate:.3f}`; the main caveat there is partial FactOwl coverage (`score_coverage={zh_cars.score_coverage:.3f}`).",
            "- ReASC does not generally degenerate to candidate 0. The corrected summaries show `selected_equals_first_rate` between 0.130 and 0.422.",
            "",
            "## Suggested Robustness Paragraph",
            "",
            "We audited the post-hoc ReASC baseline after observing near-identity with the Single baseline on cars. "
            "The issue was not that ReASC always stopped on the first candidate. Instead, the original evaluator looked up a repeated topic-level FactOwl score, "
            "which hid per-completion differences. After recomputing completion-level scores from atomic-fact decisions, ReASC remained close to Single overall, "
            "but no longer numerically identical: for example, EN cars increased from 0.344588 to 0.353859 while Single stayed at 0.344196. "
            "ZH results remain harder to interpret because the available FactOwl all-completion files cover only a subset of topics, so those rows are reported with explicit coverage values.",
            "",
            "## Five EN-cars provenance examples",
            "",
            *examples,
            "",
            "## Appendix Table Source",
            "",
            "- Full table: `paper_prep/appendix/reasc_provenance_comparison_en_cars.tsv`",
            "- Note: local LLMAgg text is taken from the stored `aggk16` answer archive because no `aggk20` answer TSV is present in `experiments/factowl_eval/`.",
        ]
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    APPENDIX.mkdir(parents=True, exist_ok=True)

    reasc_df = make_reasc_table()
    provenance_df = make_provenance_table()

    reasc_df.to_csv(TABLES / "reasc_comparison_k20.tsv", sep="\t", index=False)
    provenance_df.to_csv(APPENDIX / "reasc_provenance_comparison_en_cars.tsv", sep="\t", index=False)
    (APPENDIX / "reasc_investigation_note.md").write_text(make_note(reasc_df, provenance_df))

    preview = provenance_df.copy()
    for col in [
        "local_single_text",
        "local_reasc_text",
        "local_llmagg_k16_text",
        "released_ridic_llama31_8b_text",
        "released_ridic_qwen25_7b_text",
    ]:
        preview[col] = preview[col].map(text_snippet)
    preview.to_csv(APPENDIX / "reasc_provenance_comparison_en_cars_preview.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
