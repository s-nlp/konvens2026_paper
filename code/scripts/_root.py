"""Shared path resolution: locate the LongFT experiment checkout.

Used by every analysis script in this directory. The experiment checkout is
the one containing ``experiments/factowl_eval`` (the LongFT repo with the
saved per-topic TSVs); the fallback preserves the historic ``parents[2]``
behaviour when the marker is absent.
"""

from pathlib import Path


def find_experiment_root() -> Path:
    for cand in Path(__file__).resolve().parents:
        if (cand / "experiments" / "factowl_eval").is_dir():
            return cand
    return Path(__file__).resolve().parents[2]
