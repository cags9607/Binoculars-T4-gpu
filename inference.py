#!/usr/bin/env python3
"""
Text Inference — Binoculars (Falcon-on-T4, 4-bit, sequential)

Input:
    A CSV file. By default, the first column is treated as text.
    You can override with --text-col.

Output:
    CSV with columns:
        bino_score:  float  (higher = more human-like; lower = more AI-like)
        bino_pred:   Int64  (1 = AI-generated, 0 = human-generated)
        bino_mode:   str    ("low-fpr" or "accuracy")

Notes:
    - This uses custom FalconT4Binoculars class for scoring.
    - Scores are computed as ppl / x_ppl (higher → more human).
    - Prediction rule: score < threshold → AI (1), else Human (0).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch  # noqa: F401

from experimental.falcon_t4_binoculars import FalconT4Binoculars


def chunk_indices(n: int, batch_size: int):
    """Yield (start, end) index tuples for batching."""
    for i in range(0, n, batch_size):
        yield i, min(i + batch_size, n)


def coerce_text(x) -> Optional[str]:
    """Return a clean string or None if invalid/empty."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    return s if s else None


def predict_texts(
    df: pd.DataFrame,
    bino: FalconT4Binoculars,
    text_col: str,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Apply Binoculars scoring to a text column in a DataFrame.

    Adds columns:
        - bino_score (Float64)
        - bino_pred  (Int64)   1=AI-generated, 0=human-generated
        - bino_mode  (str)
        - bino_threshold (float)
    """
    n = len(df)
    out = df.copy().reset_index(drop=True)

    # Prepare output columns with NA-able dtypes
    out["bino_score"] = pd.Series([pd.NA] * n, dtype="Float64")
    out["bino_pred"] = pd.Series([pd.NA] * n, dtype="Int64")
    out["bino_mode"] = bino.mode if hasattr(bino, "mode") else pd.NA
    
    # We’ll try to read bino.threshold and fallback gracefully.
    threshold_value = getattr(bino, "threshold", None)
   

    skipped = []  # (row_idx, reason)

    # Build an index list of valid texts (so we can preserve alignment cleanly)
    valid_indices: List[int] = []
    valid_texts: List[str] = []
    for i in range(n):
        txt = coerce_text(out.at[i, text_col])
        if txt is None:
            skipped.append((i, "empty/NaN text"))
            continue
        valid_indices.append(i)
        valid_texts.append(txt)

    # Early exit if nothing to score
    if not valid_texts:
        if skipped:
            print(f"\n⚠ Skipped {len(skipped)} rows:", file=sys.stderr)
            for idx, err in skipped:
                print(f"   • row {idx}: {err}", file=sys.stderr)
        return out

    # Batch over valid rows only
    for start, end in tqdm(
        list(chunk_indices(len(valid_texts), batch_size)),
        desc=f"Scoring Binoculars on '{text_col}'",
    ):
        batch_texts = valid_texts[start:end]
        idx_slice = valid_indices[start:end]

        try:
            # compute_score returns float or list[float] depending on input type
            scores = bino.compute_score(batch_texts)
            if isinstance(scores, float):
                scores = [scores]
        except Exception as e:
            # If the whole batch fails, fall back to singletons to salvage what we can
            for local_j, t in enumerate(batch_texts):
                abs_idx = idx_slice[local_j]
                try:
                    s = bino.compute_score(t)
                    # Assign row
                    out.at[abs_idx, "bino_score"] = float(s)
                    # 1 = AI, 0 = Human
                    thr = getattr(bino, "threshold", None)
                    if thr is None:
                        skipped.append((abs_idx, "no threshold available on model"))
                    else:
                        pred = 1 if s < thr else 0
                        out.at[abs_idx, "bino_pred"] = int(pred)
                except Exception as e_single:
                    skipped.append((abs_idx, f"error: {e_single}"))
            continue

        # Assign rows for the successful batch
        for local_j, s in enumerate(scores):
            abs_idx = idx_slice[local_j]
            out.at[abs_idx, "bino_score"] = float(s)
            thr = getattr(bino, "threshold", None)
            if thr is None:
                skipped.append((abs_idx, "no threshold available on model"))
            else:
                pred = 1 if s < thr else 0
                out.at[abs_idx, "bino_pred"] = int(pred)

    if skipped:
        print(f"\n⚠ Skipped {len(skipped)} rows:", file=sys.stderr)
        for idx, err in skipped:
            print(f"   • row {idx}: {err}", file=sys.stderr)

    return out


def main():
    p = argparse.ArgumentParser(description="Inference: AI-text detection with Binoculars over CSV")
    p.add_argument("input_csv", help="CSV with a text column (default: first column)")
    p.add_argument("--output-csv", default="text_predictions.csv", help="Where to write results")
    p.add_argument("--text-col", default=None, help="Name of the text column (defaults to first column)")
    p.add_argument("--batch-size", type=int, default=8, help="Number of texts per batch")

    # Falcon/Binoculars knobs
    p.add_argument("--observer", default="tiiuae/falcon-7b", help="Observer model name or path")
    p.add_argument("--performer", default="tiiuae/falcon-7b-instruct", help="Performer model name or path")
    p.add_argument("--mode", choices=["low-fpr", "accuracy"], default="low-fpr", help="Binoculars threshold mode")
    p.add_argument("--max-len", type=int, default=384, help="Max tokens observed per text")

    args = p.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv)

    # Decide which column is text
    text_col = args.text_col
    if text_col is None:
        if df.shape[1] == 0:
            raise ValueError("Input CSV has no columns.")
        text_col = df.columns[0]

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV. Available: {list(df.columns)}")

    # Instantiate FalconT4Binoculars
    bino = FalconT4Binoculars(
        observer_name_or_path=args.observer,
        performer_name_or_path=args.performer,
        max_token_observed=args.max_len,
        mode=args.mode,
    )
    # class doesn't keep .mode, so we add it for convenience in the output
    setattr(bino, "mode", args.mode)

    result = predict_texts(df, bino, text_col=text_col, batch_size=args.batch_size)
    result.to_csv(args.output_csv, index=False)
    print(f"✅ Wrote {len(result)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
