"""
Core functionality for AI text detection using Binoculars (Falcon T4, 4-bit + sequential)

Notes
- Uses FalconT4Binoculars implementation (observer/performer ratio: ppl / x_ppl).
- Returns a boolean AI flag derived from score < threshold; score is NOT a calibrated probability.
- Public-facing fields mirror the image module style where possible; probability fields are None.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd

# Your custom Binoculars class
# Adjust the import path to where you placed the file:
# e.g., from binoculars.experimental.falcon_t4_binoculars import FalconT4Binoculars
from binoculars.experimental.falcon_t4_binoculars import FalconT4Binoculars

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextClassifier:
    """Core text detector using FalconT4Binoculars."""

    def __init__(
        self,
        mode: str = "low-fpr",
        max_token_observed: int = 384,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
    ) -> None:
        self.mode = mode
        self.max_token_observed = max_token_observed
        self.observer_name_or_path = observer_name_or_path
        self.performer_name_or_path = performer_name_or_path

        self.bino: Optional[FalconT4Binoculars] = None

    # ------------------------------ loading ------------------------------
    def load_models(self) -> None:
        """Instantiate FalconT4Binoculars (loads both models in 4-bit)."""
        logger.info("Loading Binoculars (Falcon T4)…")
        self.bino = FalconT4Binoculars(
            observer_name_or_path=self.observer_name_or_path,
            performer_name_or_path=self.performer_name_or_path,
            max_token_observed=self.max_token_observed,
            mode=self.mode,
        )
        logger.info("Binoculars loaded and ready.")

    def are_models_loaded(self) -> bool:
        """Check if required models are loaded."""
        return self.bino is not None

    # --------------------------- prediction (single) ---------------------------
    def _predict_one(self, text: str) -> Dict[str, Any]:
        """
        Predict for a single text string.

        Returns:
            dict with keys:
            - bino_score: float            (ratio ppl/x_ppl; lower suggests AI)
            - threshold: float             (operating threshold used)
            - mode: str                    ("low-fpr" or "accuracy")
            - prediction_textgen: int      (1 = AI-generated, 0 = human-generated)
            - label: str                   ("Most likely AI-generated" | "Most likely human-generated")
            - prob_textgen: None           (placeholder; no calibrated probability here)
            - is_fake_probability: None    (placeholder to mirror image module shape)
        """
        if self.bino is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Normalize/guard input
        if text is None:
            score = np.nan
            ai_flag = None
            label = "Invalid input"
        else:
            s = str(text).strip()
            if not s:
                score = np.nan
                ai_flag = None
                label = "Empty text"
            else:
                score = float(self.bino.compute_score(s))
                ai_flag = bool(score < self.bino.threshold)
                label = "Most likely AI-generated" if ai_flag else "Most likely human-generated"

        return {
            "bino_score": score,
            "threshold": float(self.bino.threshold) if self.bino else np.nan,
            "mode": self.mode,
            "prediction_textgen": int(ai_flag) if isinstance(ai_flag, bool) else -1,
            "label": label,
            # Placeholders to keep API shape similar to image module:
            "prob_textgen": None,            # not calibrated
            "is_fake_probability": None,     # not calibrated
        }

    def classify_text_from_string(self, text: str) -> Dict[str, Any]:
        """Public single-text entrypoint."""
        return self._predict_one(text)

    # --------------------------- prediction (batch) ---------------------------
    def classify_texts_batch(self, texts: List[Union[str, None]]) -> List[Dict[str, Any]]:
        """
        Predict for a list of texts. Preserves input order.

        Args:
            texts: list of strings (None/empty handled gracefully)

        Returns:
            List[dict] (one per input), each with the same keys as _predict_one.
        """
        if self.bino is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Use the model's batch scoring for speed, while preserving per-item error handling.
        # Split into valid and invalid to avoid exceptions mid-batch.
        idx_map: List[int] = list(range(len(texts)))
        valid_pairs = [(i, str(t).strip()) for i, t in enumerate(texts) if (t is not None and str(t).strip())]
        invalid_indices = [i for i, t in enumerate(texts) if (t is None or not str(t).strip())]

        results: List[Optional[Dict[str, Any]]] = [None] * len(texts)

        # Fill invalids first
        for i in invalid_indices:
            results[i] = {
                "bino_score": np.nan,
                "threshold": float(self.bino.threshold),
                "mode": self.mode,
                "prediction_textgen": -1,
                "label": "Invalid input" if texts[i] is None else "Empty text",
                "prob_textgen": None,
                "is_fake_probability": None,
            }

        # Batch compute scores for valids
        if valid_pairs:
            try:
                valid_indices, valid_texts = zip(*valid_pairs)
                scores = self.bino.compute_score(list(valid_texts))  # returns list of floats
                # Map to outputs
                for i, sc in zip(valid_indices, scores):
                    sc = float(sc)
                    ai_flag = bool(sc < self.bino.threshold)
                    label = "Most likely AI-generated" if ai_flag else "Most likely human-generated"
                    results[i] = {
                        "bino_score": sc,
                        "threshold": float(self.bino.threshold),
                        "mode": self.mode,
                        "prediction_textgen": int(ai_flag),
                        "label": label,
                        "prob_textgen": None,
                        "is_fake_probability": None,
                    }
            except Exception as e:
                # If batch fails, fall back to per-item robust path
                logger.warning(f"Batch scoring failed ({e}); falling back to per-item scoring.")
                for i, txt in valid_pairs:
                    try:
                        results[i] = self._predict_one(txt)
                    except Exception as ie:
                        results[i] = {
                            "bino_score": np.nan,
                            "threshold": float(self.bino.threshold),
                            "mode": self.mode,
                            "prediction_textgen": -1,
                            "label": f"Error: {ie}",
                            "prob_textgen": None,
                            "is_fake_probability": None,
                        }

        # Safety: no Nones remain
        for j, r in enumerate(results):
            if r is None:
                results[j] = {
                    "bino_score": np.nan,
                    "threshold": float(self.bino.threshold),
                    "mode": self.mode,
                    "prediction_textgen": -1,
                    "label": "Unknown error",
                    "prob_textgen": None,
                    "is_fake_probability": None,
                }

        return results

    # --------------------------- dataframe helper ---------------------------
    def classify_dataframe(
        self, df: pd.DataFrame, text_col: str = "text", batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Vectorized helper for DataFrames. Adds 'bino_score', 'prediction_textgen', 'label'.
        Uses small micro-batches for memory safety.
        """
        if self.bino is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        if text_col not in df.columns:
            raise ValueError(f"DataFrame does not have a column named '{text_col}'")

        ser = df[text_col]
        out_scores: List[float] = []
        out_preds: List[int] = []
        out_labels: List[str] = []

        # micro-batch iteration
        n = len(ser)
        for a in range(0, n, batch_size):
            chunk = ser.iloc[a : a + batch_size].tolist()
            # compute scores with validation similar to classify_texts_batch
            res = self.classify_texts_batch(chunk)
            out_scores.extend([r["bino_score"] for r in res])
            out_preds.extend([r["prediction_textgen"] for r in res])
            out_labels.extend([r["label"] for r in res])

        df_out = df.copy(deep=True)
        df_out.loc[:, "bino_score"] = out_scores
        df_out.loc[:, "prediction_textgen"] = out_preds
        df_out.loc[:, "label"] = out_labels
        return df_out

    # --------------------------- info ---------------------------
    def get_model_info(self) -> dict:
        if self.bino is None:
            return {"status": "not_loaded", "models_loaded": False}
        return {
            "status": "loaded",
            "models_loaded": True,
            "binoculars": {
                "observer": self.observer_name_or_path,
                "performer": self.performer_name_or_path,
                "mode": self.mode,
                "threshold": float(self.bino.threshold),
                "max_token_observed": int(self.max_token_observed),
            },
            "notes": "Score is ppl/x_ppl ratio; lower implies more likely AI. Not a calibrated probability.",
        }


# Global instance for backward compatibility (mirrors the image module pattern)
_default_text_classifier: Optional[TextClassifier] = None

def get_default_text_classifier() -> TextClassifier:
    """Get or create the default global text classifier instance."""
    global _default_text_classifier
    if _default_text_classifier is None:
        _default_text_classifier = TextClassifier()
    return _default_text_classifier
