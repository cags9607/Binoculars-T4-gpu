# ai_gen_text/detector.py
"""
Main detection module.

Exposes:
    detect_batch(texts, **kwargs) -> List[dict]

- Uses your Falcon* Binoculars classes (T4 by default).
- score = human-likeness (higher => more human)
- label rule: score < threshold -> AI (1), else Human (0)

Example:
    from ai_gen_text import detect_batch
    out = detect_batch(
        ["Hola!", "The quick brown fox..."],
        device="t4",
        mode="low-fpr",
        max_len=384,
        batch_size=32,
        progress=False,
    )
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Optional progress bar (not a hard dependency)
try:
    from tqdm.auto import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False

# ----- Import your T4/A100 implementations -----
from experimental.falcon_t4_binoculars import (  # type: ignore
    FalconT4Binoculars,
)
# Threshold constants (best-effort import if you exposed them)
try:  # pragma: no cover
    from experimental.falcon_t4_binoculars import (  # type: ignore
        BINOCULARS_FPR_THRESHOLD as _FPR_T4,
        BINOCULARS_ACCURACY_THRESHOLD as _ACC_T4,
    )
except Exception:  # pragma: no cover
    _FPR_T4 = 0.8536432310785527
    _ACC_T4 = 0.9015310749276843

# Optional A100 path
try:  # pragma: no cover
    from experimental.falcon_a100_binoculars import FalconA100Binoculars  # type: ignore
except Exception:  # pragma: no cover
    FalconA100Binoculars = None  # type: ignore


# ---------------- Internals ----------------

# Cache a single model instance per configuration to avoid reloads
_MODELS: dict[
    Tuple[str, str, int, Tuple[Tuple[str, object], ...]],  # (device, mode, max_len, sorted(kwargs))
    object,
] = {}


def _default_threshold(device: str, mode: str) -> float:
    """Map mode → default threshold (per device family if needed)."""
    mode = mode.lower()
    if device.lower() in ("t4", "gpu"):
        if mode == "low-fpr":
            return float(_FPR_T4)
        if mode == "accuracy":
            return float(_ACC_T4)
        raise ValueError("mode must be one of {'low-fpr','accuracy'}")
    elif device.lower() == "a100":
        # If you keep different A100 thresholds, import them above and use here.
        if mode == "low-fpr":
            return float(_FPR_T4)
        if mode == "accuracy":
            return float(_ACC_T4)
        raise ValueError("mode must be one of {'low-fpr','accuracy'}")
    else:
        raise ValueError("device must be one of {'t4','a100'}")


def _get_model(
    *,
    device: str = "t4",
    mode: str = "low-fpr",
    max_len: int = 384,
    **backend_kwargs: Any,
):
    """
    Lazily build (and cache) a Falcon* Binoculars model.
    For T4 this uses your 4-bit NF4 sequential implementation.
    """
    key = (device.lower(), mode, int(max_len), tuple(sorted(backend_kwargs.items())))
    if key in _MODELS:
        return _MODELS[key]

    if device.lower() in ("t4", "gpu"):
        model = FalconT4Binoculars(
            mode=mode,
            max_token_observed=max_len,
            **backend_kwargs,
        )
    elif device.lower() == "a100":
        if FalconA100Binoculars is None:
            raise RuntimeError("A100 path requested but FalconA100Binoculars is not available.")
        model = FalconA100Binoculars(
            mode=mode,
            max_token_observed=max_len,
            **backend_kwargs,
        )
    else:
        raise ValueError(f"Unknown device '{device}'. Use 't4' or 'a100'.")

    # Stamp convenience attributes if your class doesn't keep them
    if not hasattr(model, "mode"):
        setattr(model, "mode", mode)
    if not hasattr(model, "threshold"):
        try:
            setattr(model, "threshold", _default_threshold(device, mode))
        except Exception:
            pass

    _MODELS[key] = model
    return model


def _coerce_text(x: Any) -> Optional[str]:
    """Return clean string or None if invalid/empty."""
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _postprocess(
    texts: Sequence[Optional[str]],
    scores: Sequence[Optional[float]],   # allow None for failed/empty rows
    mode: str,
    threshold: Optional[float],
    return_text: bool,
) -> List[Dict[str, Union[str, float, int]]]:
    """
    Build the output list of dicts, applying label rule (score<thr -> 1).
    Empty/failed inputs get {'score': None, 'label': None}.
    """
    out: List[Dict[str, Union[str, float, int]]] = []
    thr_used: Optional[float] = float(threshold) if threshold is not None else None

    for i, t in enumerate(texts):
        rec: Dict[str, Union[str, float, int]] = {}

        if t is None or scores[i] is None:
            rec["score"] = None  # type: ignore[assignment]
            rec["label"] = None  # type: ignore[assignment]
        else:
            s = float(scores[i])  # type: ignore[arg-type]
            rec["score"] = s
            if thr_used is None:
                raise RuntimeError(
                    "No threshold available. Pass threshold=... to detect_batch() or expose `.threshold` on the model."
                )
            rec["label"] = int(s < thr_used)

        if return_text:
            rec["text"] = t if t is not None else ""

        rec["mode"] = mode
        if thr_used is not None:
            rec["threshold"] = thr_used

        out.append(rec)

    return out


# ---------------- Public API ----------------

def detect_batch(
    texts: Iterable[Optional[str]],
    **kwargs: Any,
) -> List[Dict[str, Union[str, float, int]]]:
    """
    Detect batch of items.

    Args:
        texts:
            Iterable of text strings to process in batch. None/empty items are handled safely
            and produce {'score': None, 'label': None}.
        **kwargs:
            device:      {'t4','a100'} (default: 't4')
            mode:        {'low-fpr','accuracy'} (default: 'low-fpr')
            max_len:     int max tokens observed per text (default: 384)
            threshold:   float to override model threshold (score<threshold -> AI=1)
            progress:    bool; display tqdm progress (default: False)
            batch_size:  int; number of texts per scoring chunk (default: 32)
            return_text: bool; include original text in each record (default: True)
            ...backend kwargs forwarded to the underlying Falcon* class
                e.g., observer_name_or_path=..., performer_name_or_path=...

    Returns:
        List of dicts with:
            - text:      original text (if return_text=True)
            - score:     float (human-likeness; higher = more human) or None for empty/failed input
            - label:     1 if AI (score<threshold), 0 if Human; None for empty/failed input
            - mode:      the thresholding profile used
            - threshold: the threshold value applied
    """
    device: str = kwargs.pop("device", "t4")
    mode: str = kwargs.pop("mode", "low-fpr")
    max_len: int = int(kwargs.pop("max_len", 384))
    threshold: Optional[float] = kwargs.pop("threshold", None)
    progress: bool = bool(kwargs.pop("progress", False))
    batch_size: int = int(kwargs.pop("batch_size", 32))
    return_text: bool = bool(kwargs.pop("return_text", True))

    # Treat accidental string input as a single-item list
    if isinstance(texts, (str, bytes)):
        texts = [texts]  # type: ignore[list-item]

    # Normalize inputs and build a mask of valid texts
    items: List[Optional[str]] = [_coerce_text(t) for t in texts]
    nonempty_idx: List[int] = [i for i, t in enumerate(items) if t is not None]
    nonempty_texts: List[str] = [items[i] for i in nonempty_idx]  # type: ignore[index]

    # Early exit: nothing to score
    if not nonempty_texts:
        # Resolve a default threshold even when no scoring happens
        if threshold is None:
            threshold = _default_threshold(device, mode)
        return _postprocess(items, [None] * len(items), mode=mode, threshold=threshold, return_text=return_text)

    # Remove consumer-only kwargs so they won't be forwarded to the model __init__
    for _k in ("batch_size", "progress", "return_text"):
        kwargs.pop(_k, None)

    # Build or reuse model (only model-relevant kwargs are forwarded)
    model = _get_model(device=device, mode=mode, max_len=max_len, **kwargs)

    # Resolve threshold: explicit arg wins; else model.threshold; else default mapping
    if threshold is None:
        threshold = getattr(model, "threshold", None)  # type: ignore[attr-defined]
        if threshold is None:
            threshold = _default_threshold(device, mode)
    thr = float(threshold)

    # --------- Chunked scoring to avoid CUDA OOM ---------
    total = len(nonempty_texts)
    all_scores: List[Optional[float]] = [None] * len(items)

    num_batches = (total + batch_size - 1) // batch_size
    batch_iter = range(0, total, batch_size)
    if progress and _HAS_TQDM:
        batch_iter = tqdm(
            batch_iter,
            total=num_batches,
            desc=f"Scoring ({device}/{mode}, max_len={max_len}, bs={batch_size})",
        )

    # Try to import torch for empty_cache on OOM; optional
    try:
        import torch  # type: ignore
        _HAS_TORCH = True
    except Exception:
        _HAS_TORCH = False

    for start in batch_iter:
        end = min(start + batch_size, total)
        idxs = nonempty_idx[start:end]
        txts = nonempty_texts[start:end]

        try:
            bscores = model.compute_score(txts)  # type: ignore[attr-defined]
            if isinstance(bscores, float):
                bscores = [bscores]
            for j, s in enumerate(bscores):
                all_scores[idxs[j]] = float(s)

        except Exception:
            # Batch failed (often CUDA OOM). Best-effort salvage per item.
            if _HAS_TORCH:
                try:
                    torch.cuda.empty_cache()  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Try singleton scoring to salvage what we can
            for j, t in enumerate(txts):
                abs_idx = idxs[j]
                try:
                    s = model.compute_score(t)  # type: ignore[attr-defined]
                    all_scores[abs_idx] = float(s)
                except Exception:
                    # leave as None (score/pred will be None)
                    if _HAS_TORCH:
                        try:
                            torch.cuda.empty_cache()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    continue

    # Build output records
    return _postprocess(items, all_scores, mode=mode, threshold=thr, return_text=return_text)
