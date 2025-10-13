from __future__ import annotations
from typing import Iterable, List, Dict, Union, Optional, Tuple

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Your README shows these classes and modes explicitly
# T4 default
from experimental.falcon_t4_binoculars import FalconT4Binoculars  # type: ignore
# Optional A100 path (safe import)
try:
    from experimental.falcon_a100_binoculars import FalconA100Binoculars  # type: ignore
except Exception:  # pragma: no cover
    FalconA100Binoculars = None  # type: ignore

# Thresholds from your README (score = human-likeness; flag AI if score < threshold)
ACCURACY_THRESHOLD = 0.9015310749276843
FPR_THRESHOLD      = 0.8536432310785527

# Small cache so we keep 1 model instance per (device, mode, max_len, extra kwargs)
_MODELS: dict[Tuple[str, str, int, Tuple[Tuple[str, object], ...]], object] = {}

def _get_model(
    device: str = "t4",
    mode: str = "low-fpr",
    max_len: int = 384,
    **backend_kwargs
):
    key = (device, mode, max_len, tuple(sorted(backend_kwargs.items())))
    if key in _MODELS:
        return _MODELS[key]

    if device.lower() in ("t4", "t4g", "gpu"):
        model = FalconT4Binoculars(mode=mode, max_token_observed=max_len, **backend_kwargs)
    elif device.lower() in ("a100",):
        if FalconA100Binoculars is None:
            raise RuntimeError("FalconA100Binoculars unavailable; ensure the class exists in experimental/")
        model = FalconA100Binoculars(mode=mode, max_token_observed=max_len, **backend_kwargs)
    else:
        raise ValueError(f"Unknown device '{device}'. Use 't4' or 'a100'.")

    _MODELS[key] = model
    return model

def _default_threshold(mode: str) -> float:
    # Mode mapping exactly as your README suggests
    if mode == "low-fpr":
        return FPR_THRESHOLD
    elif mode == "accuracy":
        return ACCURACY_THRESHOLD
    else:
        raise ValueError("mode must be one of {'low-fpr','accuracy'}")

def detect_batch(
    texts: Iterable[Optional[str]],
    *,
    device: str = "t4",
    mode: str = "low-fpr",
    max_len: int = 384,
    threshold: Optional[float] = None,
    progress: bool = False,
    return_text: bool = True,
    **backend_kwargs
) -> List[Dict[str, Union[float, int, str]]]:
    """
    Score a batch with Falcon* Binoculars (T4-optimized by default), returning score + label.
    - score = human-likeness (higher => more human)
    - label = 1 means AI-flagged (score < threshold), 0 means not flagged

    Args match your README: device('t4'|'a100'), mode('low-fpr'|'accuracy'), max_len.
    threshold: overrides default per-mode threshold if given.
    """
    items = list(texts)
    nonempty_mask = [bool(t) for t in items]
    nonempty = [t for t in items if t]

    bino = _get_model(device=device, mode=mode, max_len=max_len, **backend_kwargs)

    iterator = range(len(items))
    if progress and _HAS_TQDM:
        iterator = tqdm(iterator, desc=f"Scoring texts ({device}/{mode}, max_len={max_len})")

    scores_nonempty = bino.compute_score(nonempty) if nonempty else []
    scores: List[float] = []
    j = 0
    for i in iterator:
        if nonempty_mask[i]:
            scores.append(float(scores_nonempty[j]))
            j += 1
        else:
            scores.append(0.0)  # empty -> score 0.0 (will likely flag as AI under either threshold)

    thr = float(threshold if threshold is not None else _default_threshold(mode))
    out: List[Dict[str, Union[float, int, str]]] = []
    for i, s in enumerate(scores):
        rec: Dict[str, Union[float, int, str]] = {
            "score": s,
            "label": int(s < thr)  # IMPORTANT: flag AI if score is below threshold
        }
        if return_text:
            rec["text"] = items[i] or ""
        out.append(rec)
    return out
