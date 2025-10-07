# Falcon-on-A100 Binoculars (4-bit NF4 + bf16/fast attention, no CPU offload)
# - Loads Falcon-7B (observer) and Falcon-7B-Instruct (performer) in 4-bit NF4
# - Keeps logits on GPU (A100 has ample VRAM)
# - Prefers Flash-Attention 2 (falls back to SDPA, then eager)
# - Uses Binoculars metrics with UNshifted logits (metrics handle shift internally)

from __future__ import annotations

import os
from typing import Union

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from binoculars.metrics import perplexity, entropy
from binoculars.utils import assert_tokenizer_consistency

__all__ = ["FalconA100Binoculars"]

# Tuned operating points you provided
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843
BINOCULARS_FPR_THRESHOLD      = 0.8536432310785527

# Device selection (supports 1 or 2 GPUs)
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class FalconA100Binoculars:
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 max_token_observed: int = 768,
                 mode: str = "low-fpr",
                 ):
        # allocator tweak to keep fragmentation low
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        torch.set_grad_enabled(False)

        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)
        self.change_mode(mode)
        self.max_token_observed = max_token_observed

        # Tokenizer (shared)
        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not getattr(self.tokenizer, "pad_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prefer bf16 compute on A100; fallback to fp16 if unavailable
        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        # 4-bit NF4 quantization for both models
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        # Start with SDPA; upgrade to flash_attn2 post-load if available
        common = dict(
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            quantization_config=qcfg,
        )

        # Keep both models resident; A100 can handle full GPU logits (no CPU offload)
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path, device_map={"": DEVICE_2}, **common
        ).eval()
        self.observer_model  = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path, device_map={"": DEVICE_1}, **common
        ).eval()

        # Prefer Flash-Attention 2 if installed; else keep SDPA/eager
        for m in (self.performer_model, self.observer_model):
            try:
                m.config.attn_implementation = "flash_attention_2"
            except Exception:
                try:
                    m.config.attn_implementation = "sdpa"
                except Exception:
                    m.config.attn_implementation = "eager"

        # Not generating text
        if hasattr(self.performer_model, "config"):
            self.performer_model.config.use_cache = False
        if hasattr(self.observer_model, "config"):
            self.observer_model.config.use_cache = False

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        enc = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if len(batch) > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.observer_model.device)
        return enc

    @torch.inference_mode()
    def _get_logits(self, enc: transformers.BatchEncoding):
        """
        High-VRAM pass (A100):
          1) performer forward -> keep logits on GPU
          2) observer forward  -> keep logits on GPU
        """
        perf_logits = self.performer_model(**enc.to(DEVICE_2)).logits   # NO slicing
        obs_logits  = self.observer_model(**enc.to(DEVICE_1)).logits    # NO slicing
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return obs_logits, perf_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        enc = self._tokenize(batch)

        obs_logits, perf_logits = self._get_logits(enc)

        # Align everything on DEVICE_1 for the metric computations
        enc_dev  = enc.to(DEVICE_1)
        obs_dev  = obs_logits.to(DEVICE_1)
        perf_dev = perf_logits.to(DEVICE_1)

        # Metrics expect unshifted logits; they do the shift internally
        ppl   = perplexity(enc_dev,  perf_dev)
        x_ppl = entropy   (obs_dev,  perf_dev, enc_dev, self.tokenizer.pad_token_id)

        scores = (ppl / x_ppl).tolist()
        return scores[0] if isinstance(input_text, str) else scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        scores = np.array(self.compute_score(input_text))
        pred = np.where(
            scores < self.threshold,
            "Most likely AI-generated",
            "Most likely human-generated",
        ).tolist()
        return pred
