# Falcon-on-T4 Binoculars (4-bit + sequential)
# - Loads Falcon-7B (observer) and Falcon-7B-Instruct (performer) in 4-bit NF4
# - Runs forward passes sequentially to keep peak VRAM low
# - Offloads performer logits to CPU between passes
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

__all__ = ["FalconT4Binoculars"]

# Tuned operating points you provided
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843
BINOCULARS_FPR_THRESHOLD      = 0.8536432310785527

# Device selection (supports 1 or 2 GPUs)
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class FalconT4Binoculars:
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 use_bfloat16: bool = False,          # T4 has no bf16
                 max_token_observed: int = 384,
                 mode: str = "low-fpr",
                 ):
        # allocator tweak to reduce fragmentation on T4
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        torch.set_grad_enabled(False)

        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)
        self.change_mode(mode)
        self.max_token_observed = max_token_observed

        # Tokenizer (shared)
        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not getattr(self.tokenizer, "pad_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit NF4 quantization for both models
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        common = dict(
            trust_remote_code=False,
            attn_implementation="eager",   # safest on T4 (no flash-attn v2)
            low_cpu_mem_usage=True,
            quantization_config=qcfg,
        )

        # Keep both models resident, but run them sequentially to keep peak activations low
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path, device_map={"": DEVICE_2}, **common
        ).eval()
        self.observer_model  = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path, device_map={"": DEVICE_1}, **common
        ).eval()

        # Save VRAM (we're not generating text)
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
        Sequential low-VRAM pass:
          1) performer forward -> move full logits to CPU
          2) observer forward -> keep on DEVICE_1
        NOTE: Return FULL logits; Binoculars metrics handle the shift internally.
        """
        # Performer pass (often on DEVICE_2)
        perf_logits = self.performer_model(**enc.to(DEVICE_2)).logits  # NO slicing
        perf_logits_cpu = perf_logits.to("cpu")
        del perf_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Observer pass (on DEVICE_1)
        obs_logits = self.observer_model(**enc.to(DEVICE_1)).logits    # NO slicing
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return obs_logits, perf_logits_cpu

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        enc = self._tokenize(batch)

        obs_logits, perf_logits = self._get_logits(enc)

        # Align everything on DEVICE_1 for the metric computations
        enc_dev  = enc.to(DEVICE_1)
        obs_dev  = obs_logits.to(DEVICE_1).float()
        perf_dev = perf_logits.to(DEVICE_1).float()

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
