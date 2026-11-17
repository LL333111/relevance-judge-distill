"""Qwen3-Reranker-0.6B scoring.

Follows the model card's Transformers recipe: a fixed system prefix, the
"<Instruct>/<Query>/<Document>" body, the assistant suffix, left padding, and
the "yes"/"no" logits at the final position. The score is

    s(q, d) = P(yes) = softmax([no, yes])[yes] = sigmoid(logit_yes - logit_no)

The only departure from the card's code is computational: instead of
projecting every position through the vocabulary head, the final hidden state
is multiplied by the "yes" and "no" rows of the head alone. The result is the
same number; scripts/check_recipe.py verifies this against the card's code.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
          "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
          "<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_student_config(path: str | Path = "configs/student.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_pair(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


class Student:
    def __init__(self, cfg: dict | None = None, adapter: str | Path | None = None,
                 train_lora: bool = False, device: str = "cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg or load_student_config()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model"], padding_side="left")
        causal = AutoModelForCausalLM.from_pretrained(self.cfg["model"], dtype=dtype).to(device)
        if causal.lm_head.bias is not None:
            raise RuntimeError("expected a bias-free output head")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.body_max = self.cfg["max_length"] - len(self.prefix_ids) - len(self.suffix_ids)

        # LoRA modules are injected in place, so `causal` keeps pointing at the
        # (adapted) network either way.
        self.causal = causal
        self.peft_model = None
        if adapter is not None and train_lora:
            raise ValueError("pass either adapter or train_lora")
        if adapter is not None:
            from peft import PeftModel
            self.peft_model = PeftModel.from_pretrained(causal, str(adapter))
        elif train_lora:
            from peft import LoraConfig, get_peft_model
            self.peft_model = get_peft_model(causal, LoraConfig(**self.cfg["lora"]))
            # A step holds every candidate of its queries (one query has 188),
            # so activations are recomputed in the backward pass instead of
            # stored. The embeddings are frozen, hence the input-grad hook.
            causal.gradient_checkpointing_enable()
            causal.enable_input_require_grads()
        self.causal.eval()

    # ------------------------------------------------------------- tokenizing
    def tokenize(self, queries, docs) -> list[list[int]]:
        texts = [format_pair(self.cfg["instruction"], q, d) for q, d in zip(queries, docs)]
        body = self.tokenizer(texts, padding=False, truncation="longest_first",
                              max_length=self.body_max, add_special_tokens=False,
                              return_attention_mask=False)["input_ids"]
        return [self.prefix_ids + ids + self.suffix_ids for ids in body]

    def collate(self, token_lists: list[list[int]]) -> dict:
        """Left-pad to the longest sequence in the batch."""
        width = max(len(ids) for ids in token_lists)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((len(token_lists), width), pad, dtype=torch.long)
        mask = torch.zeros((len(token_lists), width), dtype=torch.long)
        for i, ids in enumerate(token_lists):
            input_ids[i, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            mask[i, width - len(ids):] = 1
        return {"input_ids": input_ids.to(self.device), "attention_mask": mask.to(self.device)}

    # ---------------------------------------------------------------- scoring
    def logit_diff(self, batch: dict) -> torch.Tensor:
        """logit(yes) - logit(no) at the final position, float32, differentiable."""
        hidden = self.causal.model(**batch).last_hidden_state[:, -1, :]
        head = self.causal.lm_head.weight
        direction = (head[self.yes_id] - head[self.no_id]).to(hidden.dtype)
        return (hidden @ direction).float()

    def prob_yes(self, batch: dict) -> torch.Tensor:
        return torch.sigmoid(self.logit_diff(batch))

    @torch.no_grad()
    def score(self, queries, docs, batch_size: int | None = None) -> tuple[np.ndarray, dict]:
        """P(yes) for every pair, plus timing. Batches are formed by length."""
        batch_size = batch_size or self.cfg["eval_batch_size"]
        self.causal.eval()
        start = time.perf_counter()
        tokens = self.tokenize(list(queries), list(docs))
        order = np.argsort([len(t) for t in tokens], kind="stable")[::-1]
        scores = np.empty(len(tokens), dtype=np.float64)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            batch = self.collate([tokens[j] for j in idx])
            scores[idx] = self.prob_yes(batch).cpu().numpy()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        timing = {"n_pairs": len(tokens), "seconds": seconds, "pairs_per_second": len(tokens) / seconds,
                  "n_truncated": int(sum(len(t) >= self.cfg["max_length"] for t in tokens)),
                  "mean_tokens": float(np.mean([len(t) for t in tokens]))}
        return scores, timing
