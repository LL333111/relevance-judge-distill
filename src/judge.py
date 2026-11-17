"""LLM relevance judge: prompt, label-token mapping and logit readout.

The judge sees the query and the product and answers with one letter. Nothing
is generated: the logits of the four label tokens are read at the final
position, softmaxed into class probabilities p(E), p(S), p(C), p(I), and
turned into the expected gain

    t(q, d) = sum over labels of p(label) * gain(label).

As in src/student.py, only the four label rows of the output head are
applied to the final hidden state.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.metrics import GAINS, LABELS

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = (
    "You are a search relevance judge for an online store. Given a shopping query and a product, "
    "decide how the product relates to the query.\n"
    "E (Exact): the product is relevant and satisfies all of the query's specifications.\n"
    "S (Substitute): the product is somewhat relevant. It fails some aspects of the query but "
    "could work as a functional substitute.\n"
    "C (Complement): the product does not fulfil the query but could be used together with an "
    "exact product.\n"
    "I (Irrelevant): the product is irrelevant, or it fails a central aspect of the query.\n"
    "Answer with a single letter: E, S, C or I."
)
_SENTINEL = "␟PRODUCT␟"


def load_judge_config(path: str | Path = "configs/judge.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def label_token_ids(tokenizer) -> list[int]:
    """Token id of each label letter, in LABELS order. Each must be one token."""
    ids = []
    for label in LABELS:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"label {label!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    if len(set(ids)) != len(LABELS):
        raise RuntimeError("label tokens are not distinct")
    return ids


def expected_gain(probs: np.ndarray) -> np.ndarray:
    """probs has one column per label in LABELS order."""
    return probs @ np.array([GAINS[label] for label in LABELS])


class Judge:
    def __init__(self, cfg: dict | None = None, adapter: str | Path | None = None,
                 train_lora: bool = False, device: str = "cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg or load_judge_config()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model"])
        causal = AutoModelForCausalLM.from_pretrained(self.cfg["model"], dtype=dtype).to(device)
        if causal.lm_head.bias is not None:
            raise RuntimeError("expected a bias-free output head")
        self.label_ids = label_token_ids(self.tokenizer)
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            raise RuntimeError("tokenizer has no pad token")

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
            causal.gradient_checkpointing_enable()
            causal.enable_input_require_grads()
        self.causal.eval()

    # ------------------------------------------------------------- tokenizing
    def render(self, query: str) -> tuple[str, str]:
        """Chat-template text before and after the product text (thinking off)."""
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Query: {query}\nProduct:\n{_SENTINEL}"}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        head, tail = text.split(_SENTINEL)
        return head, tail

    def tokenize(self, queries, docs) -> list[list[int]]:
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
        out = []
        for query, doc in zip(queries, docs):
            head, tail = self.render(query)
            out.append(encode(head) + encode(doc)[: self.cfg["max_product_tokens"]] + encode(tail))
        return out

    def collate(self, token_lists: list[list[int]]) -> dict:
        width = max(len(ids) for ids in token_lists)
        input_ids = torch.full((len(token_lists), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(token_lists), width), dtype=torch.long)
        for i, ids in enumerate(token_lists):
            input_ids[i, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            mask[i, width - len(ids):] = 1
        return {"input_ids": input_ids.to(self.device), "attention_mask": mask.to(self.device)}

    # ---------------------------------------------------------------- readout
    def label_logits(self, batch: dict) -> torch.Tensor:
        """Logits of the four label tokens at the final position: (batch, 4), float32."""
        hidden = self.causal.model(**batch).last_hidden_state[:, -1, :]
        rows = self.causal.lm_head.weight[self.label_ids].to(hidden.dtype)
        return (hidden @ rows.T).float()

    @torch.no_grad()
    def predict(self, queries, docs, batch_size: int | None = None, log_every: int = 0):
        """Class probabilities (n, 4) in LABELS order, plus timing."""
        batch_size = batch_size or self.cfg["eval_batch_size"]
        self.causal.eval()
        start = time.perf_counter()
        tokens = self.tokenize(list(queries), list(docs))
        order = np.argsort([len(t) for t in tokens], kind="stable")[::-1]
        probs = np.empty((len(tokens), len(LABELS)), dtype=np.float64)
        for n, i in enumerate(range(0, len(order), batch_size)):
            idx = order[i:i + batch_size]
            logits = self.label_logits(self.collate([tokens[j] for j in idx]))
            probs[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
            if log_every and n % log_every == 0:
                print(f"  scored {i + len(idx)}/{len(tokens)} pairs "
                      f"({time.perf_counter() - start:.0f}s)", flush=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        timing = {"n_pairs": len(tokens), "seconds": seconds, "pairs_per_second": len(tokens) / seconds,
                  "mean_tokens": float(np.mean([len(t) for t in tokens]))}
        return probs, timing


def judge_metrics(frame, probs: np.ndarray) -> dict:
    """Accuracy, kappa and nDCG@10 of judge probabilities against human labels."""
    from src.metrics import accuracy, cohen_kappa, confusion_matrix, labels_to_gains, ndcg_at_k

    predicted = [LABELS[i] for i in probs.argmax(axis=1)]
    truth = list(frame["esci_label"])
    ranking = ndcg_at_k(frame["query_id"], expected_gain(probs), labels_to_gains(truth), frame["example_id"])
    return {"ndcg@10": ranking.mean, "accuracy": accuracy(truth, predicted),
            "kappa": cohen_kappa(truth, predicted),
            "confusion": confusion_matrix(truth, predicted).tolist(),
            "n_queries": len(ranking.per_query)}


def scores_frame(frame, probs: np.ndarray):
    import pandas as pd

    out = pd.DataFrame({"example_id": frame["example_id"].to_numpy()})
    for i, label in enumerate(LABELS):
        out[f"p_{label}"] = probs[:, i]
    out["score"] = expected_gain(probs)
    return out
