"""Verify src/student.py against the model card's reference code.

Runs the card's Transformers snippet (full-vocabulary logits, tokenizer.pad)
and Student.score on the card's two example pairs and on 64 validation pairs,
in float32, and requires the scores to agree.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data import load_pairs  # noqa: E402
from src.student import PREFIX, SUFFIX, Student, format_pair, load_student_config  # noqa: E402


@torch.no_grad()
def card_scores(student: Student, pairs: list[str], max_length: int) -> np.ndarray:
    """The model card's process_inputs + compute_logits, verbatim in substance."""
    tokenizer, model = student.tokenizer, student.causal
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    out = []
    for i in range(0, len(pairs), 8):
        inputs = tokenizer(pairs[i:i + 8], padding=False, truncation="longest_first",
                           return_attention_mask=False,
                           max_length=max_length - len(prefix_tokens) - len(suffix_tokens))
        for j, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][j] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logits = model(**inputs).logits[:, -1, :]
        two = torch.stack([logits[:, student.no_id], logits[:, student.yes_id]], dim=1)
        out.extend(torch.nn.functional.log_softmax(two, dim=1)[:, 1].exp().tolist())
    return np.array(out)


def main() -> None:
    cfg = load_student_config()
    student = Student(cfg, dtype=torch.float32)

    # 1. The card's own example, with the card's default instruction.
    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = ["What is the capital of China?", "Explain gravity"]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical "
        "objects and is responsible for the movement of planets around the sun.",
    ]
    reference = card_scores(student, [format_pair(task, q, d) for q, d in zip(queries, documents)], 8192)
    student.cfg = {**cfg, "instruction": task}
    ours, _ = student.score(queries, documents)
    print("card example  reference:", reference.round(4), " ours:", ours.round(4))
    assert np.abs(reference - ours).max() < 1e-3
    assert (reference > 0.9).all(), "the card's relevant pairs should score near 1"
    student.cfg = cfg

    # 2. Validation pairs with the project instruction and max_length.
    frame = load_pairs("V").sample(64, random_state=0)
    pairs = [format_pair(cfg["instruction"], q, d) for q, d in zip(frame["query"], frame["product_text"])]
    reference = card_scores(student, pairs, cfg["max_length"])
    ours, timing = student.score(frame["query"], frame["product_text"], batch_size=16)
    diff = np.abs(reference - ours).max()
    print(f"64 V pairs    max |diff| = {diff:.2e}  mean tokens = {timing['mean_tokens']:.0f}")
    assert diff < 1e-3
    print("RECIPE_OK")


if __name__ == "__main__":
    main()
