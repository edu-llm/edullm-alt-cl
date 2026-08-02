"""Conversation SFT tokenization with Dolma2 chat template + prompt masking."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

log = logging.getLogger("alt_cl.sft_data")

IGNORE_INDEX = -100
DEFAULT_TOKENIZER_ID = "allenai/dolma2-tokenizer"
DEFAULT_SEQ_LEN = 2048


def iter_jsonl_gz(path: Path | str) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def load_tokenizer(tokenizer_id: str = DEFAULT_TOKENIZER_ID):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)


def messages_to_ids_and_labels(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    *,
    max_length: int = DEFAULT_SEQ_LEN,
) -> Optional[Tuple[List[int], List[int]]]:
    """Tokenize a chat with loss only on assistant completion spans.

    Uses string ``apply_chat_template`` + encode (Dolma2's ``tokenize=True`` path
    is unreliable). Assistant headers are *not* trained; content + end markers are.
    """
    if not messages:
        return None
    try:
        full_text = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("chat_template failed: %s", exc)
        return None
    if not isinstance(full_text, str) or not full_text:
        return None

    input_ids: List[int] = tokenizer.encode(full_text, add_special_tokens=False)
    if not input_ids:
        return None
    labels: List[int] = [IGNORE_INDEX] * len(input_ids)

    for i, msg in enumerate(messages):
        if str(msg.get("role") or "").lower() != "assistant":
            continue
        try:
            before_text = tokenizer.apply_chat_template(
                list(messages[:i]),
                tokenize=False,
                add_generation_prompt=True,
            )
            through_text = tokenizer.apply_chat_template(
                list(messages[: i + 1]),
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            continue
        if not (
            isinstance(before_text, str)
            and isinstance(through_text, str)
            and through_text.startswith(before_text)
        ):
            continue
        before_ids = tokenizer.encode(before_text, add_special_tokens=False)
        through_ids = tokenizer.encode(through_text, add_special_tokens=False)
        if through_ids[: len(before_ids)] != before_ids:
            continue
        if len(through_ids) > len(input_ids) or input_ids[: len(through_ids)] != through_ids:
            # Fall back: locate through_ids as a prefix of a longer encode of full_text
            # already stored in input_ids — skip if inconsistent.
            continue
        for j in range(len(before_ids), len(through_ids)):
            labels[j] = input_ids[j]

    if all(x == IGNORE_INDEX for x in labels):
        return None

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        # Drop examples that lost all supervised tokens to truncation.
        if all(x == IGNORE_INDEX for x in labels):
            return None
    return input_ids, labels


def pack_examples(
    examples: Sequence[Tuple[List[int], List[int]]],
    *,
    seq_len: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack variable-length (ids, labels) into fixed ``[n_pack, seq_len]`` tensors.

    Concatenates examples with no extra separator (EOS already in chat template).
    Pads the final pack with ``pad_token_id`` / ``IGNORE_INDEX``.
    """
    packs_ids: List[List[int]] = []
    packs_lab: List[List[int]] = []
    cur_ids: List[int] = []
    cur_lab: List[int] = []

    def flush() -> None:
        nonlocal cur_ids, cur_lab
        if not cur_ids:
            return
        if len(cur_ids) < seq_len:
            pad_n = seq_len - len(cur_ids)
            cur_ids = cur_ids + [pad_token_id] * pad_n
            cur_lab = cur_lab + [IGNORE_INDEX] * pad_n
        packs_ids.append(cur_ids[:seq_len])
        packs_lab.append(cur_lab[:seq_len])
        cur_ids, cur_lab = [], []

    for ids, lab in examples:
        if len(ids) > seq_len:
            ids = ids[:seq_len]
            lab = lab[:seq_len]
        if len(ids) == 0:
            continue
        if len(cur_ids) + len(ids) > seq_len:
            flush()
        if len(ids) == seq_len:
            packs_ids.append(list(ids))
            packs_lab.append(list(lab))
            continue
        cur_ids.extend(ids)
        cur_lab.extend(lab)
    flush()

    if not packs_ids:
        empty_i = torch.zeros((0, seq_len), dtype=torch.long)
        empty_l = torch.full((0, seq_len), IGNORE_INDEX, dtype=torch.long)
        return empty_i, empty_l
    return (
        torch.tensor(packs_ids, dtype=torch.long),
        torch.tensor(packs_lab, dtype=torch.long),
    )


class ConversationSFTDataset(Dataset):
    """In-memory SFT rows from ``train.jsonl.gz`` (tokenized + masked)."""

    def __init__(
        self,
        path: Path | str,
        *,
        tokenizer: Any = None,
        tokenizer_id: str = DEFAULT_TOKENIZER_ID,
        max_length: int = DEFAULT_SEQ_LEN,
        max_examples: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer or load_tokenizer(tokenizer_id)
        self.max_length = int(max_length)
        self.rows: List[Tuple[List[int], List[int]]] = []
        n_in = 0
        n_ok = 0
        for obj in iter_jsonl_gz(self.path):
            n_in += 1
            if max_examples is not None and n_ok >= int(max_examples):
                break
            msgs = obj.get("messages")
            if not isinstance(msgs, list):
                continue
            built = messages_to_ids_and_labels(
                self.tokenizer, msgs, max_length=self.max_length
            )
            if built is None:
                continue
            self.rows.append(built)
            n_ok += 1
            if n_ok % 25_000 == 0:
                log.info("tokenized %s conversations from %s", f"{n_ok:,}", self.path)
        log.info(
            "ConversationSFTDataset: kept %s / %s conversations from %s",
            f"{n_ok:,}",
            f"{n_in:,}",
            self.path,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.rows[idx]

    def packed_tensors(
        self, *, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return pack_examples(
            self.rows,
            seq_len=int(seq_len or self.max_length),
            pad_token_id=pad_id,
        )
