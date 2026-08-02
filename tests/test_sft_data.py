"""Unit tests for SFT chat masking + packing (no OLMo-core / GPU)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import torch

from alt_cl.sft_data import (
    IGNORE_INDEX,
    ConversationSFTDataset,
    load_tokenizer,
    messages_to_ids_and_labels,
    pack_examples,
)


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer()


def test_assistant_tokens_are_supervised(tok):
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    built = messages_to_ids_and_labels(tok, msgs, max_length=128)
    assert built is not None
    ids, labels = built
    assert len(ids) == len(labels)
    assert any(l != IGNORE_INDEX for l in labels)
    # Supervised labels must equal the underlying token ids.
    for i, lab in enumerate(labels):
        if lab != IGNORE_INDEX:
            assert lab == ids[i]
    # Prompt-heavy prefix should be masked.
    assert labels[0] == IGNORE_INDEX


def test_multi_turn_masks_both_assistant_spans(tok):
    msgs = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "3+3?"},
        {"role": "assistant", "content": "6"},
    ]
    built = messages_to_ids_and_labels(tok, msgs, max_length=256)
    assert built is not None
    _, labels = built
    n_sup = sum(1 for l in labels if l != IGNORE_INDEX)
    assert n_sup >= 2


def test_pack_examples_pads_and_preserves_labels(tok):
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]
    built = messages_to_ids_and_labels(tok, msgs, max_length=64)
    assert built is not None
    ids, labels = pack_examples([built, built], seq_len=64, pad_token_id=int(tok.pad_token_id))
    assert ids.shape == labels.shape
    assert ids.shape[1] == 64
    assert ids.shape[0] >= 1
    assert (labels == IGNORE_INDEX).any()
    assert (labels != IGNORE_INDEX).any()


def test_dataset_from_jsonl_gz(tmp_path: Path, tok):
    path = tmp_path / "toy.jsonl.gz"
    rows = [
        {
            "id": "t1",
            "source": "toy",
            "messages": [
                {"role": "user", "content": "1+1?"},
                {"role": "assistant", "content": "2"},
            ],
        }
    ]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ds = ConversationSFTDataset(path, tokenizer=tok, max_length=128)
    assert len(ds) == 1
    ids, labels = ds[0]
    assert len(ids) == len(labels)
    assert any(l != IGNORE_INDEX for l in labels)
    packed_i, packed_l = ds.packed_tensors(seq_len=128)
    assert packed_i.shape[0] >= 1
    assert packed_i.shape == packed_l.shape
