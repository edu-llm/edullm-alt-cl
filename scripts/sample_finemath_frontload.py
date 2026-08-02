#!/usr/bin/env python3
"""Deterministic FineMath-4+ → ~100M Dolma2-token frontload sample (two-pass).

Pass 1: stream HF rows; write compact index ``(rank_key, url, offset, llama_toks)``.
Sort index; select docs until Llama-token budget covers target (margin).
Pass 2: stream again; tokenize selected docs with Dolma2; pack ``.u32le.bin``.

Reproducibility: same ``--hf-revision``, ``--seed``, ``--target-tokens``, tokenizer
⇒ same selection and token bytes.

Example::

    python scripts/sample_finemath_frontload.py \\
        --out-dir artifacts/math-frontload-100m \\
        --hf-revision e92b25a616738fe95dc186b64dfb19f9c8525594 \\
        --seed 42069666 \\
        --target-tokens 100000000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from alt_cl.finemath_rank import rank_key

log = logging.getLogger("sample_finemath_frontload")

HF_ID = "HuggingFaceTB/finemath"
HF_CONFIG = "finemath-4plus"
DEFAULT_SEED = 42069666
DEFAULT_TARGET = 100_000_000
VAL_FRACTION = 0.0015
TOKENIZER_ID = "allenai/dolma2-tokenizer"
# Llama token_count underestimates/overestimates Dolma2 slightly; keep margin.
LLAMA_BUDGET_MARGIN = 1.15
PINNED_REVISION = "e92b25a616738fe95dc186b64dfb19f9c8525594"


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def tokenize_doc(tok, text: str) -> List[int]:
    ids = tok.encode(text, add_special_tokens=False)
    eos = tok.eos_token_id
    if eos is not None:
        ids = list(ids) + [int(eos)]
    return ids


def write_u32le_shards(
    token_streams: Iterator[List[int]],
    out_dir: Path,
    split: str,
    *,
    max_tokens_per_shard: int = 50_000_000,
) -> Tuple[int, List[Path]]:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    buf: List[int] = []
    total = 0
    paths: List[Path] = []

    def flush() -> None:
        nonlocal shard_idx, buf
        if not buf:
            return
        path = out_dir / f"{split}-{shard_idx:05d}.u32le.bin"
        np.asarray(buf, dtype=np.uint32).tofile(path)
        paths.append(path)
        log.info("Wrote %s (%d tokens)", path.name, len(buf))
        shard_idx += 1
        buf = []

    for ids in token_streams:
        buf.extend(ids)
        total += len(ids)
        while len(buf) >= max_tokens_per_shard:
            chunk, buf = buf[:max_tokens_per_shard], buf[max_tokens_per_shard:]
            path = out_dir / f"{split}-{shard_idx:05d}.u32le.bin"
            np.asarray(chunk, dtype=np.uint32).tofile(path)
            paths.append(path)
            log.info("Wrote %s (%d tokens)", path.name, len(chunk))
            shard_idx += 1
    flush()
    return total, paths


def pass1_build_index(
    revision: str,
    seed: int,
    index_path: Path,
    *,
    max_docs: Optional[int] = None,
) -> int:
    """Stream dataset; write TSV index lines: key\\turl\\toffset\\tllama_toks."""
    from datasets import load_dataset

    log.info("Pass1: indexing %s @ %s", HF_CONFIG, revision)
    ds = load_dataset(
        HF_ID,
        HF_CONFIG,
        split="train",
        revision=revision,
        streaming=True,
    )
    n = 0
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        for row in ds:
            url = str(row.get("url") or "")
            off = row.get("warc_record_offset")
            llama_n = int(row.get("token_count") or 0)
            key = rank_key(seed, url, off)
            # tabs unlikely in url; replace just in case
            url_s = url.replace("\t", " ").replace("\n", " ")
            f.write(f"{key}\t{url_s}\t{off}\t{llama_n}\n")
            n += 1
            if max_docs is not None and n >= max_docs:
                break
            if n % 100_000 == 0:
                log.info("Pass1 indexed %d docs", n)
    log.info("Pass1 done: %d docs → %s", n, index_path)
    return n


def select_from_index(
    index_path: Path,
    sorted_path: Path,
    *,
    target_tokens: int,
    llama_margin: float = LLAMA_BUDGET_MARGIN,
) -> List[Tuple[str, str, Any, int]]:
    """Sort index externally via Python sort; return selected (key,url,off,llama)."""
    log.info("Sorting index …")
    lines = index_path.read_text(encoding="utf-8").splitlines()
    lines.sort()  # rank_key is first field
    sorted_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    log.info("Sorted %d lines", len(lines))

    budget = int(target_tokens * llama_margin)
    selected: List[Tuple[str, str, Any, int]] = []
    running = 0
    for line in lines:
        if not line.strip():
            continue
        key, url, off_s, llama_s = line.split("\t", 3)
        llama_n = int(llama_s)
        off: Any
        try:
            off = int(off_s)
        except ValueError:
            off = off_s
        if running >= budget and selected:
            break
        selected.append((key, url, off, llama_n))
        running += llama_n
    log.info(
        "Selected %d docs, sum_llama_tokens=%d (budget=%d)",
        len(selected),
        running,
        budget,
    )
    return selected


def pass2_tokenize(
    revision: str,
    selected: List[Tuple[str, str, Any, int]],
    tok,
    *,
    target_tokens: int,
    overshoot_tol: float,
) -> List[Dict[str, Any]]:
    """Re-stream dataset; tokenize docs whose (url, offset) is selected."""
    from datasets import load_dataset

    want: Set[Tuple[str, Any]] = {(url, off) for _, url, off, _ in selected}
    # Preserve rank order for packing / val carve
    order = {(url, off): i for i, (_, url, off, _) in enumerate(selected)}

    log.info("Pass2: tokenizing up to %d selected docs …", len(want))
    ds = load_dataset(
        HF_ID,
        HF_CONFIG,
        split="train",
        revision=revision,
        streaming=True,
    )
    by_key: Dict[int, Dict[str, Any]] = {}
    seen = 0
    for row in ds:
        url = str(row.get("url") or "")
        off = row.get("warc_record_offset")
        try:
            off_k: Any = int(off) if off is not None else off
        except (TypeError, ValueError):
            off_k = off
        # index used stringified int — normalize
        cand = (url, off_k)
        if cand not in want and (url, off) not in want:
            continue
        idx = order.get(cand, order.get((url, off)))
        if idx is None:
            continue
        text = row.get("text") or ""
        ids = tokenize_doc(tok, text)
        by_key[idx] = {
            "url": url,
            "warc_record_offset": off_k,
            "hf_llama_token_count": int(row.get("token_count") or 0),
            "score": row.get("score"),
            "int_score": row.get("int_score"),
            "dolma2_tokens": len(ids),
            "rank_key": selected[idx][0],
            "_ids": ids,
        }
        seen += 1
        if seen % 500 == 0:
            log.info("Pass2 tokenized %d / %d", seen, len(want))
        if seen >= len(want):
            break

    ordered = [by_key[i] for i in range(len(selected)) if i in by_key]
    log.info("Pass2 got %d docs (wanted %d)", len(ordered), len(selected))

    # Truncate by true Dolma2 count to target (± overshoot tol)
    out: List[Dict[str, Any]] = []
    running = 0
    max_allowed = int(target_tokens * (1.0 + overshoot_tol))
    for rec in ordered:
        n = int(rec["dolma2_tokens"])
        if running >= target_tokens:
            break
        if running + n > max_allowed and running > 0:
            break
        out.append(rec)
        running += n
    log.info("After Dolma2 truncate: %d docs, %d tokens", len(out), running)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hf-revision", default=PINNED_REVISION)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--overshoot-tol", type=float, default=0.02)
    ap.add_argument("--max-docs", type=int, default=None, help="Smoke: index only first N")
    ap.add_argument("--skip-pass1", action="store_true", help="Reuse existing index.tsv")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.tsv"
    sorted_path = out / "index_sorted.tsv"

    meta = {
        "hf_id": HF_ID,
        "hf_config": HF_CONFIG,
        "hf_revision": args.hf_revision,
        "seed": args.seed,
        "target_tokens": args.target_tokens,
        "val_fraction": VAL_FRACTION,
        "tokenizer_id": TOKENIZER_ID,
        "llama_budget_margin": LLAMA_BUDGET_MARGIN,
        "rank": "sha256(f'{seed}:{url}:{warc_record_offset}') ascending",
    }
    (out / "sample_config.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    if not args.skip_pass1 or not index_path.is_file():
        pass1_build_index(
            args.hf_revision, args.seed, index_path, max_docs=args.max_docs
        )

    selected = select_from_index(
        index_path,
        sorted_path,
        target_tokens=int(args.target_tokens),
        llama_margin=LLAMA_BUDGET_MARGIN,
    )
    (out / "selected_keys.json").write_text(
        json.dumps(
            [
                {
                    "rank_key": k,
                    "url": u,
                    "warc_record_offset": o,
                    "hf_llama_token_count": n,
                }
                for k, u, o, n in selected
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    tok = load_tokenizer()
    docs = pass2_tokenize(
        args.hf_revision,
        selected,
        tok,
        target_tokens=int(args.target_tokens),
        overshoot_tol=float(args.overshoot_tol),
    )

    n_val = max(1, int(round(len(docs) * VAL_FRACTION))) if docs else 0
    val_docs = docs[-n_val:] if n_val else []
    train_docs = docs[:-n_val] if n_val else docs

    def emit(chunk: List[Dict[str, Any]]) -> Iterator[List[int]]:
        for d in chunk:
            yield d["_ids"]

    tokens_root = out / "tokens" / "finemath4plus"
    train_n, train_paths = write_u32le_shards(emit(train_docs), tokens_root, "train")
    val_n, val_paths = write_u32le_shards(emit(val_docs), tokens_root, "val")

    with (out / "sample_manifest.jsonl").open("w", encoding="utf-8") as f:
        for split, chunk in (("train", train_docs), ("val", val_docs)):
            for d in chunk:
                row = {k: v for k, v in d.items() if k != "_ids"}
                row["split"] = split
                f.write(json.dumps(row) + "\n")

    summary = {
        **meta,
        "n_docs_train": len(train_docs),
        "n_docs_val": len(val_docs),
        "dolma2_tokens_train": train_n,
        "dolma2_tokens_val": val_n,
        "train_shards": [p.name for p in train_paths],
        "val_shards": [p.name for p in val_paths],
    }
    (out / "sample_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    log.info("Done: train=%d val=%d tokens", train_n, val_n)


if __name__ == "__main__":
    main()
