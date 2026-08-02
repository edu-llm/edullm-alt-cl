#!/usr/bin/env python3
"""Build local math SFT mix: MetaMathQA + no_robots + UltraChat (~60M Dolma2 tokens).

Targets (by Dolma2 tokens on concatenated message text):
  - MetaMathQA:  ~42.0M  (70%)
  - no_robots:   ~2.6M   (full train; ~all of it)
  - UltraChat:   ~15.4M  (fill remaining general)

Output: one shuffled ``train.jsonl.gz`` of ``{"id", "source", "messages"}`` rows.
No held-out split. Not an eduLLM publish — local training artifact only.

Example::

    python scripts/build_math_sft_60m.py --out-dir artifacts/math-sft-60m
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

log = logging.getLogger("build_math_sft_60m")

DEFAULT_SEED = 42069666
DEFAULT_TOTAL = 60_000_000
METAMATH_FRAC = 0.70
NO_ROBOTS_TARGET = 2_600_000
# UltraChat fills the rest of the 30% general slice after no_robots.

METAMATH_ID = "meta-math/MetaMathQA"
METAMATH_REV = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
NO_ROBOTS_ID = "HuggingFaceH4/no_robots"
NO_ROBOTS_REV = "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b"
ULTRACHAT_ID = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_REV = "8049631c405ae6576f93f445c6b8166f76f5505a"
TOKENIZER_ID = "allenai/dolma2-tokenizer"


def rank_key(seed: int, *parts: str) -> str:
    h = hashlib.sha256()
    h.update(f"{seed}".encode())
    for p in parts:
        h.update(b"\0")
        h.update(p.encode("utf-8", errors="replace"))
    return h.hexdigest()


def messages_text(messages: Sequence[Dict[str, str]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


def count_tokens(tok: Any, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def estimate_tokens(text: str) -> int:
    # Dolma2 ≈ 4.3 chars/token on no_robots; slight overestimate → undershoot then OK.
    return max(1, int(len(text) / 4.0))


def normalize_messages(raw: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(raw, list) or not raw:
        return None
    role_map = {
        "human": "user",
        "prompter": "user",
        "gpt": "assistant",
        "bot": "assistant",
    }
    out: List[Dict[str, str]] = []
    for m in raw:
        if not isinstance(m, dict):
            return None
        role_raw = str(m.get("role") or "").strip().lower()
        role = role_map.get(role_raw, role_raw)
        content = m.get("content")
        if content is None:
            return None
        content_s = str(content).strip()
        if not content_s:
            return None
        if role not in ("system", "user", "assistant", "tool"):
            return None
        out.append({"role": role, "content": content_s})
    if not any(m["role"] == "assistant" for m in out):
        return None
    return out


def metamath_to_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    q = str(row.get("query") or "").strip()
    a = str(row.get("response") or "").strip()
    if not q or not a:
        return None
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def iter_candidates(
    *,
    source: str,
    hf_id: str,
    revision: str,
    split: str,
    seed: int,
) -> Iterator[Tuple[str, List[Dict[str, str]], int]]:
    """Yield (rank_key, messages, estimated_tokens)."""
    from datasets import load_dataset

    log.info("loading %s @ %s split=%s", hf_id, revision[:12], split)
    ds = load_dataset(hf_id, split=split, revision=revision)
    n = len(ds)
    log.info("%s: %s rows", source, f"{n:,}")
    t0 = time.time()
    kept = 0
    to_messages = (
        metamath_to_messages
        if source == "metamath"
        else (lambda r: normalize_messages(r.get("messages")))
    )
    for i, row in enumerate(ds):
        msgs = to_messages(row)
        if not msgs:
            continue
        text = messages_text(msgs)
        n_est = estimate_tokens(text)
        key = rank_key(seed, source, text[:2000], str(i))
        yield key, msgs, n_est
        kept += 1
        if (i + 1) % 50_000 == 0:
            log.info(
                "%s: scanned %s/%s kept=%s (%.1f min)",
                source,
                f"{i+1:,}",
                f"{n:,}",
                f"{kept:,}",
                (time.time() - t0) / 60.0,
            )


def select_until_budget(
    candidates: Iterator[Tuple[str, List[Dict[str, str]], int]],
    *,
    budget: int,
    source: str,
    tok: Any,
) -> Tuple[List[Dict[str, Any]], int]:
    """Sort by rank_key, take by estimate with margin, then exact-tokenize and trim."""
    rows: List[Tuple[str, List[Dict[str, str]], int]] = list(candidates)
    rows.sort(key=lambda x: x[0])
    # Over-collect on estimates so exact Dolma2 trim can hit the budget.
    est_budget = int(budget * 1.25) + 50_000
    prelim: List[Tuple[str, List[Dict[str, str]]]] = []
    est_total = 0
    for key, msgs, n_est in rows:
        if est_total >= est_budget:
            break
        prelim.append((key, msgs))
        est_total += n_est

    selected: List[Dict[str, Any]] = []
    total = 0
    for key, msgs in prelim:
        if total >= budget:
            break
        n_tok = count_tokens(tok, messages_text(msgs))
        if n_tok <= 0:
            continue
        selected.append(
            {
                "id": f"{source}-{key[:16]}",
                "source": source,
                "messages": msgs,
                "n_tokens": n_tok,
            }
        )
        total += n_tok
    log.info(
        "%s: selected %s examples / %s tokens (budget %s; pool %s; prelim %s)",
        source,
        f"{len(selected):,}",
        f"{total:,}",
        f"{budget:,}",
        f"{len(rows):,}",
        f"{len(prelim):,}",
    )
    return selected, total


def write_jsonl_gz(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            out = {
                "id": row["id"],
                "source": row["source"],
                "messages": row["messages"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/math-sft-60m"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--total-tokens", type=int, default=DEFAULT_TOTAL)
    ap.add_argument("--metamath-tokens", type=int, default=None)
    ap.add_argument("--no-robots-tokens", type=int, default=NO_ROBOTS_TARGET)
    ap.add_argument("--ultrachat-tokens", type=int, default=None)
    ap.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="If >0, cap each source to this many examples (debug)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    total = int(args.total_tokens)
    math_budget = int(args.metamath_tokens) if args.metamath_tokens else int(total * METAMATH_FRAC)
    general_budget = total - math_budget
    nr_budget = min(int(args.no_robots_tokens), general_budget)
    uc_budget = (
        int(args.ultrachat_tokens)
        if args.ultrachat_tokens is not None
        else max(general_budget - nr_budget, 0)
    )

    log.info(
        "budgets: total=%s math=%s no_robots=%s ultrachat=%s",
        f"{total:,}",
        f"{math_budget:,}",
        f"{nr_budget:,}",
        f"{uc_budget:,}",
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)

    def cap(it: Iterator) -> Iterator:
        if not args.smoke:
            yield from it
            return
        for i, x in enumerate(it):
            if i >= args.smoke:
                break
            yield x

    metamath, math_toks = select_until_budget(
        cap(
            iter_candidates(
                source="metamath",
                hf_id=METAMATH_ID,
                revision=METAMATH_REV,
                split="train",
                seed=args.seed,
            )
        ),
        budget=math_budget,
        source="metamath",
        tok=tok,
    )
    no_robots, nr_toks = select_until_budget(
        cap(
            iter_candidates(
                source="no_robots",
                hf_id=NO_ROBOTS_ID,
                revision=NO_ROBOTS_REV,
                split="train",
                seed=args.seed,
            )
        ),
        budget=nr_budget,
        source="no_robots",
        tok=tok,
    )
    ultrachat, uc_toks = select_until_budget(
        cap(
            iter_candidates(
                source="ultrachat",
                hf_id=ULTRACHAT_ID,
                revision=ULTRACHAT_REV,
                split="train_sft",
                seed=args.seed,
            )
        ),
        budget=uc_budget,
        source="ultrachat",
        tok=tok,
    )

    mixed = metamath + no_robots + ultrachat
    rng = random.Random(args.seed)
    rng.shuffle(mixed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl.gz"
    write_jsonl_gz(out_path, mixed)

    stats = {
        "dataset": "math-sft-60m",
        "seed": args.seed,
        "tokenizer": TOKENIZER_ID,
        "token_accounting": "dolma2 encode of concatenated message contents (no specials)",
        "budgets": {
            "total": total,
            "metamath": math_budget,
            "no_robots": nr_budget,
            "ultrachat": uc_budget,
        },
        "realized": {
            "total_examples": len(mixed),
            "total_tokens": math_toks + nr_toks + uc_toks,
            "metamath_examples": len(metamath),
            "metamath_tokens": math_toks,
            "no_robots_examples": len(no_robots),
            "no_robots_tokens": nr_toks,
            "ultrachat_examples": len(ultrachat),
            "ultrachat_tokens": uc_toks,
        },
        "revisions": {
            METAMATH_ID: METAMATH_REV,
            NO_ROBOTS_ID: NO_ROBOTS_REV,
            ULTRACHAT_ID: ULTRACHAT_REV,
        },
        "output": str(out_path.as_posix()),
        "format": {
            "container": "jsonl.gz",
            "row": {"id": "str", "source": "str", "messages": "[{role,content}, ...]"},
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    readme = f"""# math-sft-60m

Local SFT mix for edullm-alt-cl (not published to edullm-data).

| Source | Target tokens | Realized tokens | Examples |
|---|---:|---:|---:|
| MetaMathQA | {math_budget:,} | {math_toks:,} | {len(metamath):,} |
| no_robots | {nr_budget:,} | {nr_toks:,} | {len(no_robots):,} |
| UltraChat 200k | {uc_budget:,} | {uc_toks:,} | {len(ultrachat):,} |
| **Total** | {total:,} | {math_toks + nr_toks + uc_toks:,} | {len(mixed):,} |

- File: `train.jsonl.gz` — shuffled (`seed={args.seed}`)
- Row: `{{"id", "source", "messages": [{{"role","content"}}, ...]}}`
- Tokenizer for budgets: `{TOKENIZER_ID}`

## Use

```python
import gzip, json
with gzip.open("artifacts/math-sft-60m/train.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        messages = row["messages"]  # apply chat template + mask prompts in your SFT script
```

Rebuild: `python scripts/build_math_sft_60m.py --out-dir artifacts/math-sft-60m`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    log.info("wrote %s (%s examples)", out_path, f"{len(mixed):,}")
    log.info("manifest: %s", out_dir / "manifest.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
