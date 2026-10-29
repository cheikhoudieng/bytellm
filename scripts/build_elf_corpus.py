"""
Build a large, deduplicated corpus of executables/libraries (ELF + PE) for ByteLLM.

- Walks one or more root directories in random order (so a size cap does not just
  keep the alphabetically-first folders).
- Deduplicates by SHA-256 of the content (or of the .text section with --text-only).
- Never copies anything whose content matches a file in --exclude (held-out set),
  which prevents exact train/test leakage.

Example (Colab):
    python scripts/build_elf_corpus.py --roots /content/deb_extract /usr/bin \
        --out corpus_big --max-mb 1500 --exclude heldout_libs
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path
from typing import Optional, Set

try:
    from bytellm.extractor import extract_executable_code
except ModuleNotFoundError:  # run from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bytellm.extractor import extract_executable_code

MAGICS = (b"\x7fELF", b"MZ")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _payload(path: Path, text_only: bool) -> Optional[bytes]:
    if text_only:
        code = extract_executable_code(path)
        return code if code and len(code) >= 4096 else None
    return path.read_bytes()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True, help="directories to scan recursively")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-mb", type=float, default=1000.0, help="stop after this many MB written")
    ap.add_argument("--min-kb", type=float, default=16.0)
    ap.add_argument("--max-file-mb", type=float, default=64.0)
    ap.add_argument("--exclude", nargs="*", default=[], help="held-out directories (never copied)")
    ap.add_argument("--text-only", action="store_true", help="keep only the executable .text section")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    excluded: Set[str] = set()
    for d in args.exclude:
        for p in Path(d).rglob("*"):
            if p.is_file():
                data = _payload(p, args.text_only)
                if data:
                    excluded.add(_digest(data))
    print(f"held-out fingerprints: {len(excluded)}", flush=True)

    candidates = []
    for root in args.roots:
        for dirpath, _dirs, files in os.walk(root, followlinks=False):
            for name in files:
                candidates.append(Path(dirpath) / name)
    random.Random(args.seed).shuffle(candidates)
    print(f"candidate files: {len(candidates):,}", flush=True)

    seen: Set[str] = set()
    written = skipped_dup = skipped_heldout = 0
    total = 0
    cap = int(args.max_mb * 1024 * 1024)
    for path in candidates:
        try:
            if path.is_symlink() or not path.is_file():
                continue
            size = path.stat().st_size
            if size < args.min_kb * 1024 or size > args.max_file_mb * 1024 * 1024:
                continue
            with open(path, "rb") as fh:
                head = fh.read(4)
            if not head.startswith(MAGICS):
                continue
            data = _payload(path, args.text_only)
            if not data:
                continue
            h = _digest(data)
            if h in excluded:
                skipped_heldout += 1
                continue
            if h in seen:
                skipped_dup += 1
                continue
            seen.add(h)
            (args.out / f"{written:06d}_{path.name}.elf").write_bytes(data)
            written += 1
            total += len(data)
            if written % 500 == 0:
                print(f"  {written:,} files, {total / 1e6:,.0f} MB", flush=True)
            if total >= cap:
                break
        except (OSError, ValueError):
            continue

    print(
        f"done: {written:,} unique files, {total / 1e6:,.1f} MB "
        f"(duplicates skipped: {skipped_dup:,}, held-out matches skipped: {skipped_heldout:,})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
