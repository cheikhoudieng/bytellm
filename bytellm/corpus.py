"""
Real-corpus pipeline: mixed text + binary files across every domain, instruction
pairs (natural language -> any file type), train/val split per file, domain
mixture sampling, and per-domain evaluation (BPB + validity rate).
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler

from bytellm.codec import (
    DOMAIN_TO_TOKEN,
    SPECIAL_TOKENS,
    BinaryDomain,
    BinaryValidator,
    ByteCodec,
    compute_shannon_entropy,
)
from bytellm.model import ByteLLM
from bytellm.training import evaluate_bits_per_byte

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".c", ".h", ".cpp", ".rs", ".go",
    ".java", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".html", ".css", ".xml",
    ".csv", ".tex", ".sh", ".log", ".ini", ".wat", ".asm", ".s",
}
BINARY_EXTENSIONS: Dict[str, BinaryDomain] = {
    ".wasm": BinaryDomain.WASM,
    ".mid": BinaryDomain.MIDI,
    ".midi": BinaryDomain.MIDI,
    ".wav": BinaryDomain.AUDIO,
    ".pcap": BinaryDomain.NETWORK,
    ".gz": BinaryDomain.COMPRESSED,
    ".zlib": BinaryDomain.COMPRESSED,
    ".elf": BinaryDomain.ASM,
    ".exe": BinaryDomain.ASM,
    ".dll": BinaryDomain.ASM,
    ".so": BinaryDomain.ASM,
    ".o": BinaryDomain.ASM,
}

# Domains accepted as the target of an instruction pair.
_DOMAIN_NAMES = {d.value: d for d in BinaryDomain}


def classify_file(path: Path) -> BinaryDomain:
    """Pick a domain from the extension, falling back to magic-byte detection."""
    ext = path.suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return BINARY_EXTENSIONS[ext]
    if ext in TEXT_EXTENSIONS:
        return BinaryDomain.TEXT
    with open(path, "rb") as fh:
        head = fh.read(4096)
    return ByteCodec.detect_domain(head)


def is_validation_file(path: Union[str, Path], val_fraction: float = 0.1) -> bool:
    """Deterministic split by file (never by chunk) to avoid train/val leakage."""
    h = zlib.crc32(str(Path(path).as_posix()).encode("utf-8")) % 1000
    return h < int(val_fraction * 1000)


@dataclass
class _Record:
    domain: BinaryDomain
    path: Optional[Path] = None
    offset: int = 0
    length: int = 0
    last: bool = True
    tokens: Optional[List[int]] = None  # for in-memory (instruction) records
    response_start: int = 0


class CorpusDataset(Dataset):
    """
    Lazily chunked dataset over real files plus optional instruction pairs.

    Every sample is `[BOS][DOMAIN_X] bytes... [EOS on last chunk]` of at most
    `seq_len + 1` tokens. Instruction pairs follow `ByteCodec.encode_instruction`
    and the loss on the instruction itself is masked (PAD) so the model learns
    to *answer*, not to recite prompts.

    Instruction JSONL line:
        {"instruction": "...", "domain": "midi", "response_path": "x.mid"}
        {"instruction": "...", "domain": "text", "response": "...", "reasoning": "..."}
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        instructions: Optional[Union[str, Path]] = None,
        split: str = "train",
        seq_len: int = 256,
        val_fraction: float = 0.1,
        max_file_bytes: int = 64 * 1024 * 1024,
        compute_entropy: bool = False,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "all")
        self.seq_len = seq_len
        self.compute_entropy = compute_entropy
        self.records: List[_Record] = []
        self._file_cache: Dict[Path, bytes] = {}
        want_val = split == "val"
        use_all = split == "all"  # no train/val split: every file (held-out test sets)

        if root is not None:
            for path in sorted(Path(root).rglob("*")):
                if not path.is_file() or (
                    not use_all and is_validation_file(path, val_fraction) != want_val
                ):
                    continue
                size = path.stat().st_size
                if size == 0 or size > max_file_bytes:
                    continue
                self._add_file(path, classify_file(path), size)

        if instructions is not None:
            base = Path(instructions).parent
            with open(instructions, "r", encoding="utf-8") as fh:
                for n, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    key = f"{Path(instructions).name}:{n}"
                    if not use_all and is_validation_file(key, val_fraction) != want_val:
                        continue
                    self._add_instruction(json.loads(line), base)

    def _add_file(self, path: Path, domain: BinaryDomain, size: int) -> None:
        payload = self.seq_len + 1 - 3  # BOS + DOMAIN + EOS
        for off in range(0, size, payload):
            self.records.append(
                _Record(domain, path, off, min(payload, size - off), off + payload >= size)
            )

    def _add_instruction(self, item: Dict, base: Path) -> None:
        domain = _DOMAIN_NAMES[item.get("domain", "text")]
        if "response_path" in item:
            response: Union[bytes, str] = (base / item["response_path"]).read_bytes()
        else:
            response = item.get("response", "")
        tokens, start = ByteCodec.encode_instruction(
            item["instruction"], response, domain, item.get("reasoning")
        )
        self.records.append(
            _Record(domain, tokens=tokens[: self.seq_len + 1], response_start=start)
        )

    @property
    def domains(self) -> List[BinaryDomain]:
        return [r.domain for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def _tokens(self, r: _Record) -> Tuple[List[int], int]:
        if r.tokens is not None:
            return list(r.tokens), r.response_start
        assert r.path is not None
        buf = self._file_cache.get(r.path)
        if buf is None:
            buf = r.path.read_bytes()
            if len(self._file_cache) < 2048:
                self._file_cache[r.path] = buf
        chunk = buf[r.offset : r.offset + r.length]
        toks = [SPECIAL_TOKENS.BOS, DOMAIN_TO_TOKEN[r.domain]] + list(chunk)
        if r.last:
            toks.append(SPECIAL_TOKENS.EOS)
        return toks, 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, resp_start = self._tokens(self.records[idx])
        n = self.seq_len + 1
        if len(tokens) < n:
            tokens = tokens + [SPECIAL_TOKENS.PAD] * (n - len(tokens))
        else:
            tokens = tokens[:n]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        if resp_start > 0:
            y[: max(0, resp_start - 1)] = SPECIAL_TOKENS.PAD  # mask the instruction
        if self.compute_entropy:
            raw = [t if 0 <= t <= 255 else 0 for t in tokens[:-1]]
            ent = torch.from_numpy(compute_shannon_entropy(raw, window_size=16)).float()
        else:
            ent = torch.empty(0, dtype=torch.float32)
        return x, y, ent


def parse_domain_weights(spec: Optional[str]) -> Dict[BinaryDomain, float]:
    """Parse `text=0.4,wasm=0.1,...`; unspecified domains default to 1.0."""
    weights: Dict[BinaryDomain, float] = {}
    if spec:
        for part in spec.split(","):
            name, value = part.split("=")
            weights[_DOMAIN_NAMES[name.strip()]] = float(value)
    return weights


def build_mixture_sampler(
    dataset: CorpusDataset,
    weights: Optional[Dict[BinaryDomain, float]] = None,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Sample domains in the requested proportions regardless of corpus size."""
    weights = weights or {}
    doms = dataset.domains
    counts: Dict[BinaryDomain, int] = {}
    for d in doms:
        counts[d] = counts.get(d, 0) + 1
    per_sample = [weights.get(d, 1.0) / counts[d] for d in doms]
    return WeightedRandomSampler(
        per_sample, num_samples or len(doms), replacement=True
    )


def _spread(idxs: List[int], n: int) -> List[int]:
    """Pick `n` evenly spaced indices so the sample covers every file, not just the first ones."""
    if n <= 0 or len(idxs) <= n:
        return idxs
    step = len(idxs) / n
    return [idxs[int(i * step)] for i in range(n)]


def evaluate_per_domain(
    model: ByteLLM,
    dataset: CorpusDataset,
    batch_size: int = 8,
    max_samples_per_domain: int = 2000,
) -> Dict[str, Dict[str, float]]:
    """Bits-per-byte, byte perplexity and sample count for each domain separately."""
    by_domain: Dict[BinaryDomain, List[int]] = {}
    for i, d in enumerate(dataset.domains):
        by_domain.setdefault(d, []).append(i)
    return {
        d.value: {
            **evaluate_bits_per_byte(
                model,
                Subset(dataset, _spread(idxs, max_samples_per_domain)),
                batch_size=batch_size,
            ),
            "samples": float(len(idxs)),
        }
        for d, idxs in by_domain.items()
    }


@torch.no_grad()
def measure_validity_rate(
    model: ByteLLM,
    domain: BinaryDomain,
    n: int = 8,
    instruction: Optional[str] = None,
    constrained: bool = True,
    max_new_bytes: int = 64,
    temperature: float = 0.8,
) -> float:
    """
    Fraction of generated payloads that pass `BinaryValidator`. Run it with
    `constrained=True` and `False` to see what the model learned by itself
    versus what the grammar enforces.
    """
    ok = 0
    for _ in range(n):
        payload, _, _ = model.generate_bytes(
            domain=domain,
            instruction=instruction,
            max_new_bytes=max_new_bytes,
            temperature=temperature,
            use_grammar_constraints=constrained,
        )
        valid, _ = BinaryValidator.validate(payload, domain)
        ok += int(valid)
    return ok / max(1, n)
