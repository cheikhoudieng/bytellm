"""
Benchmark suite for ByteLLM (designed for Google Colab, works anywhere).

Suites
------
enwik8   : standard byte-level language-modelling benchmark (bits-per-byte on a
           held-out split) next to classical compressors (zlib / bz2 / lzma) and
           the order-0 entropy, so the number can be put in context.
domains  : validity rate of generated files per domain, with and without grammar
           constraints (the unconstrained rate is what the model really learned),
           plus per-domain BPB on freshly synthesised held-out data.
speed    : forward throughput (bytes/s) and parameter count.

Usage
-----
    python -m bytellm.benchmark --preset small --steps 3000 --max-mb 20
    python -m bytellm.benchmark --preset tiny --steps 200 --max-mb 2 --quick
"""

from __future__ import annotations

import argparse
import bz2
import json
import lzma
import math
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from bytellm.codec import BinaryDomain, compute_shannon_entropy
from bytellm.corpus import measure_validity_rate
from bytellm.model import (
    ByteLLM,
    ByteLLMConfig,
    TransformerBlock,
    VanillaByteTransformer,
)
from bytellm.training import (
    ByteLLMTrainer,
    MultiModalBinaryDataset,
    TrainingConfig,
    evaluate_bits_per_byte,
)

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"

PRESETS: Dict[str, Dict[str, int]] = {
    "tiny": dict(d_model=128, n_layers=3, n_heads=4),     # ~1M params
    "small": dict(d_model=384, n_layers=6, n_heads=6),    # ~20M params
    "base": dict(d_model=512, n_layers=8, n_heads=8),     # ~35M params
    "large": dict(d_model=768, n_layers=12, n_heads=12),  # ~110M params
}


def make_config(
    preset: str,
    seq_len: int,
    reasoning_steps: int,
    patch_size: int = 4,
    use_bitplane: bool = True,
    use_local_encoder: bool = True,
) -> ByteLLMConfig:
    p = PRESETS[preset]
    return ByteLLMConfig(
        d_model=p["d_model"],
        n_layers=p["n_layers"],
        n_heads=p["n_heads"],
        d_ff=int(p["d_model"] * 2.75),
        max_seq_len=seq_len,
        reasoning_steps=reasoning_steps,
        patch_size=patch_size,
        use_bitplane=use_bitplane,
        use_local_encoder=use_local_encoder,
    )


def match_vanilla_layers(cfg: ByteLLMConfig) -> int:
    """Number of vanilla blocks whose total parameters best match the ByteLLM config."""
    target = ByteLLM(cfg).count_parameters()
    block = TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
    per_block = sum(p.numel() for p in block.parameters())
    base = cfg.vocab_size * cfg.d_model + cfg.d_model  # tied embedding + final norm
    return max(1, round((target - base) / per_block))


def make_tag(args) -> str:
    if args.arch == "vanilla":
        return f"vanilla_{args.preset}_s{args.seed}"
    flags = ""
    if args.no_bitplane:
        flags += "_nobit"
    if args.no_local_encoder:
        flags += "_noconv"
    return f"bytellm_{args.preset}_r{args.reasoning_steps}_p{args.patch_size}{flags}_s{args.seed}"


class ByteStreamDataset(Dataset):
    """Contiguous raw-byte windows (no BOS/EOS): the standard enwik8 protocol."""

    def __init__(self, data: bytes, seq_len: int) -> None:
        self.buf = np.frombuffer(data, dtype=np.uint8)
        self.seq_len = seq_len
        self.n = max(0, (len(self.buf) - 1) // seq_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        s = i * self.seq_len
        chunk = self.buf[s : s + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        ent = torch.from_numpy(compute_shannon_entropy(chunk[:-1].astype(np.uint8))).float()
        return x, y, ent


def load_enwik8(cache_dir: Path, max_bytes: int) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = cache_dir / "enwik8"
    if not raw.exists():
        zpath = cache_dir / "enwik8.zip"
        print(f"downloading {ENWIK8_URL} ...", flush=True)
        urllib.request.urlretrieve(ENWIK8_URL, zpath)
        with zipfile.ZipFile(zpath) as zf:
            zf.extract("enwik8", cache_dir)
    with open(raw, "rb") as fh:
        return fh.read(max_bytes)


def baseline_bpb(data: bytes) -> Dict[str, float]:
    """Reference numbers on the same bytes (compressors see the whole stream)."""
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return {
        "order0_entropy": float(-(p * np.log2(p)).sum()),
        "zlib_9": len(zlib.compress(data, 9)) * 8 / len(data),
        "bz2_9": len(bz2.compress(data, 9)) * 8 / len(data),
        "lzma_9": len(lzma.compress(data, preset=9)) * 8 / len(data),
    }


@torch.no_grad()
def measure_speed(model: ByteLLM, seq_len: int, device: str, batch: int = 8, iters: int = 5) -> Dict[str, float]:
    model.eval()
    x = torch.randint(0, 256, (batch, seq_len), device=device)
    model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return {"forward_bytes_per_sec": batch * seq_len / dt}


def run_enwik8(args, device: str) -> Dict:
    data = load_enwik8(Path(args.cache_dir), int(args.max_mb * 1_000_000))
    n = len(data)
    train_b, val_b, test_b = data[: int(n * 0.9)], data[int(n * 0.9) : int(n * 0.95)], data[int(n * 0.95) :]
    train_ds = ByteStreamDataset(train_b, args.seq_len)
    val_ds = ByteStreamDataset(val_b, args.seq_len)
    test_ds = ByteStreamDataset(test_b, args.seq_len)

    torch.manual_seed(args.seed)
    cfg = make_config(
        args.preset, args.seq_len, args.reasoning_steps, args.patch_size,
        not args.no_bitplane, not args.no_local_encoder,
    )
    if args.arch == "vanilla":
        cfg.n_layers = args.vanilla_layers or match_vanilla_layers(cfg)
        model = VanillaByteTransformer(cfg)
    else:
        model = ByteLLM(cfg)
    tag = make_tag(args)
    print(f"model={tag} params={model.count_parameters():,}", flush=True)
    trainer = ByteLLMTrainer(
        model,
        TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            min_learning_rate=args.lr / 10,
            max_steps=args.steps,
            warmup_steps=max(1, args.steps // 20),
            seq_len=args.seq_len,
            device=device,
            amp=args.amp,
            log_every=args.log_every,
            seed=args.seed,
        ),
    )
    t0 = time.time()
    history = trainer.train(train_ds, num_steps=args.steps)
    train_time = time.time() - t0

    val = evaluate_bits_per_byte(model, val_ds, batch_size=args.batch_size, device=device)
    test = evaluate_bits_per_byte(model, test_ds, batch_size=args.batch_size, device=device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(out_dir / f"{tag}.pt")

    return {
        "tag": tag,
        "arch": args.arch,
        "preset": args.preset,
        "seed": args.seed,
        "reasoning_steps": args.reasoning_steps if args.arch == "bytellm" else None,
        "patch_size": args.patch_size if args.arch == "bytellm" else None,
        "bitplane": (not args.no_bitplane) if args.arch == "bytellm" else None,
        "local_encoder": (not args.no_local_encoder) if args.arch == "bytellm" else None,
        "n_layers": cfg.n_layers,
        "parameters": model.count_parameters(),
        "dataset_bytes": n,
        "train_bytes_seen": args.steps * args.batch_size * args.seq_len,
        "steps": args.steps,
        "train_seconds": round(train_time, 1),
        "final_train_bpb": round(history[-1]["bpb"], 4),
        "val_bpb": round(val["bpb"], 4),
        "test_bpb": round(test["bpb"], 4),
        "uniform_random_bpb": 8.0,
        "baselines_bpb": {k: round(v, 4) for k, v in baseline_bpb(test_b).items()},
        "speed": measure_speed(model, args.seq_len, device),
        "_model": model,
    }


def run_domains(model: ByteLLM, n: int, device: str) -> Dict:
    model.to(device)
    held_out = MultiModalBinaryDataset(num_samples=70, seq_len=160, seed=999)
    result: Dict = {"synthetic_heldout_bpb": round(evaluate_bits_per_byte(model, held_out, device=device)["bpb"], 4)}
    validity = {}
    for d in (BinaryDomain.WASM, BinaryDomain.MIDI, BinaryDomain.NETWORK, BinaryDomain.AUDIO,
              BinaryDomain.COMPRESSED, BinaryDomain.ASM, BinaryDomain.TEXT):
        validity[d.value] = {
            "constrained": measure_validity_rate(model, d, n=n, constrained=True, max_new_bytes=48),
            "unconstrained": measure_validity_rate(model, d, n=n, constrained=False, max_new_bytes=48),
        }
    result["validity_rate"] = validity
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bytellm.benchmark")
    ap.add_argument("--preset", choices=list(PRESETS), default="tiny")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-mb", type=float, default=2.0, help="MB of enwik8 to use (full = 100)")
    ap.add_argument("--reasoning-steps", type=int, default=3, help="1 disables recurrent pondering (ablation)")
    ap.add_argument("--arch", choices=["bytellm", "vanilla"], default="bytellm",
                    help="vanilla = standard byte transformer baseline (parameter-matched)")
    ap.add_argument("--vanilla-layers", type=int, default=0, help="override auto parameter matching")
    ap.add_argument("--patch-size", type=int, default=4, help="1 removes patch compression (ablation)")
    ap.add_argument("--no-bitplane", action="store_true", help="ablation: plain byte embedding")
    ap.add_argument("--no-local-encoder", action="store_true", help="ablation: remove multi-scale conv encoder")
    ap.add_argument("--amp", action="store_true", help="fp16 mixed precision (CUDA)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--domains", type=int, default=4, help="samples per domain for validity (0 = skip)")
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--out-dir", default="bench_out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} arch={args.arch} preset={args.preset}", flush=True)
    result = run_enwik8(args, device)
    model = result.pop("_model")
    if args.domains > 0 and args.arch == "bytellm":
        result["domains"] = run_domains(model, args.domains, device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"results_{result['tag']}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nsaved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
