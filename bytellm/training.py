"""
Production Training Pipeline, Multi-Modal Native Binary Curriculum Dataset,
and Bits-Per-Byte (BPB) Evaluation Engine for ByteLLM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bytellm.codec import (
     SPECIAL_TOKENS,
    ASMBuilder,
    BinaryDomain,
    ByteCodec,
    CompressedBuilder,
    MIDIBuilder,
    PCAPBuilder,
    WASMBuilder,
    WAVBuilder,
    compute_shannon_entropy,
)
from bytellm.model import ByteLLM, ByteLLMConfig


@dataclass
class TrainingConfig:
    """Hyperparameters for ByteLLM training and optimization."""

    batch_size: int = 8
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 100
    warmup_steps: int = 10
    grad_clip: float = 1.0
    seq_len: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    amp: bool = False  # fp16 autocast + GradScaler (CUDA only)
    log_every: int = 0  # print progress every N steps (0 = silent)

class MultiModalBinaryDataset(Dataset):
    """
    Generates a balanced curriculum of authentic binary files and abstract
    byte-reasoning tasks directly in raw octet space (0-255):
    1. WebAssembly (.wasm) modules with varied arithmetic/bitwise opcodes
    2. Native x86_64 machine code functions & ELF64 headers
    3. IPv4 + UDP network packets with exact RFC 1071 checksums
    4. GZIP / ZLIB compressed binary blocks
    5. Standard MIDI File (SMF) musical sequences (arpeggios, chords, scales)
    6. Raw RIFF/WAVE PCM audio clips
    7. Abstract Byte-Reasoning tasks with explicit `<THINK_START>...<THINK_END>`
       scratchpad traces (XOR cipher inversion, CRC/parity, byte sorting, modular arithmetic)
    """

    def __init__(
        self,
        num_samples: int = 128,
        seq_len: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        rng = np.random.default_rng(seed)

        domains = [
            BinaryDomain.WASM,
            BinaryDomain.ASM,
            BinaryDomain.NETWORK,
            BinaryDomain.COMPRESSED,
            BinaryDomain.MIDI,
            BinaryDomain.AUDIO,
            BinaryDomain.REASONING,
            BinaryDomain.TEXT,
        ]

        for i in range(num_samples):
            domain = domains[i % len(domains)]
            if domain == BinaryDomain.TEXT:
                tokens, resp_start = self._synthesize_text_sample(rng, i // len(domains))
                self.samples.append(self._prepare_tensors(tokens, b"", resp_start))
                continue
            tokens, raw_bytes = self._synthesize_sample(domain, rng, i)
            self.samples.append(self._prepare_tensors(tokens, raw_bytes))

    _TEXTS = (
        "Les octets bruts n'ont pas besoin de tokenizer.",
        "Raw bytes need no tokenizer; every language shares the same 256 symbols.",
        "原始字节不需要分词器。",
        "Los bytes crudos no necesitan tokenizador.",
        "def add(a, b):\n    return a + b\n",
        '{"format": "midi", "tempo": 120}',
    )

    def _synthesize_text_sample(
        self, rng: np.random.Generator, k: int
    ) -> Tuple[List[int], int]:
        """Plain multilingual text, or a natural-language instruction -> binary pair."""
        if k % 2 == 0:
            text = self._TEXTS[(k // 2) % len(self._TEXTS)]
            return ByteCodec.encode(text, domain=BinaryDomain.TEXT, add_eos=True), 0
        pitch = 60 + int(rng.integers(0, 12))
        if (k // 2) % 2 == 0:
            midi = MIDIBuilder.build_smf(
                [(0, pitch, 100, 240), (0, pitch + 4, 95, 240), (0, pitch + 7, 90, 240)]
            )
            instr = ("Crée un accord majeur en MIDI.", "Write a major chord as a MIDI file.")[k % 4 // 2]
            return ByteCodec.encode_instruction(instr, midi, BinaryDomain.MIDI)
        op = (WASMBuilder.OP_I32_ADD, WASMBuilder.OP_I32_MUL)[int(rng.integers(0, 2))]
        wasm = WASMBuilder.build_binary_op_module("compute", op)
        return ByteCodec.encode_instruction(
            "Génère une fonction WebAssembly à deux entiers.", wasm, BinaryDomain.WASM
        )

    def _synthesize_sample(
        self, domain: BinaryDomain, rng: np.random.Generator, idx: int
    ) -> Tuple[List[int], bytes]:
        if domain == BinaryDomain.WASM:
            opcodes = [
                WASMBuilder.OP_I32_ADD,
                WASMBuilder.OP_I32_SUB,
                WASMBuilder.OP_I32_MUL,
                WASMBuilder.OP_I32_XOR,
                WASMBuilder.OP_I32_AND,
                WASMBuilder.OP_I32_OR,
            ]
            op = opcodes[idx % len(opcodes)]
            const_val = int(rng.integers(1, 64))
            payload = WASMBuilder.build_binary_op_module(
                export_name=f"fn_{idx % 8}", opcode=op, extra_const=const_val
            )
            trace = f"wasm:op={op:#04x},c={const_val}".encode("ascii")
            tokens = ByteCodec.encode(payload, domain=domain, reasoning_trace=trace, add_eos=True)
            return tokens, payload

        if domain == BinaryDomain.ASM:
            scale = int(rng.integers(2, 16))
            bias = int(rng.integers(1, 64))
            code = ASMBuilder.build_x86_64_linear_fn(scale=scale, bias=bias)
            trace = f"x86_64:imul={scale},add={bias}".encode("ascii")
            tokens = ByteCodec.encode(code, domain=domain, reasoning_trace=trace, add_eos=True)
            return tokens, code

        if domain == BinaryDomain.NETWORK:
            msg = f"PING:{idx:04d}".encode("ascii")
            pkt = PCAPBuilder.build_ipv4_udp_packet(
                payload=msg,
                src_ip=(10, 0, int(rng.integers(0, 16)), int(rng.integers(1, 250))),
                dst_ip=(10, 0, 0, 1),
                src_port=10000 + (idx % 1000),
                dst_port=53,
            )
            trace = f"ipv4:len={len(pkt)},csum_ok".encode("ascii")
            tokens = ByteCodec.encode(pkt, domain=domain, reasoning_trace=trace, add_eos=True)
            return tokens, pkt

        if domain == BinaryDomain.COMPRESSED:
            raw_inner = bytes([int(x) for x in rng.integers(65, 91, size=24)])
            comp = (
                CompressedBuilder.build_gzip(raw_inner)
                if idx % 2 == 0
                else CompressedBuilder.build_zlib(raw_inner)
            )
            tokens = ByteCodec.encode(comp, domain=domain, add_eos=True)
            return tokens, comp

        if domain == BinaryDomain.MIDI:
            base_pitch = 60 + (idx % 12)
            chord_notes = [
                (0, base_pitch, 100, 120),
                (0, base_pitch + 4, 95, 120),
                (0, base_pitch + 7, 90, 120),
                (60, base_pitch + 12, 110, 240),
            ]
            midi_bytes = MIDIBuilder.build_smf(
                notes=chord_notes,
                ticks_per_quarter=480,
                tempo_bpm=120 + (idx % 40),
                instrument=idx % 16,
            )
            trace = f"midi:root={base_pitch},maj_triad".encode("ascii")
            tokens = ByteCodec.encode(midi_bytes, domain=domain, reasoning_trace=trace, add_eos=True)
            return tokens, midi_bytes

        if domain == BinaryDomain.AUDIO:
            f0 = 220.0 + float((idx % 12) * 35.0)
            wav_bytes = WAVBuilder.build_wav_pcm16(
                frequencies_hz=(f0, f0 * 1.5),
                sample_rate=4000,
                duration_sec=0.015,
            )
            tokens = ByteCodec.encode(wav_bytes, domain=domain, add_eos=True)
            return tokens, wav_bytes

        # BinaryDomain.REASONING: Algorithmic byte transformation with chain-of-thought
        arr = [int(x) for x in rng.integers(1, 250, size=8)]
        key = int(rng.integers(1, 255))
        sorted_arr = sorted(arr)
        xored = [(b ^ key) & 0xFF for b in sorted_arr]
        checksum = sum(xored) & 0xFF
        # Prompt: raw input bytes + key; Scratchpad: sorted intermediate bytes; Output: xored + checksum
        prompt_bytes = bytes(arr + [key])
        think_bytes = b"sort:" + bytes(sorted_arr) + f":xor={key:#04x}".encode("ascii")
        answer_bytes = bytes(xored + [checksum])
        full_payload = prompt_bytes + answer_bytes
        tokens = (
            [SPECIAL_TOKENS.BOS, SPECIAL_TOKENS.DOMAIN_REASONING]
            + list(prompt_bytes)
            + [SPECIAL_TOKENS.THINK_START]
            + list(think_bytes)
            + [SPECIAL_TOKENS.THINK_END]
            + list(answer_bytes)
            + [SPECIAL_TOKENS.EOS]
        )
        return tokens, full_payload

    def _prepare_tensors(
        self, tokens: List[int], raw_bytes: bytes, response_start: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = self.seq_len + 1
        if len(tokens) < max_len:
            padded = tokens + [SPECIAL_TOKENS.PAD] * (max_len - len(tokens))
        else:
            padded = tokens[:max_len]

        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)
        if response_start > 0:
            y[: max(0, response_start - 1)] = SPECIAL_TOKENS.PAD  # mask instruction

        # Compute target Shannon entropy per byte position
        raw_for_ent = [t if 0 <= t <= 255 else 0 for t in padded[:-1]]
        ent = compute_shannon_entropy(raw_for_ent, window_size=16)
        ent_tensor = torch.from_numpy(ent).float()
        return x, y, ent_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]


@torch.no_grad()
def evaluate_bits_per_byte(
    model: ByteLLM,
    dataset: Union[Dataset, Sequence[bytes]],
    batch_size: int = 8,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate tokenizer-free Bits-Per-Byte (BPB) and Perplexity over raw byte streams.
    BPB = CrossEntropy(nats) / ln(2). A uniform random model over 256 bytes has BPB ~ 8.0.
    """
    was_training = model.training
    model.eval()
    dev = device or str(next(model.parameters()).device)

    total_nll_nats = 0.0
    total_valid_tokens = 0
    total_ponder = 0.0
    n_batches = 0

    if isinstance(dataset, Dataset):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for x, y, _ in loader:
            x = x.to(dev)
            y = y.to(dev)
            out = model(x, targets=y)
            valid = (y != SPECIAL_TOKENS.PAD).sum().item()
            if valid > 0:
                total_nll_nats += float(out["ce_loss"].item()) * valid
                total_valid_tokens += int(valid)
                total_ponder += float(out["ponder_cost"].item())
                n_batches += 1
    else:
        for raw in dataset:
            tokens = ByteCodec.encode(raw, add_bos=True, add_eos=True)
            if len(tokens) < 2:
                continue
            t = torch.tensor([tokens], dtype=torch.long, device=dev)
            x, y = t[:, :-1], t[:, 1:]
            out = model(x, targets=y)
            valid = y.numel()
            total_nll_nats += float(out["ce_loss"].item()) * valid
            total_valid_tokens += int(valid)
            total_ponder += float(out["ponder_cost"].item())
            n_batches += 1

    if was_training:
        model.train()

    avg_nats = total_nll_nats / max(1, total_valid_tokens)
    bpb = avg_nats / math.log(2.0)
    return {
        "bpb": bpb,
        "loss_nats": avg_nats,
        "byte_perplexity": float(math.exp(min(20.0, avg_nats))),
        "mean_ponder_steps": total_ponder / max(1, n_batches),
        "evaluated_bytes": float(total_valid_tokens),
    }


class ByteLLMTrainer:
    """
    Production Trainer for ByteLLM with Cosine Warmup Schedule, Gradient Clipping,
    and Safe Checkpointing.
    """

    def __init__(
        self,
        model: ByteLLM,
        train_config: Optional[TrainingConfig] = None,
    ) -> None:
        self.model = model
        self.cfg = train_config or TrainingConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        self.step_count = 0
        self.use_amp = bool(self.cfg.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
    def _get_lr(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.learning_rate * float(step + 1) / float(max(1, self.cfg.warmup_steps))
        progress = float(step - self.cfg.warmup_steps) / float(
            max(1, self.cfg.max_steps - self.cfg.warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cfg.min_learning_rate + cosine * (
            self.cfg.learning_rate - self.cfg.min_learning_rate
        )

    def train(
        self,
        dataset: Dataset,
        num_steps: Optional[int] = None,
        sampler: Optional[torch.utils.data.Sampler] = None,
    ) -> List[Dict[str, float]]:
        """Run training loop for `num_steps` and return step-by-step telemetry."""
        steps = num_steps if num_steps is not None else self.cfg.max_steps
        self.model.train()
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=False,
        )
        history: List[Dict[str, float]] = []
        data_iter = iter(loader)

        for _ in range(steps):
            try:
                x, y, ent_target = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y, ent_target = next(data_iter)

            x = x.to(self.device)
            y = y.to(self.device)
            ent_target = ent_target.to(self.device)

            lr = self._get_lr(self.step_count)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                out = self.model(x, targets=y, entropy_targets=ent_target)
            loss = out["loss"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip).item()
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step_count += 1
            if self.cfg.log_every and self.step_count % self.cfg.log_every == 0:
                print(
                    f"step {self.step_count}/{steps} loss={loss.item():.4f} "
                    f"bpb={out['bpb'].item():.4f} lr={lr:.2e}",
                    flush=True,
                )

            history.append(
                {
                    "step": float(self.step_count),
                    "loss": float(loss.item()),
                    "bpb": float(out["bpb"].item()),
                    "ponder_cost": float(out["ponder_cost"].item()),
                    "grad_norm": grad_norm,
                    "lr": lr,
                }
            )

        return history

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        """Save model weights, configuration, and optimizer state."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.model.config.__dict__,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    @classmethod
    def load_checkpoint(
        cls, path: Union[str, Path], device: str = "cpu"
    ) -> Tuple[ByteLLM, Dict]:
        """Load a saved ByteLLM checkpoint from disk."""
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        cfg = ByteLLMConfig(**payload["config"])
        model = ByteLLM(cfg).to(device)
        model.load_state_dict(payload["model_state_dict"])
        return model, payload
