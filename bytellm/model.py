"""
ByteLLM Neural Architecture:
- Bit-Plane + Nibble Structural Byte Embeddings (inductive bias for bitwise/binary formats)
- Multi-Scale Causal Local Byte Encoder (captures 16/32/64-bit endian words & opcodes)
- Dynamic Entropy-Aware Byte Patcher (BLT / MegaByte hierarchical compression)
- Global RoPE + SwiGLU Transformer Stack
- Recurrent Abstract Reasoning Core (System-2 Latent Pondering + Working Memory Registers)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bytellm.codec import (
    DOMAIN_TO_TOKEN,
    SPECIAL_TOKENS,
    BinaryDomain,
    ByteCodec,
)
from bytellm.constraints import (
    BinaryLogitProcessor,
    NetworkPacketGrammarConstraint,
)


@dataclass
class ByteLLMConfig:
    """Configuration for the Tokenizer-Free ByteLLM architecture."""

    vocab_size: int = SPECIAL_TOKENS.VOCAB_SIZE
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 688
    max_seq_len: int = 2048
    patch_size: int = 4
    entropy_threshold: float = 3.2
    reasoning_steps: int = 3
    num_memory_registers: int = 8
    dropout: float = 0.05
    rope_theta: float = 10000.0
    ponder_loss_weight: float = 0.01
    # Ablation switches (all True = the full ByteLLM architecture)
    use_bitplane: bool = True
    use_local_encoder: bool = True

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps).to(x.dtype)
        return self.weight * x_normed


class BitPlaneByteEmbedding(nn.Module):
    """
    Combines a standard lookup table over [0..vocab_size) with explicit
    8-bit binary decomposition and high/low 4-bit nibble embeddings for
    raw bytes [0..255]. Gives the model native awareness of MSB continuation
    bits (LEB128/VLQ/MIDI), bitmasks, XOR parity, and nibble alignment.
    """

    def __init__(self, vocab_size: int, d_model: int, structural: bool = True) -> None:
        super().__init__()
        self.structural_enabled = structural
        self.token_emb = nn.Embedding(vocab_size, d_model)
        if not structural:
            return
        self.bit_proj = nn.Linear(8, d_model, bias=False)
        self.high_nibble_emb = nn.Embedding(16, d_model // 2)
        self.low_nibble_emb = nn.Embedding(16, d_model - (d_model // 2))
        self.gate = nn.Parameter(torch.tensor(0.25))

        # Precompute bit-planes for 0..255 -> shape (vocab_size, 8)
        bit_table = torch.zeros(vocab_size, 8, dtype=torch.float32)
        for b in range(min(256, vocab_size)):
            for bit in range(8):
                bit_table[b, bit] = 1.0 if ((b >> bit) & 1) else -1.0
        self.register_buffer("bit_table", bit_table, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.token_emb(input_ids)
        if not self.structural_enabled:
            return base
        is_raw_byte = (input_ids < 256).unsqueeze(-1).to(base.dtype)
        clamped_bytes = input_ids.clamp(0, 255)

        bits = self.bit_table[input_ids].to(base.dtype)
        bit_feat = self.bit_proj(bits)

        hi = (clamped_bytes >> 4) & 0x0F
        lo = clamped_bytes & 0x0F
        nibble_feat = torch.cat([self.high_nibble_emb(hi), self.low_nibble_emb(lo)], dim=-1)

        structural = (bit_feat + nibble_feat) * is_raw_byte
        return base + torch.tanh(self.gate) * structural


class CausalDepthwiseConv1d(nn.Module):
    """Causal 1D depthwise-separable convolution for local byte n-gram context."""

    def __init__(self, d_model: int, kernel_size: int) -> None:
        super().__init__()
        self.pad = kernel_size - 1
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, groups=d_model, bias=False
        )
        self.pw = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        xt = x.transpose(1, 2)
        xt = F.pad(xt, (self.pad, 0))
        out = self.dw(xt).transpose(1, 2)
        return self.pw(F.silu(out))


class LocalByteEncoder(nn.Module):
    """
    Multi-scale causal byte encoder capturing 2-byte, 4-byte, and 8-byte binary words
    (e.g., uint16, uint32, uint64, magic signatures, multi-byte opcodes).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.conv3 = CausalDepthwiseConv1d(d_model, kernel_size=3)
        self.conv5 = CausalDepthwiseConv1d(d_model, kernel_size=5)
        self.conv9 = CausalDepthwiseConv1d(d_model, kernel_size=9)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        ms = (self.conv3(h) + self.conv5(h) + self.conv9(h)) / 3.0
        return x + torch.tanh(self.gate) * self.out_proj(ms)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) for relative byte/patch distance."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len * 2)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors (B, H, T, Dh)."""

    def rotate_half(t: torch.Tensor) -> torch.Tensor:
        d = t.shape[-1] // 2
        return torch.cat((-t[..., d:], t[..., :d]), dim=-1)

    cos_u = cos.unsqueeze(0).unsqueeze(0)
    sin_u = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos_u) + (rotate_half(q) * sin_u)
    k_rot = (k * cos_u) + (rotate_half(k) * sin_u)
    return q_rot, k_rot


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with Flash SDPA and RoPE."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer Block with RoPE Causal Attention and SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class DynamicBytePatcher(nn.Module):
    """
    Entropy-Aware Hierarchical Byte Patcher (Causal Multi-Scale Branch).
    Instead of discarding fine-grained byte states by forcing all Transformer
    layers through a shifted coarse bottleneck, this module builds a strictly
    causal patch-level summary branch and fuses it back into the full-resolution
    byte stream via a learned residual gate.
    """

    def __init__(self, d_model: int, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = max(1, patch_size)
        self.entropy_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, 1),
        )
        if self.patch_size > 1:
            self.norm = RMSNorm(d_model)
            self.down_proj = nn.Linear(d_model * self.patch_size, d_model, bias=False)
            self.up_proj = nn.Linear(d_model, d_model * self.patch_size, bias=False)
            self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self, byte_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute causal entropy predictions and fuse hierarchical patch summaries
        into `byte_states` `(B, T, D)` without losing 1-byte resolution.
        """
        B, T, D = byte_states.shape
        P_size = self.patch_size
        # Detach input for auxiliary entropy estimation so it never distorts LM features
        entropy_pred = 8.0 * torch.sigmoid(self.entropy_head(byte_states.detach()).squeeze(-1))

        if P_size <= 1:
            return byte_states, entropy_pred, T

        h = self.norm(byte_states)
        pad_len = (P_size - (T % P_size)) % P_size
        if pad_len > 0:
            padded = F.pad(h, (0, 0, 0, pad_len))
            ent_padded = F.pad(entropy_pred, (0, pad_len))
        else:
            padded = h
            ent_padded = entropy_pred

        num_patches = padded.shape[1] // P_size
        grouped = padded.view(B, num_patches, P_size, D)
        ent_weights = F.softmax(ent_padded.view(B, num_patches, P_size, 1), dim=2)
        weighted = grouped * (1.0 + ent_weights)
        patches = F.silu(self.down_proj(weighted.reshape(B, num_patches, P_size * D)))

        # Shift patch summaries right by 1 patch so patch p only conditions patches >= p+1
        zero_patch = torch.zeros(B, 1, D, device=patches.device, dtype=patches.dtype)
        shifted_patches = torch.cat([zero_patch, patches[:, :-1, :]], dim=1)
        unpooled = self.up_proj(shifted_patches).view(B, num_patches * P_size, D)[:, :T, :]

        enriched = byte_states + torch.tanh(self.res_scale) * unpooled
        return enriched, entropy_pred, T

    def unpool_causal(
        self,
        byte_states: torch.Tensor,
        patch_states: torch.Tensor,
        orig_len: int,
    ) -> torch.Tensor:
        """Identity pass-through since patch_states are already at full causal byte resolution."""
        return patch_states


class AbstractReasoningCore(nn.Module):
    """
    System-2 Recurrent Abstract Reasoning Core with Working-Memory Scratchpad Registers
    and Adaptive Computation Time (ACT) Halting Gate.

    Operates at full causal byte resolution with clean pre-norm residual connections:
    1. Queries learnable Working-Memory Registers via a zero-initialized residual gate.
    2. Refines representations over `max_steps` recurrent causal Transformer steps
       without polluting the residual stream with step embeddings.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_steps: int = 3,
        num_registers: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_steps = max(1, max_steps)
        self.num_registers = num_registers

        self.memory_registers = nn.Parameter(torch.randn(1, num_registers, d_model) * 0.02)
        self.step_emb = nn.Embedding(self.max_steps, d_model)

        self.mem_norm = RMSNorm(d_model)
        self.mem_q = nn.Linear(d_model, d_model, bias=False)
        self.mem_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.mem_out = nn.Linear(d_model, d_model, bias=False)
        self.mem_gate = nn.Parameter(torch.zeros(1))

        self.reason_block = TransformerBlock(d_model, n_heads, d_ff, dropout)
        self.step_scale = nn.Parameter(torch.ones(self.max_steps) / float(self.max_steps))
        self.halt_gate = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        state = x
        accumulated_delta = torch.zeros_like(x)
        remainders = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)
        n_updates = torch.zeros(B, T, 1, device=x.device, dtype=x.dtype)

        mem = self.memory_registers.expand(B, -1, -1).to(x.dtype)
        kv = self.mem_kv(mem).view(B, self.num_registers, 2, D)
        mem_k, mem_v = kv.unbind(dim=2)
        g_mem = torch.tanh(self.mem_gate)

        for step in range(self.max_steps):
            step_vec = self.step_emb(
                torch.tensor(step, device=x.device)
            ).view(1, 1, D).to(x.dtype)

            # Query working-memory registers using step-conditioned normalized state
            q = self.mem_q(self.mem_norm(state + step_vec))
            attn_scores = torch.matmul(q, mem_k.transpose(-1, -2)) / math.sqrt(D)
            attn_probs = F.softmax(attn_scores, dim=-1)
            mem_ctx = self.mem_out(torch.matmul(attn_probs, mem_v))
            state_with_mem = state + g_mem * mem_ctx

            # Recurrent causal reasoning block (clean residual inside reason_block)
            next_state = self.reason_block(state_with_mem, cos, sin)
            delta = next_state - state
            state = state + self.step_scale[step] * delta

            if step == self.max_steps - 1:
                p_halt = remainders
            else:
                raw_p = torch.sigmoid(self.halt_gate(state.detach()))
                p_halt = torch.minimum(raw_p, remainders)

            accumulated_delta = accumulated_delta + p_halt * (state - x)
            remainders = (remainders - p_halt).clamp(min=0.0)
            n_updates = n_updates + (remainders > 1e-4).to(x.dtype)

        ponder_cost = (n_updates + remainders).mean()
        step_depths = (n_updates + remainders).squeeze(-1)
        return x + accumulated_delta, ponder_cost, step_depths


class ByteLLM(nn.Module):
    """
    Production-Ready Tokenizer-Free Byte-LLM (0-255).

    Combines:
    1. Bit-plane + nibble structural byte embeddings
    2. Multi-scale causal local byte encoder
    3. Entropy-aware dynamic latent patching
    4. Recurrent System-2 Abstract Reasoning Core (working memory + adaptive halting)
    5. Fine-grained causal byte decoder head
    """

    def __init__(self, config: Optional[ByteLLMConfig] = None) -> None:
        super().__init__()
        self.config = config or ByteLLMConfig()
        c = self.config

        self.byte_emb = BitPlaneByteEmbedding(c.vocab_size, c.d_model, structural=c.use_bitplane)
        self.local_encoder = LocalByteEncoder(c.d_model) if c.use_local_encoder else nn.Identity()
        self.patcher = DynamicBytePatcher(c.d_model, patch_size=c.patch_size)

        head_dim = c.d_model // c.n_heads
        self.patch_rope = RotaryEmbedding(head_dim, max_seq_len=c.max_seq_len, theta=c.rope_theta)
        self.byte_rope = RotaryEmbedding(head_dim, max_seq_len=c.max_seq_len, theta=c.rope_theta)

        n_latent_layers = max(1, c.n_layers - 1)
        self.latent_layers = nn.ModuleList(
            [TransformerBlock(c.d_model, c.n_heads, c.d_ff, c.dropout) for _ in range(n_latent_layers)]
        )
        self.reasoning_core = AbstractReasoningCore(
            d_model=c.d_model,
            n_heads=c.n_heads,
            d_ff=c.d_ff,
            max_steps=c.reasoning_steps,
            num_registers=c.num_memory_registers,
            dropout=c.dropout,
        )
        self.byte_decoder_block = TransformerBlock(c.d_model, c.n_heads, c.d_ff, c.dropout)
        self.final_norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

        # Tie base token embedding weights with output projection for parameter efficiency
        self.lm_head.weight = self.byte_emb.token_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        entropy_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass over raw byte token IDs `(B, T)`.
        Returns dictionary containing `logits`, `loss` (if targets provided),
        `bpb` (bits-per-byte), `ponder_cost`, `reasoning_depth`, and `entropy_pred`.
        """
        B, T = input_ids.shape
        x = self.byte_emb(input_ids)
        byte_local = self.local_encoder(x)

        # Dynamic hierarchical patch compression
        patches, entropy_pred, orig_len = self.patcher(byte_local)
        P = patches.shape[1]
        p_cos, p_sin = self.patch_rope(patches, P)

        for layer in self.latent_layers:
            patches = layer(patches, p_cos, p_sin)

        # System-2 Abstract Reasoning Core over latent patches
        patches_reasoned, ponder_cost, patch_depths = self.reasoning_core(patches, p_cos, p_sin)

        # Unpool causal patch context back to byte resolution
        byte_fused = self.patcher.unpool_causal(byte_local, patches_reasoned, orig_len)
        b_cos, b_sin = self.byte_rope(byte_fused, T)
        byte_decoded = self.byte_decoder_block(byte_fused, b_cos, b_sin)

        logits = self.lm_head(self.final_norm(byte_decoded))

        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "ponder_cost": ponder_cost,
            "reasoning_depth": patch_depths,
            "entropy_pred": entropy_pred,
        }

        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=SPECIAL_TOKENS.PAD,
            )
            bpb = ce_loss / math.log(2.0)
            total_loss = ce_loss + self.config.ponder_loss_weight * ponder_cost
            if entropy_targets is not None and entropy_targets.numel() == entropy_pred.numel():
                ent_loss = F.mse_loss(entropy_pred, entropy_targets.to(entropy_pred.dtype))
                total_loss = total_loss + 0.05 * ent_loss
            result["loss"] = total_loss
            result["ce_loss"] = ce_loss
            result["bpb"] = bpb

        return result

    @torch.no_grad()
    def generate_bytes(
        self,
        prompt: bytes = b"",
        domain: BinaryDomain = BinaryDomain.RAW,
        max_new_bytes: int = 128,
        temperature: float = 0.8,
        top_k: int = 32,
        top_p: float = 0.95,
        use_grammar_constraints: bool = True,
        allow_thinking: bool = False,
        instruction: Optional[str] = None,
        repetition_penalty: float = 1.0,
    ) -> Tuple[bytes, bytes, Dict[str, float]]:
        """
        Autoregressively generate raw binary payloads byte-by-byte without BPE.

        Returns:
            payload_bytes: Pure raw byte sequence (0x00..0xFF)
            reasoning_bytes: Extracted `<THINK_START>...<THINK_END>` scratchpad trace
            metrics: Dict of generation telemetry (mean_reasoning_depth, entropy, etc.)
        """
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device

        if instruction is None:
            prefix_tokens = ByteCodec.encode(b"", domain=domain, add_bos=True, add_eos=False)
        else:
            prefix_tokens, _ = ByteCodec.encode_instruction(
                instruction, b"", domain=domain, add_eos=False
            )
        prefix_len = len(prefix_tokens)
        init_tokens = prefix_tokens + list(prompt)
        seq = torch.tensor([init_tokens], dtype=torch.long, device=device)

        processor = (
            BinaryLogitProcessor(domain=domain, allow_thinking=allow_thinking)
            if use_grammar_constraints
            else None
        )

        depths: List[float] = []
        entropies: List[float] = []

        for _ in range(max_new_bytes):
            if seq.shape[1] >= self.config.max_seq_len:
                seq = seq[:, -self.config.max_seq_len + 1 :]

            out = self.forward(seq)
            next_logits = out["logits"][:, -1, :]
            depths.append(float(out["reasoning_depth"].mean().item()))
            entropies.append(float(out["entropy_pred"][:, -1].mean().item()))

            if processor is not None:
                # Pass only tokens generated after the domain prefix to the FSM
                gen_view = seq[:, prefix_len:]
                next_logits = processor(gen_view, next_logits)

            if repetition_penalty != 1.0 and seq.shape[1] > prefix_len:
                gen_tokens = seq[0, prefix_len:]
                counts = torch.bincount(gen_tokens, minlength=self.config.vocab_size)
                penalized = counts > 0
                next_logits[0, penalized] = torch.where(
                    next_logits[0, penalized] > 0,
                    next_logits[0, penalized] / repetition_penalty,
                    next_logits[0, penalized] * repetition_penalty,
                )

            if temperature <= 1e-5:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / temperature
                if top_k > 0:
                    k = min(top_k, scaled.shape[-1])
                    v, _ = torch.topk(scaled, k)
                    scaled = torch.where(
                        scaled < v[:, [-1]],
                        torch.full_like(scaled, float("-inf")),
                        scaled,
                    )
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum_probs = torch.cumsum(probs, dim=-1)
                    remove_mask = cum_probs > top_p
                    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                    remove_mask[..., 0] = 0
                    indices_to_remove = remove_mask.scatter(1, sorted_indices, remove_mask)
                    scaled = scaled.masked_fill(indices_to_remove, float("-inf"))

                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tid = int(next_token.item())
            seq = torch.cat([seq, next_token], dim=1)
            if tid == SPECIAL_TOKENS.EOS:
                break

        if was_training:
            self.train()

        # Extract generated reasoning trace and payload (including prompt bytes)
        all_tokens = seq[0].tolist()
        thinking_bytes, payload_bytes = ByteCodec.extract_reasoning_and_payload(all_tokens[prefix_len:])

        if use_grammar_constraints and domain == BinaryDomain.NETWORK and len(payload_bytes) >= 20:
            payload_bytes = NetworkPacketGrammarConstraint.repair_ipv4_checksums(payload_bytes)

        metrics = {
            "mean_reasoning_depth": float(sum(depths) / max(1, len(depths))),
            "mean_predicted_entropy": float(sum(entropies) / max(1, len(entropies))),
            "generated_bytes": float(len(payload_bytes)),
            "thinking_bytes": float(len(thinking_bytes)),
        }
        return payload_bytes, thinking_bytes, metrics

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VanillaByteTransformer(nn.Module):
    """
    Standard decoder-only byte transformer used as the reference baseline:
    plain byte embedding -> N identical RoPE/SwiGLU causal blocks at full byte
    resolution -> tied LM head. No bit-planes, no local conv encoder, no patching,
    no reasoning core. Uses the very same `TransformerBlock`, optimizer, data and
    init as `ByteLLM`, so any BPB difference comes from the architecture alone.
    """

    def __init__(self, config: Optional[ByteLLMConfig] = None) -> None:
        super().__init__()
        self.config = config or ByteLLMConfig()
        c = self.config
        self.token_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.rope = RotaryEmbedding(c.d_model // c.n_heads, max_seq_len=c.max_seq_len, theta=c.rope_theta)
        self.blocks = nn.ModuleList(
            [TransformerBlock(c.d_model, c.n_heads, c.d_ff, c.dropout) for _ in range(c.n_layers)]
        )
        self.final_norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(ByteLLM._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        entropy_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.token_emb(input_ids)
        cos, sin = self.rope(x, input_ids.shape[1])
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.final_norm(x))
        one = torch.ones((), device=logits.device)
        result: Dict[str, torch.Tensor] = {"logits": logits, "ponder_cost": one}
        if targets is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=SPECIAL_TOKENS.PAD,
            )
            result.update(loss=ce, ce_loss=ce, bpb=ce / math.log(2.0))
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
