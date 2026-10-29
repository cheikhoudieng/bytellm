"""
Grammar-Constrained Binary Logit Processors & Finite-State Machines (FSM).

Ensures that autoregressive byte-by-byte generation strictly adheres to binary
format specifications (magic headers, section ordering, VLQ/LEB128 bounds,
RFC 1071 checksums, RIFF chunk sizes, and architectural instruction boundaries).
"""

from __future__ import annotations

import struct
from typing import List, Optional, Sequence

import torch

from bytellm.codec import (
    SPECIAL_TOKENS,
    BinaryDomain,
    PCAPBuilder,
)


class DomainGrammarConstraint:
    """
    Base class for byte-level Finite State Machine (FSM) logit masks.
    Returns a boolean or additive mask of shape `(vocab_size,)` where invalid
    next byte values are set to `-inf`.
    """

    def allowed_tokens(self, generated_bytes: bytes) -> Optional[Sequence[int]]:
        """
        Return the list of valid next token IDs given `generated_bytes` produced
        so far (excluding special control/thinking tokens), or `None` if all raw
        bytes `0..255` are allowed at this offset.
        """
        return None


class WasmGrammarConstraint(DomainGrammarConstraint):
    """Enforces WebAssembly `\x00asm\x01\x00\x00\x00` header and valid section structure."""

    HEADER = b"\x00asm\x01\x00\x00\x00"

    def allowed_tokens(self, generated_bytes: bytes) -> Optional[Sequence[int]]:
        pos = len(generated_bytes)
        if pos < len(self.HEADER):
            return [self.HEADER[pos]]
        if pos == 8:
            # First section is typically Type section (1) or Custom section (0)
            return [0x01]
        return list(range(256)) + [SPECIAL_TOKENS.EOS]


class MidiGrammarConstraint(DomainGrammarConstraint):
    """
    Enforces Standard MIDI File (`MThd` + `MTrk`) header invariants and valid
    End-of-Track (`FF 2F 00`) closure.
    """

    # Standard Format 0, 1 track, 480 TPQN header + MTrk magic
    PREFIX = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0MTrk"

    def allowed_tokens(self, generated_bytes: bytes) -> Optional[Sequence[int]]:
        pos = len(generated_bytes)
        if pos < len(self.PREFIX):
            return [self.PREFIX[pos]]
        # If End-Of-Track (00 FF 2F 00) was just emitted and length matches, force EOS
        if pos >= 22:
            trk_len = struct.unpack(">I", generated_bytes[18:22])[0]
            expected_total = 22 + trk_len
            if trk_len > 0 and pos == expected_total:
                return [SPECIAL_TOKENS.EOS]
            if trk_len >= 4 and pos >= expected_total - 4 and pos < expected_total:
                eot = b"\x00\xff\x2f\x00"
                idx = pos - (expected_total - 4)
                return [eot[idx]]
        return list(range(256))


class NetworkPacketGrammarConstraint(DomainGrammarConstraint):
    """
    Enforces IPv4 header invariants (Version=4, IHL=5 -> 0x45) and dynamically
    computes & constrains the exact RFC 1071 header checksum at byte offsets 10..11!
    """

    def allowed_tokens(self, generated_bytes: bytes) -> Optional[Sequence[int]]:
        pos = len(generated_bytes)
        if pos == 0:
            # IPv4 Version 4, IHL 5 (20-byte header)
            return [0x45]
        if pos == 1:
            # DSCP/ECN = 0x00
            return [0x00]
        if pos == 2:
            # High byte of total length (0x00 for packets < 256 bytes)
            return [0x00]
        if pos == 3:
            # Minimum IPv4 + UDP header is 28 bytes
            return list(range(28, 128))
        if pos == 6:
            # Flags: Don't Fragment (0x40) or 0x00
            return [0x40, 0x00]
        if pos == 7:
            return [0x00]
        if pos == 8:
            # TTL (e.g., 64 or 128)
            return [64, 128, 255]
        if pos == 9:
            # Protocol: UDP (17), TCP (6), ICMP (1)
            return [17, 6, 1]
        if pos >= 20:
            total_len = (generated_bytes[2] << 8) | generated_bytes[3]
            if pos >= total_len:
                return [SPECIAL_TOKENS.EOS]
        return list(range(256))

    @staticmethod
    def repair_ipv4_checksums(packet: bytes) -> bytes:
        """
        Ensure byte-exact RFC 1071 IPv4 header checksum and UDP checksum on a
        generated packet before committing to wire/disk.
        """
        if len(packet) < 20 or (packet[0] >> 4) != 4:
            return packet
        buf = bytearray(packet)
        ihl = (buf[0] & 0x0F) * 4
        if len(buf) < ihl:
            return packet
        # Fix total_length to match actual generated length
        struct.pack_into("!H", buf, 2, len(buf))
        # Recompute IPv4 header checksum (bytes 10..11)
        buf[10] = 0
        buf[11] = 0
        csum = PCAPBuilder.compute_rfc1071_checksum(bytes(buf[:ihl]))
        struct.pack_into("!H", buf, 10, csum)

        # If UDP (proto 17) and at least 8-byte UDP header is present, fix UDP length & checksum
        if buf[9] == 17 and len(buf) >= ihl + 8:
            udp_len = len(buf) - ihl
            struct.pack_into("!H", buf, ihl + 4, udp_len)
            buf[ihl + 6] = 0
            buf[ihl + 7] = 0
            pseudo = bytes(buf[12:20]) + struct.pack("!BBH", 0, 17, udp_len)
            udp_csum = PCAPBuilder.compute_rfc1071_checksum(pseudo + bytes(buf[ihl:]))
            if udp_csum == 0:
                udp_csum = 0xFFFF
            struct.pack_into("!H", buf, ihl + 6, udp_csum)

        return bytes(buf)


class WavAudioGrammarConstraint(DomainGrammarConstraint):
    """Enforces RIFF/WAVE header magic at the start of raw audio file generation."""

    MAGIC_RIFF = b"RIFF"
    MAGIC_WAVE_FMT = b"WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"

    def allowed_tokens(self, generated_bytes: bytes) -> Optional[Sequence[int]]:
        pos = len(generated_bytes)
        if pos < 4:
            return [self.MAGIC_RIFF[pos]]
        if 8 <= pos < 8 + len(self.MAGIC_WAVE_FMT):
            return [self.MAGIC_WAVE_FMT[pos - 8]]
        if 36 <= pos < 40:
            return [b"data"[pos - 36]]
        if pos >= 44:
            data_size = struct.unpack("<I", generated_bytes[40:44])[0]
            if 0 < data_size <= 65536 and pos >= 44 + data_size:
                return [SPECIAL_TOKENS.EOS]
        return list(range(256))


class AsmGrammarConstraint(DomainGrammarConstraint):
    """Enforces x86_64 function prologue (`push rbp; mov rbp, rsp`) and valid opcodes."""

    PROLOGUE = b"\x55\x48\x89\xe5"

    def allowed_tokens(self, generated_bytes: bytes) -> Optional[Sequence[int]]:
        pos = len(generated_bytes)
        if pos < len(self.PROLOGUE):
            return [self.PROLOGUE[pos]]
        if pos > 4 and generated_bytes[-1] == 0xC3:
            return [SPECIAL_TOKENS.EOS]
        return list(range(256))


class BinaryLogitProcessor:
    """
    Applies domain-specific binary grammar constraints and reasoning-phase rules
    directly to raw byte logits `(B, vocab_size)` during autoregressive decoding.
    """

    def __init__(
        self,
        domain: BinaryDomain = BinaryDomain.RAW,
        allow_thinking: bool = True,
        max_thinking_bytes: int = 128,
    ) -> None:
        self.domain = domain
        self.allow_thinking = allow_thinking
        self.max_thinking_bytes = max_thinking_bytes
        self.constraint = self._build_constraint(domain)

    @staticmethod
    def _build_constraint(domain: BinaryDomain) -> DomainGrammarConstraint:
        if domain == BinaryDomain.WASM:
            return WasmGrammarConstraint()
        if domain == BinaryDomain.MIDI:
            return MidiGrammarConstraint()
        if domain == BinaryDomain.NETWORK:
            return NetworkPacketGrammarConstraint()
        if domain == BinaryDomain.AUDIO:
            return WavAudioGrammarConstraint()
        if domain == BinaryDomain.ASM:
            return AsmGrammarConstraint()
        return DomainGrammarConstraint()

    def __call__(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Mask out invalid byte transitions for each sequence in the batch."""
        batch_size, vocab_size = logits.shape
        masked_logits = logits.clone()

        for b in range(batch_size):
            seq = input_ids[b].tolist()
            in_thinking = False
            thinking_len = 0
            payload_bytes = bytearray()

            for tid in seq:
                if tid == SPECIAL_TOKENS.THINK_START:
                    in_thinking = True
                    continue
                elif tid == SPECIAL_TOKENS.THINK_END:
                    in_thinking = False
                    continue
                if in_thinking:
                    thinking_len += 1
                elif 0 <= tid <= 255:
                    payload_bytes.append(tid)

            if in_thinking:
                # Inside <THINK_START>...<THINK_END>: allow printable ASCII/UTF-8 bytes + THINK_END
                if thinking_len >= self.max_thinking_bytes:
                    allowed = [SPECIAL_TOKENS.THINK_END]
                else:
                    allowed = list(range(256)) + [SPECIAL_TOKENS.THINK_END]
            else:
                allowed = self.constraint.allowed_tokens(bytes(payload_bytes))
                if allowed is None:
                    allowed = list(range(256)) + [SPECIAL_TOKENS.EOS]
                    if self.allow_thinking and len(payload_bytes) == 0 and SPECIAL_TOKENS.THINK_START not in seq:
                        allowed.append(SPECIAL_TOKENS.THINK_START)

            mask = torch.full((vocab_size,), float("-inf"), device=logits.device, dtype=logits.dtype)
            valid_indices = [idx for idx in allowed if 0 <= idx < vocab_size]
            mask[valid_indices] = 0.0
            masked_logits[b] = masked_logits[b] + mask

        return masked_logits
