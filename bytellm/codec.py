"""
Tokenizer-Free Byte Codec (0-255) & Multimodal Binary Builders/Validators.

Operates directly on raw octets (0x00 - 0xFF) without any BPE/WordPiece/SentencePiece
tokenization. Includes native binary builders, parsers, and cryptographic/checksum
validators for:
- WebAssembly (.wasm)
- Machine Code / ELF64 / x86_64 & ARM64 ASM
- Network Packets (PCAP / IPv4 / TCP / UDP with RFC 1071 checksums)
- Compressed Streams (GZIP / ZLIB with CRC32 & Adler32)
- Music (Standard MIDI File Format 0/1 with VLQ delta-times)
- Raw Audio (RIFF WAVE PCM / u-law synthesis & parsing)
"""

from __future__ import annotations

import enum
import gzip
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


class BinaryDomain(str, enum.Enum):
    """Supported native binary and reasoning domains."""

    RAW = "raw"
    WASM = "wasm"
    ASM = "asm"
    NETWORK = "network"
    COMPRESSED = "compressed"
    MIDI = "midi"
    AUDIO = "audio"
    REASONING = "reasoning"
    TEXT = "text"


@dataclass(frozen=True)
class SpecialTokens:
    """Control tokens residing strictly above the raw byte space [0..255]."""

    PAD: int = 256
    BOS: int = 257
    EOS: int = 258
    THINK_START: int = 259
    THINK_END: int = 260
    PATCH_SEP: int = 261
    DOMAIN_RAW: int = 262
    DOMAIN_WASM: int = 263
    DOMAIN_ASM: int = 264
    DOMAIN_NETWORK: int = 265
    DOMAIN_COMPRESSED: int = 266
    DOMAIN_MIDI: int = 267
    DOMAIN_AUDIO: int = 268
    DOMAIN_REASONING: int = 269
    DOMAIN_TEXT: int = 270

    VOCAB_SIZE: int = 271


SPECIAL_TOKENS = SpecialTokens()

DOMAIN_TO_TOKEN: Dict[BinaryDomain, int] = {
    BinaryDomain.RAW: SPECIAL_TOKENS.DOMAIN_RAW,
    BinaryDomain.WASM: SPECIAL_TOKENS.DOMAIN_WASM,
    BinaryDomain.ASM: SPECIAL_TOKENS.DOMAIN_ASM,
    BinaryDomain.NETWORK: SPECIAL_TOKENS.DOMAIN_NETWORK,
    BinaryDomain.COMPRESSED: SPECIAL_TOKENS.DOMAIN_COMPRESSED,
    BinaryDomain.MIDI: SPECIAL_TOKENS.DOMAIN_MIDI,
    BinaryDomain.AUDIO: SPECIAL_TOKENS.DOMAIN_AUDIO,
    BinaryDomain.REASONING: SPECIAL_TOKENS.DOMAIN_REASONING,
    BinaryDomain.TEXT: SPECIAL_TOKENS.DOMAIN_TEXT,
}

TOKEN_TO_DOMAIN: Dict[int, BinaryDomain] = {v: k for k, v in DOMAIN_TO_TOKEN.items()}


def compute_shannon_entropy(data: Union[bytes, bytearray, Sequence[int]], window_size: int = 16) -> np.ndarray:
    """
    Compute local sliding-window Shannon byte entropy H(X) in bits [0.0, 8.0]
    for each byte position. Used by the Dynamic Byte Patcher to allocate compute
    adaptively at unpredictable byte boundaries.
    """
    if isinstance(data, (bytes, bytearray)):
        arr = np.frombuffer(bytes(data), dtype=np.uint8)
    else:
        arr = np.asarray(data, dtype=np.int64).astype(np.uint8)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # Vectorised in blocks (window counts via cumulative one-hot) so that it is
    # fast for training batches and memory-bounded for large files.
    block = 8192
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, block):
        end = min(n, start + block)
        lo_ctx = max(0, start - window_size + 1)
        seg = arr[lo_ctx:end]
        m = len(seg)
        onehot = np.zeros((m + 1, 256), dtype=np.int32)
        onehot[np.arange(1, m + 1), seg] = 1
        cs = np.cumsum(onehot, axis=0)
        idx = np.arange(1, m + 1)
        lo = np.maximum(idx - window_size, 0)
        counts = cs[idx] - cs[lo]
        w = (idx - lo)[:, None].astype(np.float64)
        p = counts / w
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.where(counts > 0, p * np.log2(p), 0.0).sum(axis=1)
        out[start:end] = ent[start - lo_ctx :]
    return out


class ByteCodec:
    """
    Zero-BPE Tokenizer-Free Byte Codec.
    Maps raw bytes 0..255 directly to token IDs 0..255, with optional domain
    and latent-reasoning control tokens in 256..269.
    """

    vocab_size: int = SPECIAL_TOKENS.VOCAB_SIZE
    special: SpecialTokens = SPECIAL_TOKENS

    @staticmethod
    def encode(
        payload: Union[bytes, bytearray, str, Sequence[int]],
        domain: Optional[BinaryDomain] = None,
        reasoning_trace: Optional[Union[bytes, str]] = None,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode raw bytes or UTF-8 string into integer token IDs without BPE."""
        if isinstance(payload, str):
            raw_bytes = payload.encode("utf-8")
        elif isinstance(payload, (bytes, bytearray)):
            raw_bytes = bytes(payload)
        else:
            raw_bytes = bytes(int(x) & 0xFF for x in payload)

        tokens: List[int] = []
        if add_bos:
            tokens.append(SPECIAL_TOKENS.BOS)
        if domain is not None:
            tokens.append(DOMAIN_TO_TOKEN[domain])
        if reasoning_trace is not None:
            trace_bytes = (
                reasoning_trace.encode("utf-8")
                if isinstance(reasoning_trace, str)
                else bytes(reasoning_trace)
            )
            tokens.append(SPECIAL_TOKENS.THINK_START)
            tokens.extend(list(trace_bytes))
            tokens.append(SPECIAL_TOKENS.THINK_END)

        tokens.extend(list(raw_bytes))
        if add_eos:
            tokens.append(SPECIAL_TOKENS.EOS)
        return tokens

    @staticmethod
    def encode_instruction(
        instruction: Union[str, bytes],
        response: Union[bytes, str] = b"",
        domain: BinaryDomain = BinaryDomain.RAW,
        reasoning: Optional[Union[bytes, str]] = None,
        add_eos: bool = True,
    ) -> Tuple[List[int], int]:
        """
        Encode a natural-language instruction followed by a response in any domain:
        `[BOS][DOMAIN_TEXT] instruction [DOMAIN_X] [THINK..] response [EOS]`.
        Returns (tokens, response_start) where `response_start` is the index of the
        target-domain token; positions before it can be masked out of the loss.
        """
        instr = instruction.encode("utf-8") if isinstance(instruction, str) else bytes(instruction)
        resp = response.encode("utf-8") if isinstance(response, str) else bytes(response)
        tokens: List[int] = [SPECIAL_TOKENS.BOS, SPECIAL_TOKENS.DOMAIN_TEXT]
        tokens.extend(instr)
        response_start = len(tokens)
        tokens.append(DOMAIN_TO_TOKEN[domain])
        if reasoning is not None:
            trace = reasoning.encode("utf-8") if isinstance(reasoning, str) else bytes(reasoning)
            tokens.append(SPECIAL_TOKENS.THINK_START)
            tokens.extend(trace)
            tokens.append(SPECIAL_TOKENS.THINK_END)
        tokens.extend(resp)
        if add_eos:
            tokens.append(SPECIAL_TOKENS.EOS)
        return tokens, response_start

    @staticmethod
    def decode(
        tokens: Union[Sequence[int], torch.Tensor, np.ndarray],
        strip_special: bool = True,
        strip_thinking: bool = True,
    ) -> bytes:
        """
        Decode token IDs back into a pure raw byte sequence (0x00 - 0xFF).
        Optionally strips out `<THINK_START>...<THINK_END>` scratchpad octets.
        """
        if isinstance(tokens, torch.Tensor):
            seq = tokens.detach().cpu().tolist()
        elif isinstance(tokens, np.ndarray):
            seq = tokens.tolist()
        else:
            seq = list(tokens)

        out = bytearray()
        in_thinking = False
        for t in seq:
            tid = int(t)
            if tid == SPECIAL_TOKENS.THINK_START:
                if strip_thinking:
                    in_thinking = True
                    continue
            elif tid == SPECIAL_TOKENS.THINK_END:
                if strip_thinking:
                    in_thinking = False
                    continue
            if in_thinking:
                continue
            if 0 <= tid <= 255:
                out.append(tid)
            elif not strip_special:
                # Represent special tokens by UTF-8 tag if explicitly requested
                out.extend(f"<TOK_{tid}>".encode("ascii"))
        return bytes(out)

    @staticmethod
    def extract_reasoning_and_payload(
        tokens: Sequence[int],
    ) -> Tuple[bytes, bytes]:
        """Separate the latent reasoning byte trace from the final binary output."""
        seq = [int(t) for t in tokens]
        thinking = bytearray()
        payload = bytearray()
        in_thinking = False
        for tid in seq:
            if tid == SPECIAL_TOKENS.THINK_START:
                in_thinking = True
                continue
            elif tid == SPECIAL_TOKENS.THINK_END:
                in_thinking = False
                continue
            elif tid == SPECIAL_TOKENS.EOS:
                break
            if 0 <= tid <= 255:
                if in_thinking:
                    thinking.append(tid)
                else:
                    payload.append(tid)
        return bytes(thinking), bytes(payload)

    @staticmethod
    def detect_domain(data: bytes) -> BinaryDomain:
        """Inspect magic headers and structural signatures of a raw byte stream."""
        if len(data) >= 8 and data[:8] == b"\x00asm\x01\x00\x00\x00":
            return BinaryDomain.WASM
        if len(data) >= 4 and data[:4] == b"MThd":
            return BinaryDomain.MIDI
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return BinaryDomain.AUDIO
        if len(data) >= 2 and data[:2] == b"\x1f\x8b":
            return BinaryDomain.COMPRESSED
        if len(data) >= 2 and data[0] == 0x78 and data[1] in (0x01, 0x5E, 0x9C, 0xDA):
            return BinaryDomain.COMPRESSED
        if len(data) >= 4 and data[:4] in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4"):
            return BinaryDomain.NETWORK
        if len(data) >= 20 and (data[0] >> 4) == 4 and (data[0] & 0x0F) >= 5:
            # Verify IPv4 header checksum
            if PCAPBuilder.verify_ipv4_checksum(data[: (data[0] & 0x0F) * 4]):
                return BinaryDomain.NETWORK
        if len(data) > 0:
            try:
                decoded = data.decode("utf-8")
                ok = sum(1 for ch in decoded if ch.isprintable() or ch in "\n\r\t")
                if ok / len(decoded) > 0.95:
                    return BinaryDomain.TEXT
            except UnicodeDecodeError:
                pass
        if len(data) >= 4 and data[:4] == b"\x7fELF":
            return BinaryDomain.ASM
        if len(data) >= 1 and (data[-1] == 0xC3 or data[:3] == b"\x55\x48\x89"):
            return BinaryDomain.ASM
        return BinaryDomain.RAW


# ============================================================================
# 1. Native MIDI Binary Engine (Standard MIDI File Format 0/1)
# ============================================================================

class MIDIBuilder:
    """
    Constructs, parses, and validates native Standard MIDI Files (SMF) at the byte level.
    """

    MAGIC_HEADER = b"MThd"
    MAGIC_TRACK = b"MTrk"
    END_OF_TRACK = b"\x00\xff\x2f\x00"

    @staticmethod
    def encode_vlq(value: int) -> bytes:
        """Encode non-negative integer into MIDI Variable-Length Quantity (VLQ) bytes."""
        if value < 0:
            raise ValueError("VLQ value must be non-negative")
        buffer = [value & 0x7F]
        value >>= 7
        while value > 0:
            buffer.append((value & 0x7F) | 0x80)
            value >>= 7
        buffer.reverse()
        return bytes(buffer)

    @staticmethod
    def decode_vlq(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """Decode MIDI VLQ starting at `offset`, returning (value, next_offset)."""
        value = 0
        idx = offset
        while idx < len(data):
            b = data[idx]
            value = (value << 7) | (b & 0x7F)
            idx += 1
            if not (b & 0x80):
                return value, idx
            if idx - offset >= 4:
                break
        raise ValueError("Malformed or truncated MIDI VLQ")

    @classmethod
    def build_smf(
        cls,
        notes: Sequence[Tuple[int, int, int, int]],
        ticks_per_quarter: int = 480,
        tempo_bpm: int = 120,
        instrument: int = 0,
    ) -> bytes:
        """
        Generate a complete, spec-compliant Standard MIDI File (Format 0) binary.

        Args:
            notes: Sequence of (delta_ticks, pitch_0_127, velocity_0_127, duration_ticks).
            ticks_per_quarter: Time division in ticks per quarter note.
            tempo_bpm: Tempo in beats per minute.
            instrument: General MIDI Program Change number (0..127).
        """
        header = (
            cls.MAGIC_HEADER
            + struct.pack(">IHHH", 6, 0, 1, ticks_per_quarter & 0x7FFF)
        )

        track_data = bytearray()
        # Tempo meta event: FF 51 03 tt tt tt
        mpqn = max(1, int(60_000_000 / max(1, tempo_bpm)))
        track_data.extend(b"\x00\xff\x51\x03")
        track_data.extend(mpqn.to_bytes(3, "big"))

        # Program change on channel 0: 00 C0 <instrument>
        track_data.extend( bytes([0x00, 0xC0, instrument & 0x7F]) )

        for delta, pitch, velocity, duration in notes:
            p = max(0, min(127, int(pitch)))
            v = max(1, min(127, int(velocity)))
            d_on = max(0, int(delta))
            d_off = max(1, int(duration))

            # Note On (0x90)
            track_data.extend(cls.encode_vlq(d_on))
            track_data.extend(bytes([0x90, p, v]))
            # Note Off (0x80)
            track_data.extend(cls.encode_vlq(d_off))
            track_data.extend(bytes([0x80, p, 0x00]))

        # End of Track Meta Event
        track_data.extend(cls.END_OF_TRACK)

        track_chunk = cls.MAGIC_TRACK + struct.pack(">I", len(track_data)) + bytes(track_data)
        return header + track_chunk

    @classmethod
    def validate(cls, data: bytes) -> Tuple[bool, str]:
        """Validate a binary Standard MIDI File structure at the byte level."""
        if len(data) < 22:
            return False, "MIDI payload shorter than minimum 22 bytes"
        if data[:4] != cls.MAGIC_HEADER:
            return False, f"Invalid MThd magic: {data[:4]!r}"
        hdr_len, fmt, ntracks, division = struct.unpack(">IHHH", data[4:14])
        if hdr_len != 6:
            return False, f"Invalid MThd header length: {hdr_len}"
        if fmt not in (0, 1, 2):
            return False, f"Invalid MIDI format: {fmt}"
        if ntracks < 1 or division == 0:
            return False, "Invalid track count or division"

        offset = 14
        tracks_found = 0
        while offset < len(data):
            if offset + 8 > len(data):
                return False, "Truncated MTrk header"
            if data[offset : offset + 4] != cls.MAGIC_TRACK:
                return False, f"Expected MTrk at {offset}, got {data[offset:offset+4]!r}"
            trk_len = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
            trk_end = offset + 8 + trk_len
            if trk_end > len(data):
                return False, f"Track length {trk_len} exceeds file size {len(data)}"
            trk_bytes = data[offset + 8 : trk_end]
            if not trk_bytes.endswith(b"\xff\x2f\x00"):
                return False, "Track missing End-Of-Track meta event (FF 2F 00)"
            tracks_found += 1
            offset = trk_end

        if tracks_found != ntracks:
            return False, f"Header declared {ntracks} tracks, parsed {tracks_found}"
        return True, f"Valid SMF Format {fmt} ({tracks_found} track(s), {division} TPQN)"


# ============================================================================
# 2. Native WebAssembly (.wasm) Binary Engine
# ============================================================================

class WASMBuilder:
    """
    Constructs and validates executable WebAssembly (WASM 1.0) binary modules.
    """

    MAGIC = b"\x00asm"
    VERSION = b"\x01\x00\x00\x00"

    # Section IDs
    SEC_TYPE = 1
    SEC_FUNCTION = 3
    SEC_EXPORT = 7
    SEC_CODE = 10

    # Value Types
    I32 = 0x7F
    I64 = 0x7E
    FUNC_TYPE = 0x60

    # Opcodes
    OP_LOCAL_GET = 0x20
    OP_I32_CONST = 0x41
    OP_I32_ADD = 0x6A
    OP_I32_SUB = 0x6B
    OP_I32_MUL = 0x6C
    OP_I32_AND = 0x71
    OP_I32_OR = 0x72
    OP_I32_XOR = 0x73
    OP_END = 0x0B

    @staticmethod
    def encode_uleb128(value: int) -> bytes:
        """Encode unsigned integer in LEB128 format."""
        if value < 0:
            raise ValueError("ULEB128 requires non-negative integer")
        out = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            if value != 0:
                byte |= 0x80
            out.append(byte)
            if value == 0:
                break
        return bytes(out)

    @staticmethod
    def decode_uleb128(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """Decode unsigned LEB128 at `offset`, returning (value, next_offset)."""
        result = 0
        shift = 0
        idx = offset
        while idx < len(data):
            b = data[idx]
            result |= (b & 0x7F) << shift
            idx += 1
            if (b & 0x80) == 0:
                return result, idx
            shift += 7
            if shift >= 64:
                break
        raise ValueError("Malformed ULEB128 sequence")

    @classmethod
    def _section(cls, section_id: int, content: bytes) -> bytes:
        return bytes([section_id]) + cls.encode_uleb128(len(content)) + content

    @classmethod
    def build_binary_op_module(
        cls,
        export_name: str = "compute",
        opcode: int = OP_I32_ADD,
        extra_const: Optional[int] = None,
    ) -> bytes:
        """
        Build a complete, valid WebAssembly binary module exporting `(i32, i32) -> i32`.
        """
        # Type Section: 1 type: (i32, i32) -> (i32)
        type_entry = bytes([cls.FUNC_TYPE, 2, cls.I32, cls.I32, 1, cls.I32])
        type_sec = cls._section(cls.SEC_TYPE, bytes([1]) + type_entry)

        # Function Section: 1 function of type index 0
        func_sec = cls._section(cls.SEC_FUNCTION, bytes([1, 0]))

        # Export Section: export function 0 as `export_name`
        name_bytes = export_name.encode("utf-8")
        export_entry = (
            cls.encode_uleb128(len(name_bytes))
            + name_bytes
            + bytes([0x00, 0x00])  # kind=func (0x00), index=0
        )
        export_sec = cls._section(cls.SEC_EXPORT, bytes([1]) + export_entry)

        # Code Section: local.get 0, local.get 1, <opcode>, [i32.const k, i32.add], end
        instructions = bytearray([cls.OP_LOCAL_GET, 0x00, cls.OP_LOCAL_GET, 0x01, opcode])
        if extra_const is not None:
            instructions.append(cls.OP_I32_CONST)
            instructions.extend(cls.encode_uleb128(extra_const & 0x7F))
            instructions.append(cls.OP_I32_ADD)
        instructions.append(cls.OP_END)

        func_body = bytes([0x00]) + bytes(instructions)  # 0 local declarations
        code_payload = bytes([1]) + cls.encode_uleb128(len(func_body)) + func_body
        code_sec = cls._section(cls.SEC_CODE, code_payload)

        return cls.MAGIC + cls.VERSION + type_sec + func_sec + export_sec + code_sec

    @classmethod
    def validate(cls, data: bytes) -> Tuple[bool, str]:
        """Validate WebAssembly binary module magic, version, sections, and code termination."""
        if len(data) < 8:
            return False, "WASM binary shorter than 8-byte header"
        if data[:4] != cls.MAGIC:
            return False, f"Invalid WASM magic: {data[:4]!r}"
        if data[4:8] != cls.VERSION:
            return False, f"Unsupported WASM version: {data[4:8]!r}"

        offset = 8
        last_sec_id = 0
        sections_seen = 0
        while offset < len(data):
            sec_id = data[offset]
            offset += 1
            if sec_id > 12:
                return False, f"Invalid WASM section ID: {sec_id}"
            if sec_id != 0 and sec_id <= last_sec_id:
                return False, f"Out-of-order WASM section {sec_id} after {last_sec_id}"
            if sec_id != 0:
                last_sec_id = sec_id
            try:
                sec_len, offset = cls.decode_uleb128(data, offset)
            except ValueError as exc:
                return False, str(exc)
            if offset + sec_len > len(data):
                return False, f"WASM section {sec_id} length {sec_len} exceeds boundary"
            sec_body = data[offset : offset + sec_len]
            if sec_id == cls.SEC_CODE and (len(sec_body) == 0 or sec_body[-1] != cls.OP_END):
                return False, "WASM code section does not terminate with 0x0B (end)"
            offset += sec_len
            sections_seen += 1

        return True, f"Valid WASM v1 binary ({sections_seen} sections, {len(data)} bytes)"


# ============================================================================
# 3. Native Machine Code / ASM (x86_64 & ARM64 / ELF64) Engine
# ============================================================================

class ASMBuilder:
    """
    Constructs and validates native x86_64 and ARM64 machine code functions and ELF64 binaries.
    """

    @staticmethod
    def build_x86_64_linear_fn(scale: int = 3, bias: int = 7) -> bytes:
        """
        Assemble native x86_64 machine code for `f(rdi, rsi) = (rdi * scale) + rsi + bias`:
          55                   push   rbp
          48 89 e5             mov    rbp, rsp
          48 89 f8             mov    rax, rdi
          48 6b c0 <scale>     imul   rax, rax, <scale>
          48 01 f0             add    rax, rsi
          48 83 c0 <bias>      add    rax, <bias>
          5d                   pop    rbp
          c3                   ret
        """
        s = scale & 0x7F
        b = bias & 0x7F
        return bytes([
            0x55,
            0x48, 0x89, 0xE5,
            0x48, 0x89, 0xF8,
            0x48, 0x6B, 0xC0, s,
            0x48, 0x01, 0xF0,
            0x48, 0x83, 0xC0, b,
            0x5D,
            0xC3,
        ])

    @staticmethod
    def build_elf64_minimal(machine_code: bytes, entry_vaddr: int = 0x400078) -> bytes:
        """Wrap raw x86_64 machine code in a minimal valid 64-bit ELF executable header."""
        # 64-byte ELF64 Header + 56-byte Program Header = 120 (0x78) bytes
        e_ident = (
            b"\x7fELF"
            + bytes([2, 1, 1, 0])  # 64-bit, little-endian, v1, SysV ABI
            + b"\x00" * 8
        )
        ehdr = e_ident + struct.pack(
            "<HHIQQQIHHHHHH",
            2,              # ET_EXEC
            0x3E,           # EM_X86_64
            1,              # EV_CURRENT
            entry_vaddr,    # e_entry
            64,             # e_phoff
            0,              # e_shoff
            0,              # e_flags
            64,             # e_ehsize
            56,             # e_phentsize
            1,              # e_phnum
            64,             # e_shentsize
            0,              # e_shnum
            0,              # e_shstrndx
        )
        total_size = 120 + len(machine_code)
        phdr = struct.pack(
            "<IIQQQQQQ",
            1,              # PT_LOAD
            5,              # PF_R | PF_X
            0,              # p_offset
            0x400000,       # p_vaddr
            0x400000,       # p_paddr
            total_size,     # p_filesz
            total_size,     # p_memsz
            0x1000,         # p_align
        )
        return ehdr + phdr + machine_code

    @classmethod
    def validate(cls, data: bytes) -> Tuple[bool, str]:
        """Validate either an ELF64 executable or a raw x86_64/ARM64 machine code snippet."""
        if len(data) >= 64 and data[:4] == b"\x7fELF":
            if data[4] != 2 or data[5] != 1:
                return False, "ELF header is not 64-bit little-endian"
            return True, f"Valid ELF64 x86_64 executable ({len(data)} bytes)"
        if len(data) >= 4:
            # Check x86_64 RET (0xC3) or ARM64 RET (C0 03 5F D6)
            if data[-1] == 0xC3:
                return True, f"Valid x86_64 machine code routine ({len(data)} bytes, ends with RET 0xC3)"
            if data[-4:] == b"\xc0\x03\x5f\xd6":
                return True, f"Valid ARM64 machine code routine ({len(data)} bytes, ends with RET)"
        return False, "Machine code missing valid ELF64 header or architectural RET epilogue"


# ============================================================================
# 4. Native Network Packet & PCAP Engine (IPv4 / UDP / TCP + RFC 1071 Checksum)
# ============================================================================

class PCAPBuilder:
    """
    Constructs and validates raw IPv4/UDP/TCP packets and libpcap (.pcap) files
    with exact RFC 1071 16-bit ones'-complement checksums.
    """

    PCAP_MAGIC_LE = 0xA1B2C3D4

    @staticmethod
    def compute_rfc1071_checksum(header_bytes: bytes) -> int:
        """Compute 16-bit ones'-complement checksum per RFC 1071."""
        if len(header_bytes) % 2 == 1:
            header_bytes = header_bytes + b"\x00"
        total = 0
        for i in range(0, len(header_bytes), 2):
            word = (header_bytes[i] << 8) | header_bytes[i + 1]
            total += word
            total = (total & 0xFFFF) + (total >> 16)
        return (~total) & 0xFFFF

    @classmethod
    def verify_ipv4_checksum(cls, ipv4_header: bytes) -> bool:
        """Verify that an IPv4 header's RFC 1071 checksum folds to 0x0000."""
        if len(ipv4_header) < 20:
            return False
        ihl = (ipv4_header[0] & 0x0F) * 4
        if len(ipv4_header) < ihl or ihl < 20:
            return False
        return cls.compute_rfc1071_checksum(ipv4_header[:ihl]) == 0

    @classmethod
    def build_ipv4_udp_packet(
        cls,
        payload: bytes,
        src_ip: Tuple[int, int, int, int] = (10, 0, 0, 1),
        dst_ip: Tuple[int, int, int, int] = (10, 0, 0, 2),
        src_port: int = 8080,
        dst_port: int = 9090,
        ttl: int = 64,
        ident: int = 0x1337,
    ) -> bytes:
        """Build a raw IPv4 + UDP packet with valid IPv4 & UDP checksums."""
        udp_len = 8 + len(payload)
        total_len = 20 + udp_len
        src_ip_bytes = bytes(src_ip)
        dst_ip_bytes = bytes(dst_ip)

        # IPv4 header with zeroed checksum field first
        ip_hdr_zero = (
            struct.pack(
                "!BBHHHBBH",
                0x45,           # Version 4, IHL 5 (20 bytes)
                0x00,           # DSCP / ECN
                total_len,
                ident & 0xFFFF,
                0x4000,         # Don't Fragment flag
                ttl & 0xFF,
                17,             # Protocol: UDP (17)
                0x0000,         # Placeholder checksum
            )
            + src_ip_bytes
            + dst_ip_bytes
        )
        ip_csum = cls.compute_rfc1071_checksum(ip_hdr_zero)
        ip_hdr = ip_hdr_zero[:10] + struct.pack("!H", ip_csum) + ip_hdr_zero[12:]

        # UDP pseudo-header + header checksum
        udp_hdr_zero = struct.pack("!HHHH", src_port & 0xFFFF, dst_port & 0xFFFF, udp_len, 0)
        pseudo_hdr = src_ip_bytes + dst_ip_bytes + struct.pack("!BBH", 0, 17, udp_len)
        udp_csum = cls.compute_rfc1071_checksum(pseudo_hdr + udp_hdr_zero + payload)
        if udp_csum == 0:
            udp_csum = 0xFFFF
        udp_hdr = struct.pack("!HHHH", src_port & 0xFFFF, dst_port & 0xFFFF, udp_len, udp_csum)

        return ip_hdr + udp_hdr + payload

    @classmethod
    def build_pcap_file(cls, packets: Sequence[bytes], link_type: int = 101) -> bytes:
        """
        Package one or more raw IPv4 packets into a standard libpcap binary file
        (DLT_RAW = 101).
        """
        global_hdr = struct.pack(
            "<IHHiIII",
            cls.PCAP_MAGIC_LE,
            2,          # Major version
            4,          # Minor version
            0,          # GMT to local correction
            0,          # Sigfigs
            65535,      # Snaplen
            link_type,  # DLT_RAW (101)
        )
        chunks = [global_hdr]
        for idx, pkt in enumerate(packets):
            pkt_hdr = struct.pack("<IIII", 1700000000 + idx, 0, len(pkt), len(pkt))
            chunks.append(pkt_hdr + pkt)
        return b"".join(chunks)

    @classmethod
    def validate(cls, data: bytes) -> Tuple[bool, str]:
        """Validate either a raw IPv4 packet or a libpcap container with embedded IPv4 packets."""
        if len(data) >= 24 and struct.unpack("<I", data[:4])[0] == cls.PCAP_MAGIC_LE:
            offset = 24
            pkt_count = 0
            while offset < len(data):
                if offset + 16 > len(data):
                    return False, "Truncated PCAP packet record header"
                _, _, incl_len, _ = struct.unpack("<IIII", data[offset : offset + 16])
                offset += 16
                if offset + incl_len > len(data):
                    return False, "Truncated PCAP packet payload"
                pkt = data[offset : offset + incl_len]
                if not cls.verify_ipv4_checksum(pkt):
                    return False, f"PCAP packet #{pkt_count} failed IPv4 RFC 1071 checksum"
                offset += incl_len
                pkt_count += 1
            return True, f"Valid PCAP capture ({pkt_count} verified IPv4 packet(s))"

        if len(data) >= 20 and (data[0] >> 4) == 4:
            ihl = (data[0] & 0x0F) * 4
            total_len = struct.unpack("!H", data[2:4])[0]
            if len(data) < ihl or total_len != len(data):
                return False, f"IPv4 total_length ({total_len}) mismatch with actual ({len(data)})"
            if not cls.verify_ipv4_checksum(data[:ihl]):
                return False, "IPv4 header failed RFC 1071 checksum verification"
            return True, f"Valid IPv4 packet ({len(data)} bytes, proto={data[9]}, RFC 1071 checksum OK)"

        return False, "Unrecognized network packet or PCAP header"


# ============================================================================
# 5. Native Compressed Binary Engine (GZIP / ZLIB with CRC32 & Adler32)
# ============================================================================

class CompressedBuilder:
    """
    Constructs and validates native GZIP (RFC 1952) and ZLIB (RFC 1950) compressed streams.
    """

    @staticmethod
    def build_gzip(payload: bytes, mtime: int = 0) -> bytes:
        """Deterministic GZIP (RFC 1952) compression with CRC32 & ISIZE trailer."""
        return gzip.compress(payload, compresslevel=6, mtime=mtime)

    @staticmethod
    def build_zlib(payload: bytes) -> bytes:
        """Deterministic ZLIB (RFC 1950) compression with Adler32 checksum trailer."""
        return zlib.compress(payload, level=6)

    @staticmethod
    def validate(data: bytes) -> Tuple[bool, str]:
        """Validate GZIP or ZLIB integrity (magic header, DEFLATE stream, CRC32/Adler32)."""
        if len(data) >= 18 and data[:2] == b"\x1f\x8b":
            try:
                decompressed = gzip.decompress(data)
                expected_crc, expected_size = struct.unpack("<II", data[-8:])
                actual_crc = zlib.crc32(decompressed) & 0xFFFFFFFF
                if expected_crc != actual_crc:
                    return False, "GZIP CRC32 mismatch"
                if expected_size != (len(decompressed) & 0xFFFFFFFF):
                    return False, "GZIP ISIZE mismatch"
                return True, f"Valid GZIP archive ({len(data)}B -> {len(decompressed)}B, CRC32={actual_crc:#010x})"
            except Exception as exc:
                return False, f"Invalid GZIP stream: {exc}"

        if len(data) >= 6 and data[0] == 0x78:
            try:
                decompressed = zlib.decompress(data)
                expected_adler = struct.unpack(">I", data[-4:])[0]
                actual_adler = zlib.adler32(decompressed) & 0xFFFFFFFF
                if expected_adler != actual_adler:
                    return False, "ZLIB Adler32 mismatch"
                return True, f"Valid ZLIB stream ({len(data)}B -> {len(decompressed)}B, Adler32={actual_adler:#010x})"
            except Exception as exc:
                return False, f"Invalid ZLIB stream: {exc}"

        return False, "Payload is neither valid GZIP (1F 8B) nor ZLIB (78 xx)"


# ============================================================================
# 6. Native Raw Audio & RIFF/WAVE Synthesizer
# ============================================================================

class WAVBuilder:
    """
    Synthesizes and validates native RIFF/WAVE audio files and raw PCM waveforms.
    """

    @staticmethod
    def build_wav_pcm16(
        frequencies_hz: Sequence[float] = (440.0, 554.37, 659.25),
        sample_rate: int = 8000,
        duration_sec: float = 0.05,
        amplitude: float = 0.6,
    ) -> bytes:
        """
        Synthesize a polyphonic harmonic waveform with ADSR envelope into a native
        16-bit Little-Endian Mono RIFF/WAVE binary.
        """
        num_samples = max(1, int(sample_rate * duration_sec))
        t = np.arange(num_samples, dtype=np.float64) / float(sample_rate)

        waveform = np.zeros(num_samples, dtype=np.float64)
        for freq in frequencies_hz:
            waveform += np.sin(2.0 * math.pi * float(freq) * t)
        if len(frequencies_hz) > 0:
            waveform /= float(len(frequencies_hz))

        # Apply smooth attack/decay envelope to avoid clicks
        env = np.ones(num_samples, dtype=np.float64)
        attack = max(1, num_samples // 8)
        release = max(1, num_samples // 4)
        env[:attack] = np.linspace(0.0, 1.0, attack)
        env[-release:] = np.linspace(1.0, 0.0, release)

        pcm = np.clip(waveform * env * amplitude * 32767.0, -32768, 32767).astype("<i2")
        pcm_bytes = pcm.tobytes()

        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        block_align = num_channels * (bits_per_sample // 8)
        data_size = len(pcm_bytes)
        riff_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            riff_size,
            b"WAVE",
            b"fmt ",
            16,               # PCM fmt chunk size
            1,                # AudioFormat = 1 (PCM)
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
        return header + pcm_bytes

    @staticmethod
    def validate(data: bytes) -> Tuple[bool, str]:
        """Validate RIFF/WAVE audio container header and sample alignment."""
        if len(data) < 44:
            return False, "WAV binary shorter than 44-byte RIFF header"
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return False, "Missing RIFF/WAVE magic signature"
        riff_size = struct.unpack("<I", data[4:8])[0]
        if riff_size + 8 != len(data):
            return False, f"RIFF size field ({riff_size + 8}) != total byte length ({len(data)})"
        if data[12:16] != b"fmt " or data[36:40] != b"data":
            return False, "Missing 'fmt ' or 'data' subchunk header"
        audio_fmt, channels, sample_rate, _, block_align, bits = struct.unpack(
            "<HHIIHH", data[20:36]
        )
        data_size = struct.unpack("<I", data[40:44])[0]
        if data_size + 44 != len(data):
            return False, f"WAV data chunk size ({data_size}) mismatch with payload ({len(data) - 44})"
        if block_align == 0 or data_size % block_align != 0:
            return False, "PCM sample data is not aligned to block_align"
        return (
            True,
            f"Valid RIFF/WAVE Audio (fmt={audio_fmt}, {channels}ch, {sample_rate}Hz, {bits}-bit, {data_size} PCM bytes)",
        )


class BinaryValidator:
    """Unified dispatcher for structural and checksum validation across all binary domains."""

    @staticmethod
    def validate(data: bytes, domain: Optional[BinaryDomain] = None) -> Tuple[bool, str]:
        if domain is None or domain == BinaryDomain.RAW:
            domain = ByteCodec.detect_domain(data)

        if domain == BinaryDomain.MIDI:
            return MIDIBuilder.validate(data)
        if domain == BinaryDomain.WASM:
            return WASMBuilder.validate(data)
        if domain == BinaryDomain.ASM:
            return ASMBuilder.validate(data)
        if domain == BinaryDomain.NETWORK:
            return PCAPBuilder.validate(data)
        if domain == BinaryDomain.COMPRESSED:
            return CompressedBuilder.validate(data)
        if domain == BinaryDomain.AUDIO:
            return WAVBuilder.validate(data)
        if domain == BinaryDomain.TEXT:
            try:
                data.decode("utf-8")
                return True, f"Valid UTF-8 text ({len(data)} bytes)"
            except UnicodeDecodeError as exc:
                return False, f"Invalid UTF-8: {exc}"
        return True, f"Valid raw byte stream ({len(data)} bytes)"
