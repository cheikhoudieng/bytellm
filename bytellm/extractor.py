"""
Binary Section Extractor and C/ASM <-> Machine Code Instruction Pair Generator.

1. Extractor: Extracts pure executable machine code (.text sections) from ELF and PE (.exe) files.
2. Synthetic/Decompilation Pair Generator: Generates synthetic pairs of
   (Natural language / C description <-> x86_64 machine code).
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from bytellm.codec import ASMBuilder, BinaryDomain


def extract_elf_text_section(data: bytes) -> Optional[bytes]:
    """
    Parse 64-bit ELF headers and extract the raw bytes of the executable .text section.
    """
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2:
        return None  # Not ELF64

    endian = "<" if data[5] == 1 else ">"
    e_shoff = struct.unpack_from(f"{endian}Q", data, 40)[0]
    e_shentsize = struct.unpack_from(f"{endian}H", data, 58)[0]
    e_shnum = struct.unpack_from(f"{endian}H", data, 60)[0]
    e_shstrndx = struct.unpack_from(f"{endian}H", data, 62)[0]

    if e_shnum == 0 or e_shstrndx >= e_shnum or e_shoff + (e_shnum * e_shentsize) > len(data):
        return None

    # Get section header string table
    shstr_header_off = e_shoff + (e_shstrndx * e_shentsize)
    shstr_offset = struct.unpack_from(f"{endian}Q", data, shstr_header_off + 24)[0]
    shstr_size = struct.unpack_from(f"{endian}Q", data, shstr_header_off + 32)[0]
    shstrtab = data[shstr_offset : shstr_offset + shstr_size]

    # Search for .text section
    for i in range(e_shnum):
        sh_off = e_shoff + (i * e_shentsize)
        sh_name_idx = struct.unpack_from(f"{endian}I", data, sh_off)[0]
        sh_flags = struct.unpack_from(f"{endian}Q", data, sh_off + 8)[0]
        sh_offset = struct.unpack_from(f"{endian}Q", data, sh_off + 24)[0]
        sh_size = struct.unpack_from(f"{endian}Q", data, sh_off + 32)[0]

        # Extract section name
        name_end = shstrtab.find(b"\x00", sh_name_idx)
        name = shstrtab[sh_name_idx:name_end] if name_end != -1 else b""

        # SHF_EXECINSTR = 0x4
        if name == b".text" or (sh_flags & 0x4):
            if sh_offset + sh_size <= len(data) and sh_size > 0:
                return data[sh_offset : sh_offset + sh_size]

    return None


def extract_pe_text_section(data: bytes) -> Optional[bytes]:
    """
    Parse 32-bit and 64-bit PE (.exe, .dll) headers and extract the executable code section.
    """
    if len(data) < 0x40 or data[:2] != b"MZ":
        return None

    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset + 4 > len(data) or data[pe_offset : pe_offset + 4] != b"PE\x00\x00":
        return None

    coff_offset = pe_offset + 4
    num_sections = struct.unpack_from("<H", data, coff_offset + 2)[0]
    opt_hdr_size = struct.unpack_from("<H", data, coff_offset + 16)[0]

    sec_table_offset = coff_offset + 20 + opt_hdr_size

    for i in range(num_sections):
        sec_off = sec_table_offset + (i * 40)
        if sec_off + 40 > len(data):
            break
        name = data[sec_off : sec_off + 8].rstrip(b"\x00")
        raw_size = struct.unpack_from("<I", data, sec_off + 16)[0]
        raw_offset = struct.unpack_from("<I", data, sec_off + 20)[0]
        characteristics = struct.unpack_from("<I", data, sec_off + 36)[0]

        # IMAGE_SCN_MEM_EXECUTE = 0x20000000
        if name == b".text" or (characteristics & 0x20000000):
            if raw_offset + raw_size <= len(data) and raw_size > 0:
                return data[raw_offset : raw_offset + raw_size]

    return None


def extract_executable_code(file_path: Union[str, Path]) -> Optional[bytes]:
    """Extract raw executable machine code from either an ELF or PE binary."""
    with open(file_path, "rb") as f:
        data = f.read()

    if data.startswith(b"\x7fELF"):
        return extract_elf_text_section(data)
    elif data.startswith(b"MZ"):
        return extract_pe_text_section(data)
    return None


def build_c_asm_instruction_pairs(count: int = 100, seed: int = 42) -> List[Dict]:
    """
    Generate bilingual instruction pairs:
    1. Text description / C signature -> Machine code
    2. Machine code -> Decompiled pseudo-code / explanation
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    pairs = []

    for i in range(count):
        scale = int(rng.integers(1, 20))
        bias = int(rng.integers(0, 100))
        code = ASMBuilder.build_x86_64_linear_fn(scale=scale, bias=bias)

        # Mode A: Synthesis (Instruction -> x86 machine code)
        desc_fr = f"Génère la fonction x86_64 en langage machine pour: f(a, b) = (a * {scale}) + b + {bias}"
        desc_en = f"Generate x86_64 machine code for linear function: f(a, b) = (a * {scale}) + b + {bias}"
        instruction = desc_fr if i % 2 == 0 else desc_en

        pairs.append({
            "instruction": instruction,
            "domain": BinaryDomain.ASM.value,
            "response": code.hex(),
            "reasoning": f"prologue; imul rax, {scale}; add rax, rsi; add rax, {bias}; epilogue"
        })

        # Mode B: Reverse Engineering / Decompilation (Machine code -> Explanation)
        decomp_prompt = f"Analyse et décompile cette fonction x86_64: {code.hex()}"
        decomp_resp = f"Fonction C: long compute(long a, long b) {{\n    return (a * {scale}) + b + {bias};\n}}"

        pairs.append({
            "instruction": decomp_prompt,
            "domain": BinaryDomain.TEXT.value,
            "response": decomp_resp,
            "reasoning": "Lecture des opcodes imul et add avec registres SysV ABI rdi, rsi -> retour dans rax"
        })

    return pairs
