"""
Comprehensive Unit & Integration Test Suite for ByteLLM (0-255).
Tests:
1. Tokenizer-Free Raw Byte Codec (0..255 bijection, thinking scratchpad extraction, Shannon entropy)
2. Native Binary Builders & Validators (MIDI, WebAssembly, x86_64/ELF64 ASM, IPv4/UDP/PCAP, GZIP/ZLIB, RIFF/WAVE)
3. Neural Architecture (Bit-Plane embeddings, Dynamic Byte Patching causality, Abstract Reasoning Core pondering)
4. Grammar-Constrained Binary Generation (FSM logit processors & RFC 1071 checksum repair)
5. Training Loop & Bits-Per-Byte (BPB) convergence + checkpoint persistence
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from bytellm.cli import run_demo
from bytellm.codec import (
    SPECIAL_TOKENS,
    ASMBuilder,
    BinaryDomain,
    BinaryValidator,
    ByteCodec,
    CompressedBuilder,
    MIDIBuilder,
    PCAPBuilder,
    WASMBuilder,
    WAVBuilder,
    compute_shannon_entropy,
)
from bytellm.constraints import (
    BinaryLogitProcessor,
    NetworkPacketGrammarConstraint,
)
from bytellm.model import ByteLLM, ByteLLMConfig
from bytellm.training import (
    ByteLLMTrainer,
    MultiModalBinaryDataset,
    TrainingConfig,
    evaluate_bits_per_byte,
)


def test_raw_byte_codec_bijection():
    """Verify every single octet 0..255 encodes and decodes losslessly without BPE."""
    all_octets = bytes(range(256))
    tokens = ByteCodec.encode(
        all_octets,
        domain=BinaryDomain.RAW,
        reasoning_trace=b"step1:check_bijection",
        add_bos=True,
        add_eos=True,
    )
    # Verify raw octets are mapped 1:1 to their exact integer values 0..255
    decoded = ByteCodec.decode(tokens, strip_special=True, strip_thinking=True)
    assert decoded == all_octets

    thinking, payload = ByteCodec.extract_reasoning_and_payload(tokens)
    assert thinking == b"step1:check_bijection"
    assert payload == all_octets


def test_shannon_entropy_computation():
    """Verify local sliding-window Shannon entropy on constant vs uniform random bytes."""
    zeros = bytes([0] * 64)
    ent_zeros = compute_shannon_entropy(zeros, window_size=16)
    assert np.allclose(ent_zeros, 0.0)

    distinct = bytes(range(16))
    ent_distinct = compute_shannon_entropy(distinct, window_size=16)
    # 16 distinct values in a window of 16 has exact entropy log2(16) = 4.0 bits
    assert abs(float(ent_distinct[-1]) - 4.0) < 1e-5


def test_native_midi_builder_and_validator():
    """Verify MIDI VLQ encoding/decoding and Standard MIDI File (SMF) synthesis."""
    for val in [0, 127, 128, 480, 16383, 65535]:
        vlq = MIDIBuilder.encode_vlq(val)
        decoded_val, next_off = MIDIBuilder.decode_vlq(vlq, 0)
        assert decoded_val == val
        assert next_off == len(vlq)

    smf = MIDIBuilder.build_smf(
        notes=[(0, 60, 100, 240), (120, 64, 90, 240), (120, 67, 110, 480)],
        ticks_per_quarter=480,
        tempo_bpm=120,
    )
    assert ByteCodec.detect_domain(smf) == BinaryDomain.MIDI
    valid, msg = BinaryValidator.validate(smf, BinaryDomain.MIDI)
    assert valid, msg


def test_native_wasm_builder_and_validator():
    """Verify WebAssembly ULEB128 and binary module synthesis & validation."""
    for val in [0, 1, 127, 128, 300, 65536]:
        enc = WASMBuilder.encode_uleb128(val)
        dec, off = WASMBuilder.decode_uleb128(enc, 0)
        assert dec == val and off == len(enc)

    wasm_bin = WASMBuilder.build_binary_op_module(
        export_name="mul_add",
        opcode=WASMBuilder.OP_I32_MUL,
        extra_const=19,
    )
    assert ByteCodec.detect_domain(wasm_bin) == BinaryDomain.WASM
    valid, msg = BinaryValidator.validate(wasm_bin, BinaryDomain.WASM)
    assert valid, msg


def test_native_asm_and_elf64_builder():
    """Verify x86_64 machine code and ELF64 executable header synthesis & validation."""
    fn_code = ASMBuilder.build_x86_64_linear_fn(scale=4, bias=9)
    valid_fn, msg_fn = BinaryValidator.validate(fn_code, BinaryDomain.ASM)
    assert valid_fn, msg_fn

    elf_bin = ASMBuilder.build_elf64_minimal(fn_code)
    assert ByteCodec.detect_domain(elf_bin) == BinaryDomain.ASM
    valid_elf, msg_elf = BinaryValidator.validate(elf_bin, BinaryDomain.ASM)
    assert valid_elf, msg_elf


def test_native_network_ipv4_udp_and_pcap():
    """Verify IPv4/UDP packet construction, RFC 1071 checksums, and PCAP container."""
    pkt = PCAPBuilder.build_ipv4_udp_packet(
        payload=b"HELLO_BYTE_LLM",
        src_ip=(192, 168, 1, 10),
        dst_ip=(192, 168, 1, 20),
        src_port=4433,
        dst_port=8080,
    )
    assert ByteCodec.detect_domain(pkt) == BinaryDomain.NETWORK
    valid_pkt, msg_pkt = BinaryValidator.validate(pkt, BinaryDomain.NETWORK)
    assert valid_pkt, msg_pkt

    pcap = PCAPBuilder.build_pcap_file([pkt])
    assert ByteCodec.detect_domain(pcap) == BinaryDomain.NETWORK
    valid_pcap, msg_pcap = BinaryValidator.validate(pcap, BinaryDomain.NETWORK)
    assert valid_pcap, msg_pcap


def test_native_compressed_gzip_and_zlib():
    """Verify GZIP (CRC32) and ZLIB (Adler32) binary stream synthesis & verification."""
    raw = b"Repeating binary pattern for DEFLATE compression test! " * 3
    gz = CompressedBuilder.build_gzip(raw)
    zl = CompressedBuilder.build_zlib(raw)

    assert ByteCodec.detect_domain(gz) == BinaryDomain.COMPRESSED
    assert ByteCodec.detect_domain(zl) == BinaryDomain.COMPRESSED
    assert BinaryValidator.validate(gz)[0]
    assert BinaryValidator.validate(zl)[0]


def test_native_wav_audio_synthesis():
    """Verify RIFF/WAVE PCM16 synthesis and structural validation."""
    wav = WAVBuilder.build_wav_pcm16(frequencies_hz=(440.0, 880.0), sample_rate=8000, duration_sec=0.025)
    assert ByteCodec.detect_domain(wav) == BinaryDomain.AUDIO
    valid, msg = BinaryValidator.validate(wav, BinaryDomain.AUDIO)
    assert valid, msg


def test_bytellm_causality_and_abstract_reasoning():
    """
    Verify that ByteLLM's DynamicBytePatcher and AbstractReasoningCore strictly
    preserve autoregressive causality (modifying future bytes never changes past logits).
    """
    torch.manual_seed(123)
    cfg = ByteLLMConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        patch_size=4,
        reasoning_steps=3,
        num_memory_registers=4,
        dropout=0.0,
    )
    model = ByteLLM(cfg)
    model.eval()

    seq1 = torch.tensor([[257, 10, 20, 30, 40, 50, 60, 70, 80, 90]], dtype=torch.long)
    seq2 = seq1.clone()
    # Mutate token at index 7 (inside second patch)
    seq2[0, 7] = 255

    out1 = model(seq1)
    out2 = model(seq2)

    # Logits at positions 0..6 MUST be strictly identical (no future leakage across or within patches)
    assert torch.allclose(out1["logits"][:, :7, :], out2["logits"][:, :7, :], atol=1e-5)
    assert out1["ponder_cost"].item() >= 1.0


def test_text_domain_and_instruction_codec():
    """Multilingual text is plain UTF-8 bytes; instruction pairs mask the prompt."""
    text = "Les octets: 原始字节 ✓"
    assert ByteCodec.detect_domain(text.encode("utf-8")) == BinaryDomain.TEXT
    assert BinaryValidator.validate(text.encode("utf-8"), BinaryDomain.TEXT)[0]
    assert not BinaryValidator.validate(b"\xff\xfe\xfa", BinaryDomain.TEXT)[0]

    midi = MIDIBuilder.build_smf([(0, 60, 100, 240)])
    tokens, start = ByteCodec.encode_instruction("fais un MIDI", midi, BinaryDomain.MIDI)
    assert tokens[start] == SPECIAL_TOKENS.DOMAIN_MIDI
    assert bytes(tokens[2 : start]).decode("utf-8") == "fais un MIDI"
    assert ByteCodec.decode(tokens[start:]) == midi


def test_corpus_pipeline_all_domains(tmp_path: Path):
    """Real files of every domain + instruction JSONL -> split, mixture, per-domain eval."""
    import json

    from bytellm.corpus import (
        CorpusDataset,
        build_mixture_sampler,
        evaluate_per_domain,
        is_validation_file,
        measure_validity_rate,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(12):
        (corpus / f"t{i}.txt").write_text(f"Texte {i}: bonjour le monde éàü 你好 " * 8, encoding="utf-8")
        (corpus / f"m{i}.mid").write_bytes(MIDIBuilder.build_smf([(0, 60 + i, 100, 240)]))
        (corpus / f"w{i}.wasm").write_bytes(WASMBuilder.build_binary_op_module(f"f{i}"))
        (corpus / f"a{i}.wav").write_bytes(WAVBuilder.build_wav_pcm16(duration_sec=0.01))
        (corpus / f"p{i}.pcap").write_bytes(
            PCAPBuilder.build_pcap_file([PCAPBuilder.build_ipv4_udp_packet(b"hi%d" % i)])
        )
        (corpus / f"c{i}.gz").write_bytes(CompressedBuilder.build_gzip(b"data%d" % i * 5))
    (corpus / "m0.mid").write_bytes(MIDIBuilder.build_smf([(0, 60, 100, 240)]))
    instr = tmp_path / "instr.jsonl"
    instr.write_text(
        "\n".join(
            json.dumps({"instruction": f"Écris le mot {i}", "domain": "text", "response": f"mot {i}"})
            for i in range(20)
        ),
        encoding="utf-8",
    )

    train_ds = CorpusDataset(corpus, instr, "train", seq_len=96)
    val_ds = CorpusDataset(corpus, instr, "val", seq_len=96)
    seen = {d.value for d in train_ds.domains}
    assert {"text", "midi", "wasm", "audio", "network", "compressed"} <= seen
    assert len(val_ds) > 0
    # Split is per file/record: no file appears in both splits.
    train_files = {r.path for r in train_ds.records if r.path}
    val_files = {r.path for r in val_ds.records if r.path}
    assert not (train_files & val_files)
    assert all(is_validation_file(p) for p in val_files)

    # Instruction tokens are masked out of the loss.
    idx = next(i for i, r in enumerate(train_ds.records) if r.tokens is not None)
    _, y, _ = train_ds[idx]
    assert (y[:5] == SPECIAL_TOKENS.PAD).all()

    cfg = ByteLLMConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=128, reasoning_steps=2)
    model = ByteLLM(cfg)
    trainer = ByteLLMTrainer(model, TrainingConfig(batch_size=4, max_steps=4, warmup_steps=1, seq_len=96, device="cpu"))
    trainer.train(train_ds, sampler=build_mixture_sampler(train_ds, {BinaryDomain.TEXT: 3.0}))
    per_domain = evaluate_per_domain(model, val_ds)
    assert per_domain and all(v["bpb"] > 0 for v in per_domain.values())

    rate = measure_validity_rate(model, BinaryDomain.WASM, n=2, instruction="Génère du WASM", max_new_bytes=12)
    assert 0.0 <= rate <= 1.0


def test_constrained_generation_and_training_convergence(tmp_path: Path):
    """
    Verify grammar-constrained binary generation, training BPB reduction, and CLI demo.
    """
    report = run_demo(tmp_path / "demo_out", train_steps=12)
    assert report["final_bpb"] < report["initial_bpb"]
    for domain_key, info in report["artifacts"].items():
        assert info["valid"], f"Artifact {domain_key} failed validation: {info}"


if __name__ == "__main__":
    test_raw_byte_codec_bijection()
    test_shannon_entropy_computation()
    test_native_midi_builder_and_validator()
    test_native_wasm_builder_and_validator()
    test_native_asm_and_elf64_builder()
    test_native_network_ipv4_udp_and_pcap()
    test_native_compressed_gzip_and_zlib()
    test_native_wav_audio_synthesis()
    test_bytellm_causality_and_abstract_reasoning()
    test_text_domain_and_instruction_codec()
    import tempfile
    test_corpus_pipeline_all_domains(Path(tempfile.mkdtemp()))
    test_constrained_generation_and_training_convergence(Path("scratch_test"))
    print("ALL BYTE-LLM TESTS PASSED SUCCESSFULLY!")
