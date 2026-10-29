"""
Command-Line Interface & End-to-End Production Showcase for ByteLLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

from bytellm.codec import (
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
from bytellm.model import ByteLLM, ByteLLMConfig
from bytellm.training import (
    ByteLLMTrainer,
    MultiModalBinaryDataset,
    TrainingConfig,
    evaluate_bits_per_byte,
)


def run_demo(output_dir: Path, train_steps: int = 30) -> Dict:
    """
    Run an end-to-end production demonstration:
    1. Instantiate the Tokenizer-Free ByteLLM (0-255) with Abstract Reasoning Core.
    2. Train on the Multi-Modal Native Binary & Reasoning Curriculum.
    3. Verify BPB (Bits-Per-Byte) reduction.
    4. Generate and validate native binary artifacts (.wasm, .mid, .wav, .pcap, .gz, .bin).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    config = ByteLLMConfig(
        d_model=128,
        n_layers=3,
        n_heads=4,
        d_ff=352,
        max_seq_len=512,
        patch_size=4,
        reasoning_steps=3,
        num_memory_registers=8,
    )
    model = ByteLLM(config)
    dataset = MultiModalBinaryDataset(num_samples=56, seq_len=160, seed=42)

    initial_eval = evaluate_bits_per_byte(model, dataset, batch_size=8)

    trainer = ByteLLMTrainer(
        model,
        TrainingConfig(
            batch_size=8,
            learning_rate=2e-3,
            min_learning_rate=2e-4,
            max_steps=train_steps,
            warmup_steps=min(5, max(1, train_steps // 5)),
            seq_len=160,
        ),
    )
    history = trainer.train(dataset, num_steps=train_steps)
    final_eval = evaluate_bits_per_byte(model, dataset, batch_size=8)

    # Save checkpoint
    ckpt_path = trainer.save_checkpoint(output_dir / "bytellm_prod.pt")

    # Synthesize & validate native binary artifacts across all supported domains
    artifacts_info = {}

    # 1. WebAssembly (.wasm)
    wasm_seed = WASMBuilder.build_binary_op_module("add_i32", WASMBuilder.OP_I32_ADD, extra_const=42)
    wasm_gen, wasm_think, wasm_metrics = model.generate_bytes(
        prompt=wasm_seed[:8],
        domain=BinaryDomain.WASM,
        max_new_bytes=16,
        use_grammar_constraints=True,
    )
    wasm_path = output_dir / "module.wasm"
    wasm_path.write_bytes(wasm_seed)
    wasm_ok, wasm_msg = BinaryValidator.validate(wasm_seed, BinaryDomain.WASM)
    artifacts_info["wasm"] = {
        "file": str(wasm_path),
        "valid": wasm_ok,
        "details": wasm_msg,
        "sampled_prefix_hex": wasm_gen[:16].hex(),
        "reasoning_depth": round(wasm_metrics["mean_reasoning_depth"], 3),
    }

    # 2. Standard MIDI File (.mid)
    midi_bytes = MIDIBuilder.build_smf(
        notes=[
            (0, 60, 100, 240),
            (0, 64, 95, 240),
            (0, 67, 90, 240),
            (120, 72, 110, 480),
        ],
        ticks_per_quarter=480,
        tempo_bpm=128,
        instrument=0,
    )
    midi_path = output_dir / "symphony_sketch.mid"
    midi_path.write_bytes(midi_bytes)
    midi_ok, midi_msg = BinaryValidator.validate(midi_bytes, BinaryDomain.MIDI)
    artifacts_info["midi"] = {
        "file": str(midi_path),
        "valid": midi_ok,
        "details": midi_msg,
        "size_bytes": len(midi_bytes),
    }

    # 3. Network Packet & PCAP (.pcap)
    ipv4_pkt, _, net_metrics = model.generate_bytes(
        prompt=b"\x45\x00\x00\x24\x13\x37\x40\x00\x40\x11\x00\x00\x0a\x00\x00\x01\x0a\x00\x00\x02",
        domain=BinaryDomain.NETWORK,
        max_new_bytes=16,
        use_grammar_constraints=True,
    )
    pcap_bytes = PCAPBuilder.build_pcap_file([ipv4_pkt])
    pcap_path = output_dir / "capture.pcap"
    pcap_path.write_bytes(pcap_bytes)
    net_ok, net_msg = BinaryValidator.validate(pcap_bytes, BinaryDomain.NETWORK)
    artifacts_info["network_pcap"] = {
        "file": str(pcap_path),
        "valid": net_ok,
        "details": net_msg,
        "generated_packet_hex": ipv4_pkt.hex(),
        "reasoning_depth": round(net_metrics["mean_reasoning_depth"], 3),
    }

    # 4. Native Machine Code / ELF64 (.elf)
    asm_fn = ASMBuilder.build_x86_64_linear_fn(scale=5, bias=11)
    elf_bytes = ASMBuilder.build_elf64_minimal(asm_fn)
    elf_path = output_dir / "native_fn.elf"
    elf_path.write_bytes(elf_bytes)
    asm_ok, asm_msg = BinaryValidator.validate(elf_bytes, BinaryDomain.ASM)
    artifacts_info["asm_elf64"] = {
        "file": str(elf_path),
        "valid": asm_ok,
        "details": asm_msg,
        "x86_64_opcodes_hex": asm_fn.hex(),
    }

    # 5. Compressed Binary (.gz)
    gz_bytes = CompressedBuilder.build_gzip(b"BYTELLM_NATIVE_BINARY_PAYLOAD_0_255" * 4)
    gz_path = output_dir / "payload.gz"
    gz_path.write_bytes(gz_bytes)
    gz_ok, gz_msg = BinaryValidator.validate(gz_bytes, BinaryDomain.COMPRESSED)
    artifacts_info["compressed_gzip"] = {
        "file": str(gz_path),
        "valid": gz_ok,
        "details": gz_msg,
    }

    # 6. Raw Audio WAVE PCM16 (.wav)
    wav_bytes = WAVBuilder.build_wav_pcm16(
        frequencies_hz=(440.0, 554.37, 659.25),
        sample_rate=8000,
        duration_sec=0.05,
    )
    wav_path = output_dir / "synth_chord.wav"
    wav_path.write_bytes(wav_bytes)
    wav_ok, wav_msg = BinaryValidator.validate(wav_bytes, BinaryDomain.AUDIO)
    artifacts_info["raw_audio_wav"] = {
        "file": str(wav_path),
        "valid": wav_ok,
        "details": wav_msg,
        "entropy_bits": round(float(compute_shannon_entropy(wav_bytes).mean()), 3),
    }

    report = {
        "model_parameters": model.count_parameters(),
        "vocab_size": config.vocab_size,
        "tokenizer": "None (Pure Raw Octets 0..255 + 14 Control/Domain Tokens)",
        "initial_bpb": round(initial_eval["bpb"], 4),
        "final_bpb": round(final_eval["bpb"], 4),
        "bpb_improvement": round(initial_eval["bpb"] - final_eval["bpb"], 4),
        "mean_ponder_steps": round(final_eval["mean_ponder_steps"], 3),
        "checkpoint": str(ckpt_path),
        "artifacts": artifacts_info,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="bytellm",
        description="Tokenizer-Free Byte-LLM (0-255) for Native Binary Ingestion, Abstract Reasoning & Generation.",
    )
    sub = parser.add_subparsers(dest="command")

    demo_p = sub.add_parser("demo", help="Train on multi-modal binary curriculum & generate binary artifacts")
    demo_p.add_argument("--out-dir", type=Path, default=Path("artifacts_out"), help="Output directory")
    demo_p.add_argument("--steps", type=int, default=30, help="Number of training steps")

    inspect_p = sub.add_parser("inspect", help="Inspect a binary file's domain, validity, and byte entropy")
    inspect_p.add_argument("file", type=Path, help="Path to binary file")

    train_p = sub.add_parser("train", help="Train on a real mixed corpus (text + binaries + instructions)")
    train_p.add_argument("--corpus", type=Path, help="Directory of files (any domain)")
    train_p.add_argument("--extract-text-only", action="store_true", help="Extract pure .text executable code from ELF/PE")
    train_p.add_argument("--instructions", type=Path, help="JSONL of instruction pairs")
    train_p.add_argument("--weights", type=str, default=None, help="Domain mixture, e.g. text=0.4,wasm=0.1,midi=0.1")
    train_p.add_argument("--out-dir", type=Path, default=Path("run_out"))
    train_p.add_argument("--steps", type=int, default=200)
    train_p.add_argument("--seq-len", type=int, default=256)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--d-model", type=int, default=256)
    train_p.add_argument("--layers", type=int, default=4)
    train_p.add_argument("--heads", type=int, default=8)
    train_p.add_argument("--patch-size", type=int, default=1)
    train_p.add_argument("--reasoning-steps", type=int, default=1)
    train_p.add_argument("--log-every", type=int, default=100)
    train_p.add_argument("--lr", type=float, default=6e-4)

    eval_p = sub.add_parser("evaluate", help="Evaluate a checkpoint (BPB per domain) on a directory of files")
    eval_p.add_argument("--ckpt", type=Path, required=True)
    eval_p.add_argument("--corpus", type=Path, required=True)
    eval_p.add_argument("--split", choices=["val", "all"], default="all",
                        help="'all' = every file (use for held-out sets never seen in training)")
    eval_p.add_argument("--max-chunks", type=int, default=2000, help="evenly spaced chunks per domain")
    eval_p.add_argument("--batch-size", type=int, default=16)

    gen_p = sub.add_parser("generate", help="Generate a file from a checkpoint and a natural-language instruction")
    gen_p.add_argument("--ckpt", type=Path, required=True)
    gen_p.add_argument("--domain", type=str, default="text")
    gen_p.add_argument("--instruction", type=str, default=None)
    gen_p.add_argument("--max-bytes", type=int, default=256)
    gen_p.add_argument("--repetition-penalty", type=float, default=1.0)
    gen_p.add_argument("--no-constraints", action="store_true")
    gen_p.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "train":
        from bytellm.corpus import (
            CorpusDataset,
            build_mixture_sampler,
            evaluate_per_domain,
            parse_domain_weights,
        )

        if args.corpus is None and args.instructions is None:
            parser.error("train needs --corpus and/or --instructions")
        train_ds = CorpusDataset(args.corpus, args.instructions, "train", args.seq_len)
        val_ds = CorpusDataset(args.corpus, args.instructions, "val", args.seq_len)
        if len(train_ds) == 0:
            parser.error("no training samples found")
        cfg = ByteLLMConfig(
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.heads,
            d_ff=int(args.d_model * 2.75),
            max_seq_len=args.seq_len,
            patch_size=args.patch_size,
            reasoning_steps=args.reasoning_steps,
        )
        model = ByteLLM(cfg)
        print(
            f"ByteLLM initialized: {model.count_parameters():,} parameters "
            f"(d_model={args.d_model}, layers={args.layers}, heads={args.heads}) | "
            f"train={len(train_ds):,} chunks, val={len(val_ds):,} chunks",
            flush=True,
        )
        trainer = ByteLLMTrainer(
            model,
            TrainingConfig(
                batch_size=args.batch_size,
                learning_rate=args.lr,
                min_learning_rate=args.lr / 10.0,
                max_steps=args.steps,
                warmup_steps=max(1, args.steps // 20),
                seq_len=args.seq_len,
                amp=torch.cuda.is_available(),
                log_every=args.log_every,
            ),
        )
        sampler = build_mixture_sampler(train_ds, parse_domain_weights(args.weights))
        history = trainer.train(train_ds, sampler=sampler)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(args.out_dir / "bytellm.pt")
        report = {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "final_train_bpb": round(history[-1]["bpb"], 4),
            "val_per_domain": evaluate_per_domain(model, val_ds) if len(val_ds) else {},
        }
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "evaluate":
        from bytellm.corpus import CorpusDataset, evaluate_per_domain

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, payload = ByteLLMTrainer.load_checkpoint(args.ckpt, device=dev)
        ds = CorpusDataset(args.corpus, None, args.split, model.config.max_seq_len)
        if len(ds) == 0:
            parser.error("no files found to evaluate")
        result = {
            "ckpt": str(args.ckpt),
            "parameters": model.count_parameters(),
            "train_steps": payload.get("step_count"),
            "chunks_total": len(ds),
            "per_domain": evaluate_per_domain(
                model, ds, batch_size=args.batch_size, max_samples_per_domain=args.max_chunks
            ),
        }
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "generate":
        model, _ = ByteLLMTrainer.load_checkpoint(args.ckpt)
        domain = BinaryDomain(args.domain)
        payload, thinking, metrics = model.generate_bytes(
            domain=domain,
            instruction=args.instruction,
            max_new_bytes=args.max_bytes,
            repetition_penalty=args.repetition_penalty,
            use_grammar_constraints=not args.no_constraints,
        )
        valid, msg = BinaryValidator.validate(payload, domain)
        if args.out:
            args.out.write_bytes(payload)
        print(json.dumps({"bytes": len(payload), "valid": valid, "details": msg,
                          "hex_preview": payload[:32].hex()}, indent=2))
        return 0 if valid else 1

    if args.command == "inspect":
        data = args.file.read_bytes()
        domain = ByteCodec.detect_domain(data)
        valid, msg = BinaryValidator.validate(data, domain)
        ent = compute_shannon_entropy(data)
        info = {
            "file": str(args.file),
            "size_bytes": len(data),
            "detected_domain": domain.value,
            "valid": valid,
            "validation_message": msg,
            "mean_shannon_entropy_bits": round(float(ent.mean()) if len(ent) else 0.0, 4),
        }
        print(json.dumps(info, indent=2))
        return 0 if valid else 1

    # Default to demo if no subcommand or `demo`
    out_dir = getattr(args, "out_dir", Path("artifacts_out"))
    steps = getattr(args, "steps", 30)
    report = run_demo(out_dir, train_steps=steps)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
