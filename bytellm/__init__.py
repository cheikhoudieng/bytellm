"""
ByteLLM: Tokenizer-Free Raw Byte Large Language Model (0-255)
with Hierarchical Latent Patching, Recursive Abstract Reasoning,
and Grammar-Constrained Native Binary Generation.
"""

from bytellm.codec import (
    ByteCodec,
    BinaryDomain,
    BinaryValidator,
    MIDIBuilder,
    WASMBuilder,
    ASMBuilder,
    PCAPBuilder,
    WAVBuilder,
    CompressedBuilder,
    compute_shannon_entropy,
)
from bytellm.model import (
    ByteLLMConfig,
    ByteLLM,
    VanillaByteTransformer,
    AbstractReasoningCore,
    DynamicBytePatcher,
)
from bytellm.constraints import (
    BinaryLogitProcessor,
    DomainGrammarConstraint,
)
from bytellm.extractor import (
    extract_executable_code,
    extract_elf_text_section,
    extract_pe_text_section,
    build_c_asm_instruction_pairs,
)
from bytellm.training import (
    ByteLLMTrainer,
    TrainingConfig,
    MultiModalBinaryDataset,
    evaluate_bits_per_byte,
)

__version__ = "1.0.0"
__all__ = [
    "ByteCodec",
    "BinaryDomain",
    "BinaryValidator",
    "MIDIBuilder",
    "WASMBuilder",
    "ASMBuilder",
    "PCAPBuilder",
    "WAVBuilder",
    "CompressedBuilder",
    "compute_shannon_entropy",
    "ByteLLMConfig",
    "ByteLLM",
    "VanillaByteTransformer",
    "AbstractReasoningCore",
    "DynamicBytePatcher",
    "BinaryLogitProcessor",
    "DomainGrammarConstraint",
    "ByteLLMTrainer",
    "TrainingConfig",
    "MultiModalBinaryDataset",
    "evaluate_bits_per_byte",
]
