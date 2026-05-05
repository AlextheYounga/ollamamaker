"""Shared data models and constants for ollamamaker."""

from dataclasses import dataclass


class OllamaError(Exception):
    """Raised when communication with Ollama fails."""


@dataclass
class ModelArch:
    """Normalized architecture metadata returned from Ollama."""

    name: str
    architecture: str
    context_length: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_ctx_override: int
    parameter_count: int
    quantization_level: str
    model_weight_mib: int


KNOWN_OUTPUT_LIMITS: dict[str, int] = {
    "qwen3": 32_768,
    "qwen35": 8_192,
    "qwen3moe": 32_768,
    "mistral3": 32_768,
    "llama3": 8_192,
    "llama": 4_096,
    "gemma3": 8_192,
    "phi3": 4_096,
}
DEFAULT_OUTPUT_LIMIT = 8_192

KV_CACHE_TYPES: dict[str, float] = {
    "f16": 2.0,
    "q8_0": 1.1,
    "q4_0": 0.6,
}
DEFAULT_KV_CACHE_TYPE = "f16"

CTX_TIERS = [2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144]
OLLAMA_API = "http://localhost:11434"

QUANT_BITS_PER_PARAM: dict[str, float] = {
    "Q2_K": 2.6,
    "Q3_K_S": 3.4,
    "Q3_K_M": 3.9,
    "Q3_K_L": 4.3,
    "Q4_0": 4.5,
    "Q4_1": 5.0,
    "Q4_K_S": 4.6,
    "Q4_K_M": 4.8,
    "Q5_0": 5.5,
    "Q5_1": 6.0,
    "Q5_K_S": 5.5,
    "Q5_K_M": 5.7,
    "Q6_K": 6.6,
    "Q8_0": 8.5,
    "FP16": 16.0,
    "F16": 16.0,
    "BF16": 16.0,
    "F32": 32.0,
}
DEFAULT_QUANT_BITS = 4.8
WEIGHT_OVERHEAD_FACTOR = 1.1


def estimate_model_weight_mib(parameter_count: int, quantization_level: str) -> int:
    """Estimate model memory footprint in MiB from parameters and quant."""
    if parameter_count == 0:
        return 0
    bits = QUANT_BITS_PER_PARAM.get(quantization_level.upper(), DEFAULT_QUANT_BITS)
    return int((parameter_count * bits / 8 * WEIGHT_OVERHEAD_FACTOR) / (1024 * 1024))
