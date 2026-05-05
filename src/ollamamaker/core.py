"""Core Ollama data fetching and recommendation calculations."""

import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .models import (
    CTX_TIERS,
    DEFAULT_KV_CACHE_TYPE,
    DEFAULT_OUTPUT_LIMIT,
    KNOWN_OUTPUT_LIMITS,
    KV_CACHE_TYPES,
    OLLAMA_API,
    ModelArch,
    OllamaError,
    estimate_model_weight_mib,
)

RUNTIME_OVERHEAD_MIB = 512
THOUSAND = 1_000
MILLION = 1_000_000
MIB = 1_024


def fmt_tokens(n: int) -> str:
    """Format token counts in compact human-readable units."""
    if n >= MILLION:
        return f"{n / MILLION:.0f}M"
    if n >= THOUSAND:
        return f"{n / THOUSAND:.0f}k"
    return str(n)


def fmt_mib(n: int) -> str:
    """Format MiB values as MB/GB display strings."""
    if n >= MIB:
        return f"{n / MIB:.1f} GB"
    return f"{n} MB"


def parse_num_ctx(parameters_str: str) -> int:
    """Extract a `num_ctx` value from Ollama parameters text."""
    for line in parameters_str.splitlines():
        match = re.match(r"^\s*num_ctx\s+(\d+)", line)
        if match:
            return int(match.group(1))
    return 0


def kv_cache_bytes(
    arch: ModelArch,
    num_ctx: int,
    kv_cache_type: str = DEFAULT_KV_CACHE_TYPE,
) -> int:
    """Compute KV cache bytes for a context size."""
    dtype_bytes = KV_CACHE_TYPES.get(kv_cache_type, KV_CACHE_TYPES[DEFAULT_KV_CACHE_TYPE])
    return int(num_ctx * arch.num_layers * arch.num_kv_heads * arch.head_dim * 2 * dtype_bytes)


def max_output_tokens(arch: ModelArch) -> int:
    """Return architecture output-token cap using known defaults."""
    return KNOWN_OUTPUT_LIMITS.get(arch.architecture, DEFAULT_OUTPUT_LIMIT)


def recommended_tiers(
    arch: ModelArch,
    vram_mib: int,
    ram_mib: int,
    kv_cache_type: str = DEFAULT_KV_CACHE_TYPE,
) -> list[tuple[int, int, str]]:
    """Return context tiers with KV size and placement label."""
    vram_for_kv = max(0, vram_mib - arch.model_weight_mib - RUNTIME_OVERHEAD_MIB)
    total_memory_mib = vram_mib + ram_mib
    results = []
    for ctx in CTX_TIERS:
        if ctx > arch.context_length:
            break
        kv_mib = kv_cache_bytes(arch, ctx, kv_cache_type) // (1024 * 1024)
        label = "too large"
        if kv_mib <= vram_for_kv:
            label = "GPU-only"
        elif kv_mib <= total_memory_mib:
            label = "GPU+RAM"
        results.append((ctx, kv_mib, label))
    return results


def recommended_context(
    arch: ModelArch,
    vram_mib: int,
    ram_mib: int,
    kv_cache_type: str = DEFAULT_KV_CACHE_TYPE,
) -> int:
    """Choose the highest recommended context based on available memory."""
    tiers = recommended_tiers(arch, vram_mib, ram_mib, kv_cache_type)
    viable = [(ctx, lbl) for ctx, _, lbl in tiers if lbl != "too large"]
    if not viable:
        return 0
    gpu_only = [ctx for ctx, lbl in viable if lbl == "GPU-only"]
    return gpu_only[-1] if gpu_only else viable[-1][0]


def detect_hardware() -> tuple[int, int]:
    """Detect total VRAM/RAM in MiB from host machine."""
    vram = 0
    ram = 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],  # noqa: S607
            stderr=subprocess.DEVNULL,
            text=True,
        )
        vram = sum(int(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit())
    except (subprocess.SubprocessError, OSError, ValueError):
        vram = 0

    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    ram = int(line.split()[1]) // 1024
                    break
    except (OSError, ValueError, IndexError):
        ram = 0
    return vram, ram


def _api_post(path: str, payload: dict, api_base: str) -> dict:
    """POST helper for Ollama API calls."""
    req = urllib.request.Request(  # noqa: S310
        f"{api_base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:  # noqa: S310
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            msg = json.loads(body).get("error", body)
        except (json.JSONDecodeError, ValueError):
            msg = body
        raise OllamaError(msg) from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise OllamaError(f"could not connect to Ollama at {api_base} - {exc}") from exc


def _api_get(path: str, api_base: str) -> dict:
    req = urllib.request.Request(f"{api_base}{path}")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=10) as response:  # noqa: S310
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return {}


def fetch_available_models(api_base: str = OLLAMA_API) -> list[str]:
    """List locally available Ollama model ids."""
    resp = _api_get("/api/tags", api_base)
    return [model.get("name", "") for model in resp.get("models", []) if model.get("name")]


def _fetch_model_vram_from_ps(model: str, api_base: str) -> int:
    ps = _api_get("/api/ps", api_base)
    for candidate in ps.get("models", []):
        if candidate.get("name") == model or candidate.get("model") == model:
            return int(candidate.get("size_vram", 0)) // (1024 * 1024)
    return 0


def fetch_model_arch(model: str, api_base: str = OLLAMA_API) -> ModelArch:
    """Fetch and normalize architecture metadata for a model."""
    resp = _api_post("/api/show", {"model": model}, api_base)
    model_info = resp.get("model_info", {})
    details = resp.get("details", {})
    parameters = resp.get("parameters", "")

    architecture = model_info.get("general.architecture", details.get("family", "unknown"))

    def pick_value(suffix: str, default: int = 0) -> int:
        return int(model_info.get(f"{architecture}.{suffix}", default) or default)

    context_length = pick_value("context_length") or pick_value("max_position_embeddings")
    num_layers = pick_value("block_count")
    num_kv_heads = pick_value("attention.head_count_kv") or pick_value("attention.head_count")
    head_dim = pick_value("attention.key_length") or pick_value("attention.value_length")

    if not all([context_length, num_layers, num_kv_heads, head_dim]):
        raise OllamaError(
            "could not extract architecture metadata "
            f"for '{model}' (context={context_length}, layers={num_layers}, "
            f"kv_heads={num_kv_heads}, head_dim={head_dim})"
        )

    parameter_count = int(model_info.get("general.parameter_count", 0) or 0)
    quantization_level = str(details.get("quantization_level", ""))
    model_weight_mib = _fetch_model_vram_from_ps(model, api_base)
    if model_weight_mib == 0 and parameter_count and quantization_level:
        model_weight_mib = estimate_model_weight_mib(parameter_count, quantization_level)

    return ModelArch(
        name=model,
        architecture=architecture,
        context_length=context_length,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_ctx_override=parse_num_ctx(parameters),
        parameter_count=parameter_count,
        quantization_level=quantization_level,
        model_weight_mib=model_weight_mib,
    )
