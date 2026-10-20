"""Terminal reporting for recommended Ollama runtime settings."""

from dataclasses import dataclass

from .core import fmt_mib, fmt_tokens, kv_cache_bytes, max_output_tokens, recommended_tiers
from .models import ModelArch


@dataclass(frozen=True)
class RecommendationReport:
    """Values displayed after selecting a runtime configuration."""

    arch: ModelArch
    vram_mib: int
    ram_mib: int
    kv_cache_type: str
    context: int
    profile: str
    parallel: int
    output_limit: int


def print_report(report: RecommendationReport) -> None:
    """Print model details, memory tiers, and the chosen configuration."""
    arch = report.arch
    tiers = recommended_tiers(
        arch,
        report.vram_mib,
        report.ram_mib,
        report.kv_cache_type,
        report.parallel,
    )
    if report.context not in {ctx for ctx, _, _ in tiers}:
        kv_mib = (
            kv_cache_bytes(arch, report.context, report.kv_cache_type)
            * report.parallel
            // (1024 * 1024)
        )
        tiers.append((report.context, kv_mib, "GPU-only"))
        tiers.sort()

    print()
    print(f"  Model            : {arch.name}")
    print(f"  Architecture     : {arch.architecture}")
    print(f"  Arch max context : {fmt_tokens(arch.context_length)}")
    print(f"  Layers           : {arch.num_layers}")
    print(f"  KV heads         : {arch.num_kv_heads}")
    print(f"  Head dim         : {arch.head_dim}")
    if arch.model_weight_mib:
        print(f"  Model weight     : {fmt_mib(arch.model_weight_mib)}")
    if arch.quantization_level:
        print(f"  Quantization     : {arch.quantization_level}")
    print(f"  KV cache type    : {report.kv_cache_type}")
    print(f"  Context profile  : {report.profile}")
    print(f"  Parallel requests: {report.parallel}")
    if arch.num_ctx_override:
        print(f"  Current num_ctx  : {fmt_tokens(arch.num_ctx_override)}  (baked into Modelfile)")
    else:
        print("  Current num_ctx  : Ollama default (no Modelfile override)")
    print()
    print(f"  VRAM available   : {fmt_mib(report.vram_mib)}")
    print(f"  RAM available    : {fmt_mib(report.ram_mib)}")
    print()
    print(f"  Max output tokens: {fmt_tokens(max_output_tokens(arch))}  (model architecture limit)")
    print(f"  Chosen output    : {fmt_tokens(report.output_limit)}")
    print()
    print("  num_ctx        KV allocation  % of VRAM    placement")
    print("  -----------------------------------------------------------------")
    for ctx, kv_mib, placement in tiers:
        marker = "        <-- recommended" if ctx == report.context else ""
        pct = f"{(kv_mib / report.vram_mib) * 100:.0f}%" if report.vram_mib else "n/a"
        print(f"  {fmt_tokens(ctx):<14} {fmt_mib(kv_mib):<14} {pct:<12} {placement:<14}{marker}")
