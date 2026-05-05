import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from InquirerPy import inquirer

from .core import (
    detect_hardware,
    fetch_available_models,
    fetch_model_arch,
    fmt_mib,
    fmt_tokens,
    max_output_tokens,
    recommended_context,
    recommended_tiers,
)
from .models import DEFAULT_KV_CACHE_TYPE, KV_CACHE_TYPES, OllamaError


def _print_report(
    arch, vram_mib: int, ram_mib: int, kv_cache_type: str, recommended_ctx: int
) -> None:
    tiers = recommended_tiers(arch, vram_mib, ram_mib, kv_cache_type)
    max_out = max_output_tokens(arch)
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
    print(f"  KV cache type    : {kv_cache_type}")
    if arch.num_ctx_override:
        print(
            f"  Current num_ctx  : {fmt_tokens(arch.num_ctx_override)}  (baked into Modelfile)"
        )
    else:
        print("  Current num_ctx  : 2048  (Ollama default - no Modelfile override)")
    print()
    print(f"  VRAM available   : {fmt_mib(vram_mib)}")
    print(f"  RAM available    : {fmt_mib(ram_mib)}")
    print()
    print(f"  Max output tokens: {fmt_tokens(max_out)}  (model architecture limit)")
    print()
    print("  num_ctx        KV cache     % of VRAM    placement")
    print("  -----------------------------------------------------------------")
    for ctx, kv_mib, placement in tiers:
        marker = "        <-- recommended" if ctx == recommended_ctx else ""
        pct = f"{(kv_mib / vram_mib) * 100:.0f}%" if vram_mib else "n/a"
        print(
            f"  {fmt_tokens(ctx):<14} {fmt_mib(kv_mib):<12} {pct:<12} {placement:<14}{marker}"
        )


def _ensure_opencode_config(path: Path, context_limit: int, output_limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            data = {}
    if not isinstance(data, dict):
        data = {}
    limit = data.get("limit", {})
    if not isinstance(limit, dict):
        limit = {}
    limit["context"] = context_limit
    limit["output"] = output_limit
    data["limit"] = limit
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_modelfile(path: Path, source_model: str, context_limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"FROM {source_model}\nPARAMETER num_ctx {context_limit}\n",
        encoding="utf-8",
    )


def _open_editor(path: Path) -> None:
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor:
        subprocess.run([editor, str(path)], check=True)
        return
    subprocess.run(["xdg-open", str(path)], check=True)


def _collect_inputs(args: argparse.Namespace) -> tuple[str, str, str]:
    available_models = fetch_available_models()
    if not available_models:
        raise OllamaError("no local Ollama models found")
    model_id = (
        args.model
        or inquirer.select(
            message="Select source Ollama model",
            choices=sorted(available_models),
        ).execute()
    )
    if model_id not in available_models:
        raise OllamaError(f"model '{model_id}' is not installed locally")
    new_model = (
        args.name
        or inquirer.text(
            message="New model name",
            default=f"{model_id.split(':')[0]}-custom:latest",
        ).execute()
    )
    kv_cache_type = (
        args.kv_cache_type
        or inquirer.select(
            message="KV cache type",
            choices=sorted(KV_CACHE_TYPES.keys()),
            default=DEFAULT_KV_CACHE_TYPE,
        ).execute()
    )
    return model_id, new_model, kv_cache_type


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create tuned Ollama models from local hardware recommendations."
    )
    parser.add_argument(
        "model", nargs="?", help="Source Ollama model id, example qwen3.5:9b"
    )
    parser.add_argument("--name", help="Name for the newly created model")
    parser.add_argument("--kv-cache-type", choices=sorted(KV_CACHE_TYPES.keys()))
    parser.add_argument("--vram", type=int, default=None, metavar="GB")
    parser.add_argument("--ram", type=int, default=None, metavar="GB")
    parser.add_argument(
        "--modelfile", default="Modelfile", help="Path to write/read Modelfile"
    )
    parser.add_argument(
        "--edit", action="store_true", help="Open the Modelfile in your editor"
    )
    args = parser.parse_args()

    modelfile_path = Path(args.modelfile).resolve()
    if args.edit:
        modelfile_path.parent.mkdir(parents=True, exist_ok=True)
        if not modelfile_path.exists():
            modelfile_path.write_text("", encoding="utf-8")
        _open_editor(modelfile_path)
        return

    try:
        model_id, new_model_name, kv_cache_type = _collect_inputs(args)
        arch = fetch_model_arch(model_id)
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    auto_vram, auto_ram = detect_hardware()
    vram_mib = args.vram * 1024 if args.vram is not None else auto_vram
    ram_mib = args.ram * 1024 if args.ram is not None else auto_ram
    if not vram_mib:
        print(
            "warning: VRAM not detected; pass --vram for better recommendations",
            file=sys.stderr,
        )

    chosen_ctx = recommended_context(arch, vram_mib, ram_mib, kv_cache_type)
    if not chosen_ctx:
        print(
            "error: could not calculate a viable context recommendation",
            file=sys.stderr,
        )
        sys.exit(1)

    _print_report(arch, vram_mib, ram_mib, kv_cache_type, chosen_ctx)
    output_limit = max_output_tokens(arch)

    _write_modelfile(modelfile_path, model_id, chosen_ctx)
    opencode_path = Path.cwd() / ".opencode" / "opencode.json"
    _ensure_opencode_config(opencode_path, chosen_ctx, output_limit)

    print()
    print(f"Updated Modelfile: {modelfile_path}")
    print(f"Updated opencode config: {opencode_path}")
    print(f"Creating model: {new_model_name}")

    try:
        subprocess.run(
            ["ollama", "create", new_model_name, "-f", str(modelfile_path)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"error: ollama create failed with exit code {exc.returncode}",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)

    print(f"Model created: {new_model_name}")


if __name__ == "__main__":
    main()
