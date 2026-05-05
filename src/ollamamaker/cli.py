"""CLI entrypoint for building tuned Ollama models."""

import argparse
import json
import os
import re
import shlex
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

MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._/-]+(?::[A-Za-z0-9._-]+)?$")
DEFAULT_MODELFILE_DIR = Path.home() / ".ollama" / "custom"


def _print_report(
    arch,
    vram_mib: int,
    ram_mib: int,
    kv_cache_type: str,
    recommended_ctx: int,
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
        print(f"  Current num_ctx  : {fmt_tokens(arch.num_ctx_override)}  (baked into Modelfile)")
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
        print(f"  {fmt_tokens(ctx):<14} {fmt_mib(kv_mib):<12} {pct:<12} {placement:<14}{marker}")


def _model_display_name(model_name: str) -> str:
    base = model_name.split(":", 1)[0]
    return base.replace("-", " ").replace("_", " ").title()


def _dict_value(container: dict, key: str) -> dict:
    value = container.get(key, {})
    if isinstance(value, dict):
        return value
    return {}


def _ensure_opencode_config(
    path: Path,
    model_name: str,
    context_limit: int,
    output_limit: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            data = {}
    if not isinstance(data, dict):
        data = {}
    if "$schema" not in data:
        data["$schema"] = "https://opencode.ai/config.json"

    provider = _dict_value(data, "provider")
    ollama_provider = _dict_value(provider, "ollama")

    ollama_provider.setdefault("npm", "@ai-sdk/openai-compatible")
    ollama_provider.setdefault("name", "Ollama (local)")

    options = _dict_value(ollama_provider, "options")
    options.setdefault("baseURL", "http://localhost:11434/v1")
    ollama_provider["options"] = options

    models = _dict_value(ollama_provider, "models")
    model_config = _dict_value(models, model_name)

    model_config.setdefault("name", _model_display_name(model_name))
    model_config["limit"] = {
        "context": context_limit,
        "output": output_limit,
    }
    models[model_name] = model_config
    ollama_provider["models"] = models
    provider["ollama"] = ollama_provider
    data["provider"] = provider

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
        subprocess.run([*shlex.split(editor), str(path)], check=True)  # noqa: S603
        return
    subprocess.run(["xdg-open", str(path)], check=True)  # noqa: S603,S607


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create tuned Ollama models from local hardware recommendations.")
    parser.add_argument("model", nargs="?", help="Source Ollama model id, example qwen3.5:9b")
    parser.add_argument("--name", help="Name for the newly created model")
    parser.add_argument("--kv-cache-type", choices=sorted(KV_CACHE_TYPES.keys()))
    parser.add_argument("--vram", type=int, default=None, metavar="GB")
    parser.add_argument("--ram", type=int, default=None, metavar="GB")
    parser.add_argument(
        "--modelfile",
        default=None,
        help="Path to write/read Modelfile (default: ~/.ollama/custom/<name>.Modelfile)",
    )
    parser.add_argument("--edit", action="store_true", help="Open the Modelfile in your editor")
    return parser


def _handle_edit(modelfile_path: Path) -> None:
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    if not modelfile_path.exists():
        modelfile_path.write_text("", encoding="utf-8")
    _open_editor(modelfile_path)


def _default_modelfile_path(new_model_name: str) -> Path:
    safe_name = new_model_name.replace("/", "_").replace(":", "_")
    return DEFAULT_MODELFILE_DIR / f"{safe_name}.Modelfile"


def _resolve_edit_modelfile_path(args: argparse.Namespace) -> Path:
    if args.modelfile:
        return Path(args.modelfile).expanduser().resolve()

    if args.name:
        return _default_modelfile_path(args.name.strip())

    DEFAULT_MODELFILE_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(DEFAULT_MODELFILE_DIR.glob("*.Modelfile"))

    if existing:
        create_label = "Create new Modelfile"
        selected = inquirer.select(
            message="Select Modelfile to edit",
            choices=[*map(str, existing), create_label],
        ).execute()
        if selected != create_label:
            return Path(selected)

    entered_name = inquirer.text(
        message="Model name for new Modelfile",
        default="",
    ).execute()
    model_name = entered_name.strip()
    if not model_name or not MODEL_NAME_RE.fullmatch(model_name):
        raise OllamaError(
            "invalid model name. Use letters/numbers and ._-/ with an optional "
            "single ':tag' suffix; spaces are not allowed"
        )
    return _default_modelfile_path(model_name)


def _run_create_model(new_model_name: str, modelfile_path: Path) -> None:
    try:
        subprocess.run(  # noqa: S603
            ["ollama", "create", new_model_name, "-f", str(modelfile_path)],  # noqa: S607
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"error: ollama create failed with exit code {exc.returncode}",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)


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
    suggested_name = f"{model_id.split(':')[0]}-custom:latest"
    if args.name:
        new_model = args.name.strip()
    else:
        entered_name = inquirer.text(
            message=f"New model name (default: {suggested_name})",
            default="",
        ).execute()
        new_model = entered_name.strip() or suggested_name
    if not new_model or not MODEL_NAME_RE.fullmatch(new_model):
        raise OllamaError(
            "invalid model name. Use letters/numbers and ._-/ with an optional "
            "single ':tag' suffix; spaces are not allowed"
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
    """Run the ollamamaker command line workflow."""
    args = _build_parser().parse_args()

    if args.edit:
        try:
            _handle_edit(_resolve_edit_modelfile_path(args))
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    try:
        model_id, new_model_name, kv_cache_type = _collect_inputs(args)
        arch = fetch_model_arch(model_id)
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    modelfile_path = (
        Path(args.modelfile).expanduser().resolve() if args.modelfile else _default_modelfile_path(new_model_name)
    )

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
    _ensure_opencode_config(opencode_path, new_model_name, chosen_ctx, output_limit)

    print()
    print(f"Updated Modelfile: {modelfile_path}")
    print(f"Updated opencode config: {opencode_path}")
    print(f"Creating model: {new_model_name}")

    _run_create_model(new_model_name, modelfile_path)

    print(f"Model created: {new_model_name}")


if __name__ == "__main__":
    main()
