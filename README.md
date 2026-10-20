# ollamamaker

Create tuned Ollama models from your local hardware profile, generate Modelfiles automatically, and update local `.opencode/opencode.json` settings for the new model.

`ollamamaker` expands on the original context-limit helper workflow by not only recommending a context window, but also writing the configuration and creating the model for you.

## What it does

- Inspects an installed local Ollama model (via Ollama API).
- Estimates practical `num_ctx` tiers based on:
  - model architecture metadata,
  - available VRAM,
  - available RAM,
  - selected KV cache type (`f16`, `q8_0`, `q4_0`).
- Chooses a recommended context size.
- Uses workload profiles instead of automatically selecting the largest context that fits:
  - `fast`: 32K context / 4K output
  - `balanced`: 64K context / 8K output (default)
  - `long`: 128K context / 8K output
  - `maximum`: largest GPU-safe context / up to 16K output
- Writes a Modelfile (default location: `~/.ollama/custom`).
- Updates `.opencode/opencode.json` with provider/model limits.
- Runs `ollama create` for your new model.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Ollama installed and running locally
  - expected base URL: `http://localhost:11434`
- Linux is currently the primary target for hardware autodetection
  - VRAM: `nvidia-smi`
  - RAM: `/proc/meminfo`

## Install

From this repository:

```bash
uv tool install -e .
```

## Quick start

Interactive flow:

```bash
ollamamaker
```

You will be prompted for:

- source model (must already exist in local Ollama)
- new model name (blank uses suggested default)
- KV cache type
- context profile

Then `ollamamaker` will:

1. print a recommendation report,
2. write Modelfile,
3. update `.opencode/opencode.json`,
4. run `ollama create`.

## CLI usage

```text
ollamamaker [model] [--name NAME] [--kv-cache-type {f16,q4_0,q8_0}] [--profile {fast,balanced,long,maximum} | --context TOKENS] [--output TOKENS] [--parallel N] [--vram GB] [--ram GB] [--modelfile PATH] [--edit]
```

### Arguments

- `model` (optional): source model id, example `qwen3.5:9b`

### Options

- `--name NAME`: output model name for `ollama create`
- `--kv-cache-type {f16,q4_0,q8_0}`: cache precision used for recommendations
- `--profile {fast,balanced,long,maximum}`: context and output policy
- `--context TOKENS`: exact context target instead of a profile target
- `--output TOKENS`: output target, capped by the model limit and one quarter of context
- `--parallel N`: parallel requests included in KV-cache estimates; defaults to `OLLAMA_NUM_PARALLEL` or `1`
- `--vram GB`: override autodetected VRAM
- `--ram GB`: override autodetected RAM
- `--modelfile PATH`: explicit Modelfile path
- `--edit`: open a Modelfile in your editor without running model creation

## Modelfile behavior

Default Modelfile location:

- `~/.ollama/custom/<model_name>.Modelfile`

Name sanitization for filename:

- `/` and `:` are converted to `_`

Example:

- model name: `todobot:latest`
- Modelfile path: `~/.ollama/custom/todobot_latest.Modelfile`

## Edit mode

Use edit mode to manually open Modelfiles:

```bash
uv run ollamamaker --edit
```

Edit resolution order:

1. `--modelfile` if provided
2. default path from `--name` if provided
3. otherwise interactive selection from existing `~/.ollama/custom/*.Modelfile` or creation of a new one

## opencode.json updates

`ollamamaker` updates/creates `.opencode/opencode.json` in the current working directory.

It ensures and updates:

- `$schema`: `https://opencode.ai/config.json`
- `provider.ollama.npm`: `@ai-sdk/openai-compatible`
- `provider.ollama.name`: `Ollama (local)`
- `provider.ollama.options.baseURL`: `http://localhost:11434/v1`
- `provider.ollama.models.<new_model_name>.name`
- `provider.ollama.models.<new_model_name>.limit.context`
- `provider.ollama.models.<new_model_name>.limit.output`

Other existing config keys are preserved.

## Context profiles

Profiles keep context and output budgets coherent for interactive use. A target is automatically reduced when it
exceeds the model architecture limit or cannot fit entirely in GPU memory. Recommendations reserve 10% of VRAM plus
runtime overhead, and context memory is multiplied by the configured parallel-request count.

`maximum` is deliberately opt-in. It selects the largest architecture-supported tier that remains GPU-only; the CLI
never recommends a context that requires model placement in system RAM.

Examples:

```bash
ollamamaker qwen3:4b --name qwen3-balanced:latest --profile balanced
ollamamaker qwen3:4b --name qwen3-long:latest --profile long
ollamamaker qwen3:4b --name qwen3-96k:latest --context 98304 --output 8192
```

## Notes on KV cache type

KV cache type is a runtime memory format, not the model weight quantization.

- `f16`: highest memory use, baseline quality
- `q8_0`: balanced memory savings
- `q4_0`: largest memory savings, most quality tradeoff risk

Pick the type that matches your runtime configuration expectations.

## Development

Run lint checks:

```bash
uv run ruff check
```

Run tests:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

Show CLI help:

```bash
uv run ollamamaker --help
```
