# Ollama Maker

I want to expand upon a project I created at one point for helping me create custom Ollama models. I want to automate this process. 

For reference to my past project, please check `./docs/references/ollama_ctx_limits.md`. This was a simple CLI helper that would examine an Ollama model and output the suggested model configuration settings. 

Here is some example output from my previous CLI `ctxlimits`:
```
❯ ctxlimits qwen3.5:9b

  Model            : qwen3.5:9b
  Architecture     : qwen35
  Type             : dense
  Arch max context : 262k
  Layers           : 32
  KV heads         : 16
  Head dim         : 256
  Model weight     : 5.9 GB
  Quantization     : Q4_K_M
  KV cache type    : f16
  Current num_ctx  : 2048  (Ollama default — no Modelfile override)

  VRAM available   : 16.0 GB
  RAM available    : 125.7 GB

  Max output tokens: 8k  (model architecture limit)

  num_ctx        KV cache     % of VRAM    placement
  -----------------------------------------------------------------
  2k             1.0 GB       6%           GPU-only
  4k             2.0 GB       13%          GPU-only
  8k             4.0 GB       25%          GPU-only
  16k            8.0 GB       50%          GPU-only        <-- recommended
  33k            16.0 GB      100%         GPU+RAM
  66k            32.0 GB      200%         GPU+RAM
  131k           64.0 GB      400%         GPU+RAM
  262k           128.0 GB     800%         GPU+RAM

  opencode.json snippet:
    "limit": {
      "context": 16384,
      "output": 8192
    }

  Modelfile snippet (to bake num_ctx into Ollama):
    FROM qwen3.5:9b
    PARAMETER num_ctx 16384
```

## Requirements

I would like to create a new CLI, mirroring the old one, that does the following:

1. Examines an existing ollama model (passed in by argument) and determines the model's appropriate configurations, exactly like ollama_ctx_limits did. 
2. Updates the appropriate Ollama configuration files (Modelfile) and adds the appropriate opencode JSON settings in the .opencode/opencode.json of the current directory (create if non-existent). The previous CLI simply pointed the user in the right direction 
3. Runs the ollama creation command to generate a new model based on the new Modelfile. 
4. Allows manual editing existing Modelfiles with an --edit command that takes the user directly to the appropriate Modelfile and opens it with the default editor. 

## Details

- The CLI will be named `ollamamaker`
- We will use `inquirer` to prompt the user for the required information unless they pass the appropriate flags. 
- We will need to collect the appropriate model id from the user (should be an existing Ollama model on disk, fail otherwise). Example: "qwen3.5:9b"
- We will need to collect the appropriate model name from the user, the name the user wants to call this new model. 


