# Agent notes for ~/LLM

One native **buun-llama-cpp** image, three models, one GPU, one port. Keep the folder small.

## Layout

| Path | Role |
|---|---|
| `docker-compose.yml` | Shared stack (image, `gpus: all`, port 8080, CUDA require) |
| `qwen.yml` / `qwen-uncensored.yml` / `ornith.yml` | Model overlays: `container_name`, `command`, `x-download` |
| `llm` | One helper: `qwen` / `qwen-uncensored` / `ornith` / `stop` / `build` / `download` |
| `Dockerfile` | Native CUDA 13.3.1 image `buun-llama:native` |
| `models/` | GGUFs |
| `scripts/` | Benchmarks only (`compare.sh`, `speed-results/`) |

Do not add BeeLlama, extra Dockerfiles, `params.env`, `common.sh`, or extra start scripts. Measuring tools stay under `scripts/`.

## Start / switch

```bash
./llm qwen
./llm qwen-uncensored
./llm ornith
./llm stop
./llm build
./llm download
```

Never run more than one model at once (24 GB card). GPU access is `--gpus all` via NVIDIA Container Toolkit. Do **not** bind-mount host `libcuda.so`.

Host driver is CUDA **13.2**. The image is CUDA **13.3.1**, so set `NVIDIA_REQUIRE_CUDA=cuda>=13.2` (already in the image and compose). Smoke test:

```bash
docker run --rm --gpus all -e NVIDIA_REQUIRE_CUDA="cuda>=13.2" --entrypoint nvidia-smi buun-llama:native
```

## `command: >` format (required)

Model overlays use a **folded scalar**, not a YAML list. Compose word-splits it into llama-server argv. Dockerfile `ENTRYPOINT` stays `/app/llama-server`; do not add a `sh -c` wrapper.

```yaml
services:
  llm:
    container_name: qwen
    command: >
      -m /models/Qwen3.8-27B-NVFP4-MTP-HIGH.gguf
      -a qwen3.8-27b
      --host 0.0.0.0
      --port 8080
      -ngl 99
      -np 3
      -c 262144
      -ct vbr
      --kv-unified
      --vbr-vram auto
      --fit on
      --fit-target 1024
      --no-mmproj
      --temp 0.6
      --top-p 0.95
      --top-k 20
      --min-p 0.0
      --presence-penalty 0.0
      --repeat-penalty 1.0
      -lv 3
```

**Do not write** a list (this is what we moved away from):

```yaml
# wrong
command:
  - -m
  - /models/Qwen3.8-27B-NVFP4-MTP-HIGH.gguf
  - -a
  - qwen3.8-27b
```

```yaml
# also wrong
command:
  - -m /models/Qwen3.8-27B-NVFP4-MTP-HIGH.gguf
  - -a qwen3.8-27b
```

Put flag and value on the same line inside the `>` block (`-m /models/...`, `--fit-target 1024`). One flag per line. No quotes unless a value has spaces.

`ornith.yml` is the same shape (`container_name: ornith`, `-np 2`, Ornith GGUF / alias). `qwen-uncensored.yml` is the Huihui abliterated NVFP4 sibling (`container_name: qwen-uncensored`, alias `qwen3.8-27b-uncensored`).

## Serving rules

- **No MTP.** Do not pass `--spec-type`. The GGUFs still contain the draft head; 24 GB is not enough for it plus VBR KV. Leave `nextn` tensors unloaded.
- **VRAM:** `-ngl 99`, `--vbr-vram auto`, `--fit on`, `--fit-target 1024` (~1 GB free on the RTX 5090 Laptop). Intel iGPU owns the desktop, so almost all 24 GB NVIDIA VRAM is for inference.
- Image: `buun-llama:native` (CUDA **13.3.1** toolkit, `sm_120`, `GGML_NATIVE=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `NVIDIA_REQUIRE_CUDA=cuda>=13.2`). Rebuild with `./llm build`.
