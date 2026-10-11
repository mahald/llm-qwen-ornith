# Agent notes for ~/LLM

One native **BeeLlama.cpp** image, three models, one GPU, one port. Keep the folder small.

## Layout

| Path | Role |
|---|---|
| `docker-compose.yml` | Shared stack (image, `gpus: all`, port 8080, CUDA require) |
| `qwen.yml` / `qwen-uncensored.yml` / `cybertiel.yml` | Model overlays: `container_name`, `command`, `x-download` |
| `llm` | One helper: `qwen` / `qwen-uncensored` / `cybertiel` / `stop` / `build` / `download` |
| `Dockerfile` | Native CUDA 13.3.1 image `beellama:native` |
| `models/` | GGUFs |
| `scripts/` | `compare.sh` (all three, or a subset) and `speed-results/` |

Do not add extra Dockerfiles, `params.env`, `common.sh`, or extra start scripts. Measuring tools stay under `scripts/`.

## Start / switch

```bash
./llm qwen
./llm qwen-uncensored
./llm cybertiel
./llm stop
./llm build
./llm download
```

Never run more than one model at once (24 GB card). GPU access is `--gpus all` via NVIDIA Container Toolkit. Do **not** bind-mount host `libcuda.so`.

Host driver is CUDA **13.2**. The image is CUDA **13.3.1**, so set `NVIDIA_REQUIRE_CUDA=cuda>=13.2` (already in the image and compose). Smoke test:

```bash
docker run --rm --gpus all -e NVIDIA_REQUIRE_CUDA="cuda>=13.2" --entrypoint nvidia-smi beellama:native
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
      -np 1
      -c 262144
      -ctk kvarn6
      -ctv kvarn6
      --kv-tail-tokens 1024
      --kv-unified
      --flash-attn on
      --fit on
      --fit-target 1024
      --no-mmproj
      --reasoning on
      --reasoning-effort xhigh
      --reasoning-preserve
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

`qwen-uncensored.yml` is the Huihui abliterated NVFP4 sibling (`container_name: qwen-uncensored`, alias `qwen3.8-27b-uncensored`, `-np 1`); MTP head in-file, unused. `cybertiel.yml` is the Huihui-abliterated Ornith-1.5 coder (`container_name: cybertiel`, `-np 1`, alias `cyber-tiel-coder-35b`, UD-Q4_K_XL); MTP stays off.

## Serving rules

- **No MTP.** Do not pass `--spec-type`. All three GGUFs still contain the draft block; without `--spec-type` llama-server skips the whole extra layer (`blk.64` / `blk.40`), not only `nextn.*`. Measured unused: HIGH 215 MiB, Huihui 810 MiB, CyberTiel 371 MiB. Inventory (also covers models since removed): `scripts/speed-results/model-sizes-20260916.txt`.
- **One slot.** Every overlay runs `-np 1`: the only client is a single pi agent, without subagents or parallel requests. `-c 262144` is the whole window for that one slot; `--kv-unified` stays in place. Do not raise `-np` without a concurrent client.
- **Shared alias:** every overlay uses `-a <model-alias>,local-qwen`. pi (`~/.pi/agent/models.json`) defines one model `local/local-qwen` (`contextWindow: 262144`) that hits whichever container is up on :8080. Keep `local-qwen` on any new overlay.
- **KV cache:** `-ctk kvarn6` / `-ctv kvarn6`, `--kv-tail-tokens 1024` (BeeLlama KVarN + F16 precision tail). Not VBR (`-ct vbr` is buun-only).
- **VRAM:** `-ngl 99`, `--fit on`, `--fit-target 1024` (~1 GB free on the RTX 5090 Laptop). Intel iGPU owns the desktop, so almost all 24 GB NVIDIA VRAM is for inference. Do not pass `--n-cpu-moe`. Do not enable unified memory. `--fit` must not lower `-ngl` (it is pinned at 99); it only shrinks `-c`. 27B weights + KV stay on the GPU; GGUF mmap is file cache, not CPU offload. Leave mmap on (`--no-mmap` OOMs the 27Bs against the 12g cgroup).
- **Compaction:** pi compacts in `~/.pi/agent/settings.json` via `compaction.modelOverrides["local/local-qwen"].reserveTokens: 12144`, i.e. above 262144 − 12144 = **250000** tokens. If `-c` changes, update `contextWindow` (models.json) and `reserveTokens` together.
- **Thinking:** `--reasoning on`, `--reasoning-effort xhigh`, `--reasoning-preserve` (Qwen3.8 top effort; keep the flag explicit; a baked chat template can default to `medium`). Compose also sets `LLAMA_ARG_REASONING=on`, `LLAMA_ARG_REASONING_EFFORT=xhigh`, `LLAMA_ARG_REASONING_PRESERVE=1`. pi must send `reasoning_effort` (not only `enable_thinking`); its `thinkingLevelMap` sends `xhigh` for `high` / `xhigh` / `max`.
- Image: `beellama:native` ([Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp), CUDA **13.3.1** toolkit, `sm_120`, `GGML_NATIVE=ON`, `GGML_CUDA_KVARN=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_SCHED_MAX_COPIES=1`, `NVIDIA_REQUIRE_CUDA=cuda>=13.2`). Rebuild with `./llm build`.
