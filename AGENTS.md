# Agent notes for ~/LLM

One native **llama.cpp** image (official [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master`), four models, one GPU, one port. Keep the folder small.

## Layout

| Path | Role |
|---|---|
| `docker-compose.yml` | Shared stack (image, `gpus: all`, port 8080, CUDA require) |
| `qwen.yml` / `qwen-uncensored.yml` / `cybertiel.yml` / `saluki.yml` | Model overlays: `container_name`, `command`, `x-download` |
| `llm` | One helper: `qwen` / `qwen-uncensored` / `cybertiel` / `saluki` / `stop` / `build` / `download` |
| `Dockerfile` | Native CUDA 13.3.1 image `llamacpp:native` |
| `models/` | GGUFs |
| `scripts/` | `compare.sh` (all four, or a subset) and `speed-results/` |

Do not add extra Dockerfiles, `params.env`, `common.sh`, or extra start scripts. Measuring tools stay under `scripts/`.

## Start / switch

```bash
./llm qwen
./llm qwen-uncensored
./llm cybertiel
./llm saluki
./llm stop
./llm build
./llm download
```

Never run more than one model at once (24 GB card). GPU access is `--gpus all` via NVIDIA Container Toolkit. Do **not** bind-mount host `libcuda.so`.

Host driver is CUDA **13.2**. The image is CUDA **13.3.1**, so set `NVIDIA_REQUIRE_CUDA=cuda>=13.2` (already in the image and compose). Smoke test:

```bash
docker run --rm --gpus all -e NVIDIA_REQUIRE_CUDA="cuda>=13.2" --entrypoint nvidia-smi llamacpp:native
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
      -c 184320
      -ctk q8_0
      -ctv q8_0
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

`qwen-uncensored.yml` is the Huihui abliterated NVFP4 sibling (`container_name: qwen-uncensored`, alias `qwen3.8-27b-uncensored`, `-np 1`); MTP head in-file, unused. `cybertiel.yml` is the Huihui-abliterated Ornith-1.5 coder (`container_name: cybertiel`, `-np 1`, alias `cyber-tiel-coder-35b`, UD-Q4_K_XL); MTP stays off. `saluki.yml` is Underdog Saluki 27B 1.0 IQ2-mix (`container_name: saluki`, `-np 1`, alias `underdog-saluki-27b`, 7.9 GB, text only, no MTP block; the `mmproj-*` vision add-on is deliberately not downloaded).

## Serving rules

- **No MTP.** Do not pass `--spec-type`. The Qwen/CyberTiel GGUFs still contain the draft block; without `--spec-type` llama-server logs `unused tensor … -- ignoring` for the whole extra layer (`blk.64` / `blk.40`), not only `nextn.*`. Measured unused (llama.cpp `b11557`, same as BeeLlama): HIGH 215 MiB, Huihui 810 MiB, CyberTiel 371 MiB; Saluki has none. Inventory (also covers models since removed): `scripts/speed-results/model-sizes-20260916.txt`.
- **One slot.** Every overlay runs `-np 1`: the only client is a single pi agent, without subagents or parallel requests. `-c 184320` (the "180k" window) is the whole window for that one slot; `--kv-unified` stays in place. Do not raise `-np` without a concurrent client.
- **Shared alias:** every overlay uses `-a <model-alias>,local-qwen`. pi (`~/.pi/agent/models.json`) defines one model `local/local-qwen` (`contextWindow: 184320`) that hits whatever is listening on :8080: whichever container is up, or Strata (`~/qwen38-next/Strata`). There is no separate `strata` provider in pi. Keep `local-qwen` on any new overlay.
- **KV cache:** `-ctk q8_0` / `-ctv q8_0` on every overlay. Official llama.cpp has no KVarN/VBR (`kvarn6`, `--kv-tail-tokens`, `-ct vbr` are BeeLlama/buun-only; llama-server rejects them). The build keeps `GGML_CUDA_FA_ALL_QUANTS=ON`, so a mixed pair such as `q8_0`/`q4_0` is available if VRAM ever gets tight.
- **VRAM:** `-ngl 99`, `--fit on`, `--fit-target 1024`. Intel iGPU owns the desktop, so almost all 24 GB NVIDIA VRAM is for inference. Do not pass `--n-cpu-moe`. Do not enable unified memory. With `-ngl` and `-c` both set, `--fit` aborts (`n_gpu_layers already set by user to 99, abort`) and does **not** shrink `-c`; the window is allocated as given and the headroom is something you measure. Measured 2026-10-11 at `-c 184320`, q8_0: HIGH / Huihui 22268 MiB used after load (1717 free), CyberTiel 23248 (737 free; tightest), Saluki 14075 (9910 free); one 167k-token prompt peaked at 23260 MiB (HIGH) / 23858 MiB (CyberTiel) of 24463 without OOM. Re-test with a long prompt if `-c` or the weights grow. GGUF mmap is file cache, not CPU offload. Leave mmap on (`--no-mmap` OOMs the 27Bs against the 12g cgroup).
- **Compaction:** pi compacts in `~/.pi/agent/settings.json` via `compaction.modelOverrides["local/local-qwen"].reserveTokens: 14320`, i.e. above 184320 − 14320 = **170000** tokens. If `-c` changes, update `contextWindow` (models.json) and `reserveTokens` together.
- **Thinking:** `--reasoning on`, `--reasoning-effort xhigh`, `--reasoning-preserve` (Qwen3.8 top effort; keep the flag explicit; a baked chat template can default to `medium`). Compose also sets `LLAMA_ARG_REASONING=on`, `LLAMA_ARG_REASONING_EFFORT=xhigh`, `LLAMA_ARG_REASONING_PRESERVE=1` (llama-server warns that the CLI overrides them; harmless). pi must send `reasoning_effort` and `preserve_thinking` (upstream's `--reasoning-preserve` sets `preserve_reasoning`, which Qwen3.8's template does not read); its `thinkingLevelMap` sends `xhigh` for `high` / `xhigh` / `max`.
- **Image:** `llamacpp:native`, built by `./llm build` from the current `master` of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (treeless clone so the build number is right; now `b11557`, `5c8854f`), CUDA **13.3.1** toolkit, `sm_120` (upstream rewrites to `120a`), `GGML_NATIVE=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_SCHED_MAX_COPIES=1`, `NVIDIA_REQUIRE_CUDA=cuda>=13.2`; only `llama-server` is built. Old `beellama:native` / `beellama:prev` images may still sit in the local Docker store; they are the rollback.
