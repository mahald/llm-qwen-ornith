# Local LLMs — BeeLlama.cpp on the RTX 5090 Laptop (24 GB)

One image, three models, one port. Runs on the **NVIDIA GeForce RTX 5090 Laptop** (Blackwell `sm_120`, **24 GB VRAM**). The Intel Arrow Lake iGPU drives the desktop, so almost all 24 GB on the 5090 is available for inference.

Compose shows what is running:

```bash
docker compose ls
docker compose -f docker-compose.yml -f qwen.yml ps    # or qwen-uncensored.yml / cybertiel.yml
```

- **GPU:** RTX 5090 Laptop, 24 GB
- **CPU:** Intel Core Ultra 9 275HX, image built with `-march=native`
- **Engine:** [BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) (`kvarn6` / `kvarn6` KV cache)

Weights + KV **entirely in VRAM**, ~1 GB headroom (`--fit-target 1024`).

## Qwen HIGH — the quality model

The default is **Qwen3.8-27B NVFP4 HIGH** (`./llm qwen`). In the [esatapedico family](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF) the compact tiers share the same 448-tensor NVFP4 backbone; they differ only in the ten extra tensors (LM head, embeddings, MTP draft).

On **HIGH the LM head (`output.weight`) stays BF16** — the same type as in the source conversion (`ORIG`), so it is left unchanged. The LM head maps hidden states to the vocabulary; leaving it unquantized is the biggest quality lever on this ladder. HIGH should therefore be the highest-quality variant that still runs well on **24 GB**: 17.57 GB on disk (17.35 GB loaded; 215 MiB unused MTP), Blackwell NVFP4, and leftover VRAM for the KVarN KV cache.

Qwen HIGH is the accurate dense 27B. `./llm qwen-uncensored` is the Huihui abliterated NVFP4 sibling — same 27B dense layout, refusals stripped. `./llm cybertiel` is Cyber-Tiel-Coder, the Huihui-abliterated Ornith-1.5 coder at **UD-Q4_K_XL** (22.75 GB on disk, 22.36 GB loaded; MTP off).

## Quick start

```bash
cd ~/LLM
./llm download
./llm build
./llm qwen              # NVFP4 HIGH on :8080
./llm qwen-uncensored   # Huihui abliterated NVFP4 on :8080
./llm cybertiel         # Cyber-Tiel-Coder UD-Q4_K_XL on :8080
./llm stop
```

- API: `http://127.0.0.1:8080/v1` (no key) · UI: http://127.0.0.1:8080
- Names: `qwen3.8-27b` / `qwen3.8-27b-uncensored` / `cyber-tiel-coder-35b`

Both think. The answer is in `message.content`, reasoning in `message.reasoning_content`.

## Layout

| | |
|---|---|
| `docker-compose.yml` | Image, GPU (`--gpus all`), port 8080, `NVIDIA_REQUIRE_CUDA` |
| `qwen.yml` / `qwen-uncensored.yml` / `cybertiel.yml` | Model, context, sampling, KVarN — **tune here** |
| `llm` | `qwen` / `qwen-uncensored` / `cybertiel` / `stop` / `build` / `download` |
| `Dockerfile` | Native BeeLlama build, CUDA **13.3.1** |
| `models/` | The GGUFs (not in git — `./llm download`) |
| `scripts/` | `compare.sh` (TPS via `/v1/chat/completions`) and `speed-results/` |

## Models

Disk is the GGUF on disk (decimal GB). **Loaded** is what BeeLlama maps without `--spec-type`: the extra MTP block is logged `unused tensor … ignoring` and never occupies VRAM. Measured 2026-09-16; tensor dump in [`scripts/speed-results/model-sizes-20260916.txt`](scripts/speed-results/model-sizes-20260916.txt).

| Overlay | What | NVFP4 | Disk | MTP in file | MTP (unloaded) | Loaded |
|---|---|---|---|---|---|---|
| `qwen` | Qwen3.8-27B NVFP4 **HIGH**, LM head BF16 | **yes** | 17.571 GB | yes, `blk.64` IQ4_XS | **215 MiB** | 17.345 GB |
| `qwen-uncensored` | Huihui abliterated NVFP4 | **yes** | 19.654 GB | yes, `blk.64` BF16 | **810 MiB** | 18.804 GB |
| `cybertiel` | Cyber-Tiel-Coder 35B-A3B UD-Q4_K_XL | no (UD-Q4_K_XL) | 22.750 GB | yes, grafted `blk.40` Q3_K | **371 MiB** | 22.360 GB |

Sources: [HIGH](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF) · [uncensored](https://huggingface.co/renketong/Huihui-Qwen3.8-27B-abliterated-NVFP4-GGUF) · [CyberTiel MTP](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP). Files: `Qwen3.8-27B-NVFP4-MTP-HIGH.gguf`, `Qwen3.8-27B-huihui-NVFP4.gguf`, `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`.

## No MTP

**MTP stays off.** All three GGUFs still contain the draft block; it is only loaded with `--spec-type draft-mtp`. Without that flag llama-server skips the **whole extra block** (`blk.64` on the 27Bs, `blk.40` on CyberTiel) — not just the `nextn.*` tensors. Measured unused sizes: HIGH **215 MiB**, Huihui **810 MiB**, CyberTiel **371 MiB**. 24 GB is not enough for the head plus a useful KVarN cache.

## VRAM

```
RTX 5090 Laptop       24463 MiB
Desktop (Intel iGPU)  ~6 MiB on the 5090
fit-target            1024 MiB free
Rest                  weights + KVarN KV, all GPU (-ngl 99)
```

`-ctk kvarn6 -ctv kvarn6 --kv-tail-tokens 1024`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0`, `--no-mmproj`. 35B-A3B MoE: whole trunk on the GPU (stock placement). **Nothing is meant to spill into host RAM for inference** — including the 27B dense models.

All three overlays pin `-ngl 99` and never pass `--n-cpu-moe`. `--fit on` cannot drop layers (BeeLlama logs `n_gpu_layers already set by user to 99, abort`); it only shrinks `-c` so the KVarN pool plus scratch still leave `--fit-target 1024` MiB free. Unified memory is off, so CUDA will not silently page VRAM into RAM. The compose cgroup is `mem_limit: 12g` with no container swap (`memswap_limit: 12g`).

llama.cpp still **mmaps** the GGUF (default). That is file-backed page cache of the weights file, not CPU offload. On a live load (SuperQwen, since removed from the stack) the GPU held **22691 MiB**; cgroup anon (actual process RAM) was **~366 MiB**; the rest of the 12g cap was reclaimable GGUF cache. `--no-mmap` would copy the whole file into anonymous RAM first and **OOM the 27Bs** against the 12g cap — leave mmap on.

Qwen3.8 only caches 16 of 64 layers (Gated Attention); the other 48 Gated DeltaNet layers keep a constant-size recurrent state. Native context is 262,144 tokens. `--fit on` shrinks `-c` if the KVarN pool plus graph scratch would miss `--fit-target`.

## pi

[pi](https://github.com/earendil-works/pi) is the only client: one agent, no subagents, no parallel requests. One slot (`-np 1`) is therefore enough and the whole 262,144-token window belongs to it.

```bash
pi                          # defaults to local/local-qwen
pi --model local/local-qwen
```

`~/.pi/agent/models.json` defines one provider `local` (`http://127.0.0.1:8080/v1`, no key) with one model `local-qwen`. Every overlay serves that alias (`-a <name>,local-qwen`), so whichever container is up on :8080 answers — switch with `./llm qwen` / `qwen-uncensored` / `cybertiel` and pi needs no change.

| pi setting | Value | Meaning |
|---|---|---|
| `contextWindow` (models.json) | 262144 | Same as `-c 262144` (`n_ctx_slot`) |
| `maxTokens` (models.json) | 49152 | Cap per reply |
| `compaction.modelOverrides["local/local-qwen"].reserveTokens` (settings.json) | 12144 | Auto-compaction above 262144 − 12144 = **250000** tokens |

Keep `contextWindow` and `reserveTokens` in step with `-c`: if `--fit` ever shrinks the window, the server log shows the real `n_ctx_slot`.

Default thinking is **xhigh** (`--reasoning-effort xhigh` on the server). pi sends `reasoning_effort` and `enable_thinking` / `preserve_thinking` as chat-template kwargs; its `thinkingLevelMap` turns `high` / `xhigh` / `max` into `xhigh`. Qwen3.8 only accepts `low` / `medium` / `xhigh` — not `high`. pi sends no sampling parameters, so the overlay's `--temp` / `--top-p` / `--top-k` defaults apply.

## Speed

RTX 5090 Laptop (24463 MiB), BeeLlama.cpp `beellama:native`, **2026-09-16**. HIGH and Uncensored re-measured **2026-10-02** on BeeLlama 0.4.7 (`58a1629`). Profile (all three): MTP off, `-ngl 99`, `kvarn6`/`kvarn6`, `--kv-tail-tokens 1024`, `--flash-attn on`, `--fit-target 1024`, thinking **xhigh**. All overlays run `-np 1` (one pi agent, no parallelism, one slot); the numbers below were measured earlier with 3 slots (27B dense) / 2 slots (35B-A3B). Native `-c 262144`.

The image ships `llama-server` only — no `llama-bench`. The standard tool is [`./scripts/compare.sh`](scripts/compare.sh): same `/v1/chat/completions` path pi uses. `./scripts/compare.sh` runs all three; `./scripts/compare.sh qwen cybertiel` a subset. The older logs in `scripts/speed-results/` still hold runs of models that are no longer in the stack.

| | Qwen HIGH | Uncensored | CyberTiel |
|---|---|---|---|
| NVFP4 | **yes** | **yes** | no (UD-Q4_K_XL) |
| Decode | **36.2 t/s** | **35.7 t/s** | **137 t/s** |
| Prefill ~1.9k | 1748 t/s | 1724 t/s | 2528 t/s |
| VRAM after load | 23149 / 835 free | 23149 / 835 | 23403 / 581 |
| Slots (as measured) | 3 | 3 | 2 |

Decode is generation t/s on a short prompt (mean of 2, `max_tokens=128`). Prefill is a ~1.9k-token summarize prompt. `--fit-target 1024` fills leftover VRAM with the KVarN pool, so used-MiB is **not** the weight size — that is the Loaded column above.

HIGH / CyberTiel: [`summary-qwen-20260916-031329.txt`](scripts/speed-results/summary-qwen-20260916-031329.txt), [`summary-cybertiel-20260916-031030.txt`](scripts/speed-results/summary-cybertiel-20260916-031030.txt). Uncensored: [`summary-20260916-135246.txt`](scripts/speed-results/summary-20260916-135246.txt) (that run also covers SuperQwen and Tiel, since removed). HIGH / Uncensored (2026-10-02): [`summary-20261002-222859.txt`](scripts/speed-results/summary-20261002-222859.txt).

## Build

`./llm build` sets `CMAKE_CUDA_ARCHITECTURES=120`. Image: CUDA **13.3.1**. Host driver 595.x speaks CUDA 13.2, so `NVIDIA_REQUIRE_CUDA=cuda>=13.2` (otherwise the 13.3.1 image will not start). `GGML_CUDA_KVARN=ON` and `GGML_CUDA_FA_ALL_QUANTS=ON` compile KVarN plus the full FlashAttention pair matrix. `GGML_SCHED_MAX_COPIES=1` skips pipeline-parallel input copies (ggml default is 4; this is a single-GPU box). GPU access is via NVIDIA Container Toolkit (`--gpus all`).

```bash
docker run --rm --gpus all -e NVIDIA_REQUIRE_CUDA="cuda>=13.2" --entrypoint nvidia-smi beellama:native
```

## Links

- https://github.com/Anbeeld/beellama.cpp
- https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF
- https://huggingface.co/renketong/Huihui-Qwen3.8-27B-abliterated-NVFP4-GGUF
- https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP
