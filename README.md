# Local LLMs — llama.cpp on the RTX 5090 Laptop (24 GB)

One image, four models, one port. Runs on the **NVIDIA GeForce RTX 5090 Laptop** (Blackwell `sm_120`, **24 GB VRAM**). The Intel Arrow Lake iGPU drives the desktop, so almost all 24 GB on the 5090 is available for inference.

Compose shows what is running:

```bash
docker compose ls
docker compose -f docker-compose.yml -f qwen.yml ps    # or qwen-uncensored.yml / cybertiel.yml / saluki.yml
```

- **GPU:** RTX 5090 Laptop, 24 GB
- **CPU:** Intel Core Ultra 9 275HX, image built with `-march=native`
- **Engine:** official [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master` (build `b11557`, commit `5c8854f`), `q8_0` / `q8_0` KV cache, 184,320-token window

Weights + KV **entirely in VRAM**, ~1 GB headroom (`--fit-target 1024`).

## Qwen HIGH — the quality model

The default is **Qwen3.8-27B NVFP4 HIGH** (`./llm qwen`). In the [esatapedico family](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF) the compact tiers share the same 448-tensor NVFP4 backbone; they differ only in the ten extra tensors (LM head, embeddings, MTP draft).

On **HIGH the LM head (`output.weight`) stays BF16** — the same type as in the source conversion (`ORIG`), so it is left unchanged. The LM head maps hidden states to the vocabulary; leaving it unquantized is the biggest quality lever on this ladder. HIGH should therefore be the highest-quality variant that still runs well on **24 GB**: 17.57 GB on disk (17.35 GB loaded; 215 MiB unused MTP), Blackwell NVFP4, and leftover VRAM for the q8_0 KV cache.

Qwen HIGH is the accurate dense 27B. `./llm qwen-uncensored` is the Huihui abliterated NVFP4 sibling — same 27B dense layout, refusals stripped. `./llm cybertiel` is Cyber-Tiel-Coder, the Huihui-abliterated Ornith-1.5 coder at **UD-Q4_K_XL** (22.75 GB on disk, 22.36 GB loaded; MTP off). `./llm saluki` is [Underdog Saluki 27B 1.0](https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0), a 2-bit IQ2-mix of Qwen3.8-27B tuned to keep tool calling intact (7.9 GB on disk, text only).

## Quick start

```bash
cd ~/LLM
./llm download
./llm build
./llm qwen              # NVFP4 HIGH on :8080
./llm qwen-uncensored   # Huihui abliterated NVFP4 on :8080
./llm cybertiel         # Cyber-Tiel-Coder UD-Q4_K_XL on :8080
./llm saluki            # Underdog Saluki 27B IQ2-mix on :8080
./llm stop
```

- API: `http://127.0.0.1:8080/v1` (no key) · UI: http://127.0.0.1:8080
- Names: `qwen3.8-27b` / `qwen3.8-27b-uncensored` / `cyber-tiel-coder-35b` / `underdog-saluki-27b` (plus the shared `local-qwen`)

Both think. The answer is in `message.content`, reasoning in `message.reasoning_content`.

## Layout

| | |
|---|---|
| `docker-compose.yml` | Image, GPU (`--gpus all`), port 8080, `NVIDIA_REQUIRE_CUDA` |
| `qwen.yml` / `qwen-uncensored.yml` / `cybertiel.yml` / `saluki.yml` | Model, context, sampling, KV cache — **tune here** |
| `llm` | `qwen` / `qwen-uncensored` / `cybertiel` / `saluki` / `stop` / `build` / `download` |
| `Dockerfile` | Native llama.cpp (ggml-org `master`) build, CUDA **13.3.1** |
| `models/` | The GGUFs (not in git — `./llm download`) |
| `scripts/` | `compare.sh` (TPS via `/v1/chat/completions`) and `speed-results/` |

## Models

Disk is the GGUF on disk (decimal GB). **Loaded** is what llama-server maps without `--spec-type`: the extra MTP block is logged `unused tensor … ignoring` and never occupies VRAM. Measured 2026-09-16 (unused sizes re-checked 2026-10-11 on llama.cpp `b11557`, identical); tensor dump in [`scripts/speed-results/model-sizes-20260916.txt`](scripts/speed-results/model-sizes-20260916.txt).

| Overlay | What | NVFP4 | Disk | MTP in file | MTP (unloaded) | Loaded |
|---|---|---|---|---|---|---|
| `qwen` | Qwen3.8-27B NVFP4 **HIGH**, LM head BF16 | **yes** | 17.571 GB | yes, `blk.64` IQ4_XS | **215 MiB** | 17.345 GB |
| `qwen-uncensored` | Huihui abliterated NVFP4 | **yes** | 19.654 GB | yes, `blk.64` BF16 | **810 MiB** | 18.804 GB |
| `cybertiel` | Cyber-Tiel-Coder 35B-A3B UD-Q4_K_XL | no (UD-Q4_K_XL) | 22.750 GB | yes, grafted `blk.40` Q3_K | **371 MiB** | 22.360 GB |
| `saluki` | Underdog Saluki 27B 1.0 IQ2-mix (2-bit, text only) | no (IQ2-mix) | 7.898 GB | no | — | ≈7.9 GB |

Sources: [HIGH](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF) · [uncensored](https://huggingface.co/renketong/Huihui-Qwen3.8-27B-abliterated-NVFP4-GGUF) · [CyberTiel MTP](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP) · [Saluki](https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0). Files: `Qwen3.8-27B-NVFP4-MTP-HIGH.gguf`, `Qwen3.8-27B-huihui-NVFP4.gguf`, `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`, `Underdog-Saluki-27B-1.0-IQ2-mix.gguf`. The Saluki vision add-on (`mmproj-…`) is not downloaded; pi only sends text.

**Saluki is meant to run on a 16 GB graphics card too.** At `-c 184320` with `q8_0` KV it used 14076 MiB after load and peaked at 15136 MiB on a 167k-token prompt (measured on the 24 GB card, not on a real 16 GB one) — about 1 GiB spare on 16 GB, provided nothing else uses that GPU's VRAM. If it gets tight, lower `-c` (and pi's `contextWindow` / `reserveTokens` with it) or use `-ctv q4_0`.

## No MTP

**MTP stays off.** The three Qwen/CyberTiel GGUFs still contain the draft block (Saluki has none); it is only loaded with `--spec-type draft-mtp`. Without that flag llama-server skips the **whole extra block** (`blk.64` on the 27Bs, `blk.40` on CyberTiel) — not just the `nextn.*` tensors. Measured unused sizes: HIGH **215 MiB**, Huihui **810 MiB**, CyberTiel **371 MiB**. 24 GB is not enough for the head plus a useful KV cache.

## VRAM

```
RTX 5090 Laptop       24463 MiB
Desktop (Intel iGPU)  ~6 MiB on the 5090
fit-target            1024 MiB free
Rest                  weights + q8_0 KV, all GPU (-ngl 99)
```

`-c 184320 -ctk q8_0 -ctv q8_0`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0`, `--no-mmproj`. 35B-A3B MoE: whole trunk on the GPU (stock placement). **Nothing is meant to spill into host RAM for inference** — including the 27B dense models.

All four overlays pin `-ngl 99` and never pass `--n-cpu-moe`. With `-ngl` and `-c` both set, `--fit on` has nothing to adjust (llama.cpp logs `n_gpu_layers already set by user to 99, abort`), so `-c 184320` is simply allocated. The headroom is measured, not enforced: after load 1717 MiB free on the 27B NVFP4s and 737 MiB on CyberTiel; a single 167k-token prompt pushed the peak to 23260 MiB (HIGH) / 23858 MiB (CyberTiel) of 24463 without an OOM. Unified memory is off, so CUDA will not silently page VRAM into RAM. The compose cgroup is `mem_limit: 12g` with no container swap (`memswap_limit: 12g`).

llama.cpp still **mmaps** the GGUF (default). That is file-backed page cache of the weights file, not CPU offload. On a live load (SuperQwen, since removed from the stack) the GPU held **22691 MiB**; cgroup anon (actual process RAM) was **~366 MiB**; the rest of the 12g cap was reclaimable GGUF cache. `--no-mmap` would copy the whole file into anonymous RAM first and **OOM the 27Bs** against the 12g cap — leave mmap on.

Qwen3.8 only caches 16 of 64 layers (Gated Attention); the other 48 Gated DeltaNet layers keep a constant-size recurrent state. Native context is 262,144 tokens; the stack runs 184,320 (180k) so the `q8_0` KV cache (≈6 GiB for the 27Bs) leaves VRAM headroom on 24 GB.

## pi

[pi](https://github.com/earendil-works/pi) is the only client: one agent, no subagents, no parallel requests. One slot (`-np 1`) is therefore enough and the whole 184,320-token window belongs to it.

```bash
pi                          # default model from ~/.pi/agent/settings.json
pi --model local/local-qwen
```

`~/.pi/agent/models.json` defines the provider `local` (`http://127.0.0.1:8080/v1`, no key) with the model `local-qwen` for this stack. Every overlay serves that alias (`-a <name>,local-qwen`), so whichever container is up on :8080 answers — switch with `./llm qwen` / `qwen-uncensored` / `cybertiel` / `saluki` and pi needs no change. The same `local/local-qwen` entry also works against Strata when that is what is listening on :8080 — pi has no separate Strata provider any more.

| pi setting | Value | Meaning |
|---|---|---|
| `contextWindow` (models.json) | 184320 | Same as `-c 184320` (`n_ctx_slot`) |
| `maxTokens` (models.json) | 49152 | Cap per reply |
| `compaction.modelOverrides["local/local-qwen"].reserveTokens` (settings.json) | 14320 | Auto-compaction above 184320 − 14320 = **170000** tokens |

Keep `contextWindow` and `reserveTokens` in step with `-c`; the server log shows the real `n_ctx_slot`. pi sends `preserve_thinking` itself; the server-side `--reasoning-preserve` sets llama.cpp's `preserve_reasoning` kwarg, which Qwen3.8's template ignores.

Default thinking is **xhigh** (`--reasoning-effort xhigh` on the server). pi sends `reasoning_effort` and `enable_thinking` / `preserve_thinking` as chat-template kwargs; its `thinkingLevelMap` turns `high` / `xhigh` / `max` into `xhigh`. Qwen3.8 only accepts `low` / `medium` / `xhigh` — not `high`. pi sends no sampling parameters, so the overlay's `--temp` / `--top-p` / `--top-k` defaults apply.

## Speed

RTX 5090 Laptop (24463 MiB), llama.cpp `llamacpp:native` (ggml-org `master`, build `b11557`, commit `5c8854f`, CUDA 13.3.1, `sm_120a`), **2026-10-11**. Profile (all four): MTP off, `-ngl 99`, `-np 1`, `-c 184320`, `-ctk q8_0 -ctv q8_0`, `--flash-attn on`, thinking **xhigh**.

The image ships `llama-server` only — no `llama-bench`. The standard tool is [`./scripts/compare.sh`](scripts/compare.sh): same `/v1/chat/completions` path pi uses. `./scripts/compare.sh` runs all four; `./scripts/compare.sh qwen cybertiel` a subset. The older logs in `scripts/speed-results/` hold BeeLlama-era runs (`kvarn6`, `-c 262144`, 2–3 slots) and models that are no longer in the stack.

| | Qwen HIGH | Uncensored | CyberTiel | Saluki |
|---|---|---|---|---|
| NVFP4 | **yes** | **yes** | no (UD-Q4_K_XL) | no (IQ2-mix) |
| Decode | **37.7 t/s** | **37.4 t/s** | **155.6 t/s** | **52.9 t/s** |
| Prefill ~1.9k | 2438 t/s | 2369 t/s | 3704 t/s | 1407 t/s |
| VRAM after load | 22268 / 1717 free | 22268 / 1717 | 23248 / 737 | 14075 / 9910 |
| 167k-token prompt | 782 t/s prefill, 20.4 t/s decode, peak 23260 MiB | — | 2155 t/s prefill, 62.4 t/s decode, peak 23858 MiB | — |

Decode is generation t/s on a short prompt (mean of 2, `max_tokens=128`). Prefill is a ~1.9k-token summarize prompt. The 167k row is one prompt of ~167,500 tokens (llama.cpp sources) with `max_tokens=200`. The two 27B NVFP4 files use the same VRAM: llama.cpp keeps the input layer (`token_embd.weight`, Q6_K in HIGH, BF16 in Huihui) in mapped host memory, `output.weight` is BF16 in both, and the MTP block is never loaded, so the extra Huihui size does not reach the GPU. Saluki's VRAM is mostly the KV cache and compute buffers, not weights.

Reference BeeLlama numbers (`kvarn6`, `-c 262144`, 2026-10-02): HIGH 36.2 t/s, Uncensored 35.7 t/s, CyberTiel 137 t/s decode.

Summary and raw timings: [`summary-20261011-172718.txt`](scripts/speed-results/summary-20261011-172718.txt), `raw-20261011-172718.jsonl`. Earlier: HIGH / CyberTiel [`summary-qwen-20260916-031329.txt`](scripts/speed-results/summary-qwen-20260916-031329.txt), [`summary-cybertiel-20260916-031030.txt`](scripts/speed-results/summary-cybertiel-20260916-031030.txt); HIGH / Uncensored (2026-10-02) [`summary-20261002-222859.txt`](scripts/speed-results/summary-20261002-222859.txt).

## Build

`./llm build` sets `CMAKE_CUDA_ARCHITECTURES=120` (upstream rewrites it to `120a` for the FP4 tensor cores) and builds `llama-server` from the current ggml-org `master` commit. Image: CUDA **13.3.1**. Host driver 595.x speaks CUDA 13.2, so `NVIDIA_REQUIRE_CUDA=cuda>=13.2` (otherwise the 13.3.1 image will not start). `GGML_CUDA_FA_ALL_QUANTS=ON` compiles the full FlashAttention K/V pair matrix, so mixed cache types stay possible. `GGML_SCHED_MAX_COPIES=1` skips pipeline-parallel input copies (ggml default is 4; this is a single-GPU box). GPU access is via NVIDIA Container Toolkit (`--gpus all`).

```bash
docker run --rm --gpus all -e NVIDIA_REQUIRE_CUDA="cuda>=13.2" --entrypoint nvidia-smi llamacpp:native
```

## Links

- https://github.com/ggml-org/llama.cpp
- https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF
- https://huggingface.co/renketong/Huihui-Qwen3.8-27B-abliterated-NVFP4-GGUF
- https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP
- https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0
