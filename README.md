# Local LLMs — BeeLlama.cpp on the RTX 5090 Laptop (24 GB)

One image, six models, one port. Runs on the **NVIDIA GeForce RTX 5090 Laptop** (Blackwell `sm_120`, **24 GB VRAM**). The Intel Arrow Lake iGPU drives the desktop, so almost all 24 GB on the 5090 is available for inference.

Compose shows what is running:

```bash
docker compose ls
docker compose -f docker-compose.yml -f qwen.yml ps    # or qwen-uncensored.yml / superqwen.yml / cybertiel.yml / tiel.yml / genesis.yml
```

- **GPU:** RTX 5090 Laptop, 24 GB
- **CPU:** Intel Core Ultra 9 275HX, image built with `-march=native`
- **Engine:** [BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) (`kvarn6` / `kvarn6` KV cache)

Weights + KV **entirely in VRAM**, ~1 GB headroom (`--fit-target 1024`).

## Qwen HIGH — the quality model

The default is **Qwen3.8-27B NVFP4 HIGH** (`./llm qwen`). In the [esatapedico family](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF) the compact tiers share the same 448-tensor NVFP4 backbone; they differ only in the ten extra tensors (LM head, embeddings, MTP draft).

On **HIGH the LM head (`output.weight`) stays BF16** — the same type as in the source conversion (`ORIG`), so it is left unchanged. The LM head maps hidden states to the vocabulary; leaving it unquantized is the biggest quality lever on this ladder. HIGH should therefore be the highest-quality variant that still runs well on **24 GB**: 17.57 GB on disk (17.35 GB loaded; 215 MiB unused MTP), Blackwell NVFP4, and leftover VRAM for the KVarN KV cache.

Qwen HIGH is the accurate dense 27B. `./llm qwen-uncensored` is the Huihui abliterated NVFP4 sibling — same 27B dense layout, refusals stripped. `./llm superqwen` is SuperQwen3.8-27B abliterated **Q4_K_M** (text-only target; MTP draft and mmproj stay on Hugging Face). `./llm cybertiel` is Cyber-Tiel-Coder, the Huihui-abliterated Ornith-1.5 coder at **UD-Q4_K_XL** (22.75 GB on disk, 22.36 GB loaded; MTP off). `./llm tiel` is the guardrailed Sharp-template sibling at **UD-Q4_K_XL** (22.36 GB; no MTP head in the file). `./llm genesis` is Tiel-Coder Genesis Hermes **NVFP4 v4** (21.48 GB; no MTP in the file).

## Quick start

```bash
cd ~/LLM
./llm download
./llm build
./llm qwen              # NVFP4 HIGH on :8080
./llm qwen-uncensored   # Huihui abliterated NVFP4 on :8080
./llm superqwen         # SuperQwen abliterated Q4_K_M on :8080
./llm cybertiel         # Cyber-Tiel-Coder UD-Q4_K_XL on :8080
./llm tiel              # Tiel-Coder UD-Q4_K_XL on :8080
./llm genesis           # Tiel-Coder Genesis Hermes NVFP4 v4 on :8080
./llm stop
```

- API: `http://127.0.0.1:8080/v1` (no key) · UI: http://127.0.0.1:8080
- Names: `qwen3.8-27b` / `qwen3.8-27b-uncensored` / `superqwen3.8-27b` / `cyber-tiel-coder-35b` / `tiel-coder-35b` / `tiel-coder-35b-genesis`

Both think. The answer is in `message.content`, reasoning in `message.reasoning_content`.

## Layout

| | |
|---|---|
| `docker-compose.yml` | Image, GPU (`--gpus all`), port 8080, `NVIDIA_REQUIRE_CUDA` |
| `qwen.yml` / `qwen-uncensored.yml` / `superqwen.yml` / `cybertiel.yml` / `tiel.yml` / `genesis.yml` | Model, slots, sampling, KVarN — **tune here** |
| `llm` | `qwen` / `qwen-uncensored` / `superqwen` / `cybertiel` / `tiel` / `genesis` / `stop` / `build` / `download` |
| `Dockerfile` | Native BeeLlama build, CUDA **13.3.1** |
| `models/` | The GGUFs (not in git — `./llm download`) |
| `scripts/` | `compare.sh` (TPS via `/v1/chat/completions`) and `speed-results/` |

## Models

Disk is the GGUF on disk (decimal GB). **Loaded** is what BeeLlama maps without `--spec-type`: the extra MTP block is logged `unused tensor … ignoring` and never occupies VRAM. Measured 2026-09-16; tensor dump in [`scripts/speed-results/model-sizes-20260916.txt`](scripts/speed-results/model-sizes-20260916.txt).

| Overlay | What | NVFP4 | Disk | MTP in file | MTP (unloaded) | Loaded |
|---|---|---|---|---|---|---|
| `qwen` | Qwen3.8-27B NVFP4 **HIGH**, LM head BF16 | **yes** | 17.571 GB | yes, `blk.64` IQ4_XS | **215 MiB** | 17.345 GB |
| `qwen-uncensored` | Huihui abliterated NVFP4 | **yes** | 19.654 GB | yes, `blk.64` BF16 | **810 MiB** | 18.804 GB |
| `superqwen` | SuperQwen3.8-27B abliterated Q4_K_M | no (Q4_K_M) | 16.547 GB | **no** (sidecar 1.56 GiB not downloaded) | 0 | 16.547 GB |
| `cybertiel` | Cyber-Tiel-Coder 35B-A3B UD-Q4_K_XL | no (UD-Q4_K_XL) | 22.750 GB | yes, grafted `blk.40` Q3_K | **371 MiB** | 22.360 GB |
| `tiel` | Tiel-Coder 35B-A3B UD-Q4_K_XL, Sharp template | no (UD-Q4_K_XL) | 22.360 GB | **no** | 0 | 22.360 GB |
| `genesis` | Tiel-Coder Genesis Hermes NVFP4 v4 | **yes** | 21.475 GB | **no** | 0 | 21.475 GB |

Sources: [HIGH](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF) · [uncensored](https://huggingface.co/renketong/Huihui-Qwen3.8-27B-abliterated-NVFP4-GGUF) · [SuperQwen](https://huggingface.co/Jiunsong/SuperQwen3.8-27b-abliterated-GGUF) · [CyberTiel MTP](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP) · [Tiel](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) · [Genesis](https://huggingface.co/jan1k/Tiel-Coder-35B-A3B-Genesis-Hermes-NVFP4-GGUF). Files: `Qwen3.8-27B-NVFP4-MTP-HIGH.gguf`, `Qwen3.8-27B-huihui-NVFP4.gguf`, `SuperQwen3.8-27b-abliterated-Q4_K_M.gguf`, `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`, `Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf`, `Tiel-Coder-35B-A3B-Genesis-Hermes-NVFP4-v4.gguf`.

## No MTP

**MTP stays off.** Three of the six GGUFs still contain the draft block; it is only loaded with `--spec-type draft-mtp`. Without that flag llama-server skips the **whole extra block** (`blk.64` on the 27Bs, `blk.40` on CyberTiel) — not just the `nextn.*` tensors. Measured unused sizes: HIGH **215 MiB**, Huihui **810 MiB**, CyberTiel **371 MiB**. SuperQwen's draft is a separate 1.56 GiB GGUF we do not keep; Tiel and Genesis have no head (Genesis MTP sibling not downloaded). 24 GB is not enough for the head plus a useful KVarN cache.

## VRAM

```
RTX 5090 Laptop       24463 MiB
Desktop (Intel iGPU)  ~6 MiB on the 5090
fit-target            1024 MiB free
Rest                  weights + KVarN KV, all GPU (-ngl 99)
```

`-ctk kvarn6 -ctv kvarn6 --kv-tail-tokens 1024`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0`, `--no-mmproj`. 35B-A3B MoE: whole trunk on the GPU (stock placement). **Nothing is meant to spill into host RAM for inference** — including the 27B dense models.

All six overlays pin `-ngl 99` and never pass `--n-cpu-moe`. `--fit on` cannot drop layers (BeeLlama logs `n_gpu_layers already set by user to 99, abort`); it only shrinks `-c` so the KVarN pool plus scratch still leave `--fit-target 1024` MiB free. Unified memory is off, so CUDA will not silently page VRAM into RAM. The compose cgroup is `mem_limit: 12g` with no container swap (`memswap_limit: 12g`).

llama.cpp still **mmaps** the GGUF (default). That is file-backed page cache of the weights file, not CPU offload. On a live SuperQwen load the GPU held **22691 MiB**; cgroup anon (actual process RAM) was **~366 MiB**; the rest of the 12g cap was reclaimable GGUF cache. `--no-mmap` would copy the whole file into anonymous RAM first and **OOM the 27Bs** against the 12g cap — leave mmap on.

Qwen3.8 only caches 16 of 64 layers (Gated Attention); the other 48 Gated DeltaNet layers keep a constant-size recurrent state. Native context is 262,144 tokens. `--fit on` shrinks `-c` if the KVarN pool plus graph scratch would miss `--fit-target`.

## omp / pi

```bash
omp --model qwen-local/qwen3.8-27b
omp --model qwen-local/qwen3.8-27b-uncensored
omp --model qwen-local/superqwen3.8-27b
omp --model cybertiel-local/cyber-tiel-coder-35b
omp --model tiel-local/tiel-coder-35b
omp --model genesis-local/tiel-coder-35b-genesis
```

Set sampling in omp to `-1` so the YAML defaults apply. Default thinking is **xhigh** (`--reasoning-effort xhigh` on the server; `defaultThinkingLevel: xhigh` in omp and pi). Qwen3.8 only accepts `low` / `medium` / `xhigh` — not `high`.

## Speed

RTX 5090 Laptop (24463 MiB), BeeLlama.cpp `beellama:native`, **2026-09-16**. Profile (all six): MTP off, `-ngl 99`, `kvarn6`/`kvarn6`, `--kv-tail-tokens 1024`, `--flash-attn on`, `--fit-target 1024`, thinking **xhigh**. 27B dense uses 3 slots, 35B-A3B uses 2; native `-c 262144`.

The image ships `llama-server` only — no `llama-bench`. The standard tool is [`./scripts/compare.sh`](scripts/compare.sh): same `/v1/chat/completions` path omp/pi use. `./scripts/compare.sh` runs all six; `./scripts/compare.sh superqwen tiel genesis` a subset.

| | Qwen HIGH | Uncensored | SuperQwen | CyberTiel | Tiel | Genesis |
|---|---|---|---|---|---|---|
| NVFP4 | **yes** | **yes** | no (Q4_K_M) | no (UD-Q4_K_XL) | no (UD-Q4_K_XL) | **yes** |
| Decode | **36.2 t/s** | **35.6 t/s** | **34.8 t/s** | **137 t/s** | **141 t/s** | **157 t/s** |
| Prefill ~1.9k | 1646 t/s | 1595 t/s | 1111 t/s | 2528 t/s | 2530 t/s | 3206 t/s |
| VRAM after load | 23203 / 781 free | 23141 / 843 | 22691 / 1293 | 23403 / 581 | 23403 / 582 | 22123 / 1861 |
| Slots | 3 | 3 | 3 | 2 | 2 | 2 |

Decode is generation t/s on a short prompt (mean of 2, `max_tokens=128`). Prefill is a ~1.9k-token summarize prompt. `--fit-target 1024` fills leftover VRAM with the KVarN pool, so used-MiB is **not** the weight size — that is the Loaded column above.

HIGH / CyberTiel: [`summary-qwen-20260916-031329.txt`](scripts/speed-results/summary-qwen-20260916-031329.txt), [`summary-cybertiel-20260916-031030.txt`](scripts/speed-results/summary-cybertiel-20260916-031030.txt). Uncensored / SuperQwen / Tiel: [`summary-20260916-135246.txt`](scripts/speed-results/summary-20260916-135246.txt). Genesis: [`summary-20260916-141519.txt`](scripts/speed-results/summary-20260916-141519.txt).

## Build

`./llm build` sets `CMAKE_CUDA_ARCHITECTURES=120`. Image: CUDA **13.3.1**. Host driver 595.x speaks CUDA 13.2, so `NVIDIA_REQUIRE_CUDA=cuda>=13.2` (otherwise the 13.3.1 image will not start). `GGML_CUDA_KVARN=ON` and `GGML_CUDA_FA_ALL_QUANTS=ON` compile KVarN plus the full FlashAttention pair matrix. `GGML_SCHED_MAX_COPIES=1` skips pipeline-parallel input copies (ggml default is 4; this is a single-GPU box). GPU access is via NVIDIA Container Toolkit (`--gpus all`).

```bash
docker run --rm --gpus all -e NVIDIA_REQUIRE_CUDA="cuda>=13.2" --entrypoint nvidia-smi beellama:native
```

## Links

- https://github.com/Anbeeld/beellama.cpp
- https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF
- https://huggingface.co/renketong/Huihui-Qwen3.8-27B-abliterated-NVFP4-GGUF
- https://huggingface.co/Jiunsong/SuperQwen3.8-27b-abliterated-GGUF
- https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP
- https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF
- https://huggingface.co/jan1k/Tiel-Coder-35B-A3B-Genesis-Hermes-NVFP4-GGUF
- https://omp.sh
