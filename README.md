# Local LLMs — BeeLlama.cpp on the RTX 5090 Laptop (24 GB)

One image, five models, one port. Runs on the **NVIDIA GeForce RTX 5090 Laptop** (Blackwell `sm_120`, **24 GB VRAM**). The Intel Arrow Lake iGPU drives the desktop, so almost all 24 GB on the 5090 is available for inference.

Compose shows what is running:

```bash
docker compose ls
docker compose -f docker-compose.yml -f qwen.yml ps    # or qwen-uncensored.yml / superqwen.yml / cybertiel.yml / tiel.yml
```

- **GPU:** RTX 5090 Laptop, 24 GB
- **CPU:** Intel Core Ultra 9 275HX, image built with `-march=native`
- **Engine:** [BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) (`kvarn6` / `kvarn6` KV cache)

Weights + KV **entirely in VRAM**, ~1 GB headroom (`--fit-target 1024`).

## Qwen HIGH — the quality model

The default is **Qwen3.8-27B NVFP4 HIGH** (`./llm qwen`). In the [esatapedico family](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF) the compact tiers share the same 448-tensor NVFP4 backbone; they differ only in the ten extra tensors (LM head, embeddings, MTP draft).

On **HIGH the LM head (`output.weight`) stays BF16** — the same type as in the source conversion (`ORIG`), so it is left unchanged. The LM head maps hidden states to the vocabulary; leaving it unquantized is the biggest quality lever on this ladder. HIGH should therefore be the highest-quality variant that still runs well on **24 GB**: 17.57 GB weights, Blackwell NVFP4, and leftover VRAM for the KVarN KV cache.

Qwen HIGH is the accurate dense 27B. `./llm qwen-uncensored` is the Huihui abliterated NVFP4 sibling — same 27B dense layout, refusals stripped. `./llm superqwen` is SuperQwen3.8-27B abliterated **Q4_K_M** (text-only target; MTP draft and mmproj stay on Hugging Face). `./llm cybertiel` is Cyber-Tiel-Coder, the Huihui-abliterated Ornith-1.5 coder at **UD-Q4_K_XL** (22.7 GB; MTP off). `./llm tiel` is the guardrailed Sharp-template sibling at **UD-Q4_K_XL** (22.4 GB; no MTP head in the file).

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
./llm stop
```

- API: `http://127.0.0.1:8080/v1` (no key) · UI: http://127.0.0.1:8080
- Names: `qwen3.8-27b` / `qwen3.8-27b-uncensored` / `superqwen3.8-27b` / `cyber-tiel-coder-35b` / `tiel-coder-35b`

Both think. The answer is in `message.content`, reasoning in `message.reasoning_content`.

## Layout

| | |
|---|---|
| `docker-compose.yml` | Image, GPU (`--gpus all`), port 8080, `NVIDIA_REQUIRE_CUDA` |
| `qwen.yml` / `qwen-uncensored.yml` / `superqwen.yml` / `cybertiel.yml` / `tiel.yml` | Model, slots, sampling, KVarN — **tune here** |
| `llm` | `qwen` / `qwen-uncensored` / `superqwen` / `cybertiel` / `tiel` / `stop` / `build` / `download` |
| `Dockerfile` | Native BeeLlama build, CUDA **13.3.1** |
| `models/` | The GGUFs (not in git — `./llm download`) |
| `scripts/` | Benchmarks (`compare.sh`, `speed-results/`) |

## Models

| | Qwen NVFP4 **HIGH** | Qwen uncensored | SuperQwen | CyberTiel | Tiel |
|---|---|---|---|---|---|
| File | `Qwen3.8-27B-NVFP4-MTP-HIGH.gguf` (17.57 GB) | `Qwen3.8-27B-huihui-NVFP4.gguf` (19.65 GB) | `SuperQwen3.8-27b-abliterated-Q4_K_M.gguf` (15.41 GiB) | `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf` (22.7 GB) | `Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf` (22.4 GB) |
| | NVFP4 backbone, **LM head BF16** | Huihui abliterated NVFP4 | Abliterated Q4_K_M | Abliterated Ornith-1.5 coder UD-Q4_K_XL | Sharp-template Ornith-1.5 coder UD-Q4_K_XL, no MTP head |
| Slots | 3 | 3 | 3 | 2 | 2 |
| Source | [esatapedico/…](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF) | [renketong/…](https://huggingface.co/renketong/Huihui-Qwen3.8-27B-abliterated-NVFP4-GGUF) | [Jiunsong/…](https://huggingface.co/Jiunsong/SuperQwen3.8-27b-abliterated-GGUF) | [peculiar-ragdoll/…](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP) | [peculiar-ragdoll/…](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) |

## No MTP

**MTP stays off.** Most GGUFs still contain the draft head; it is only loaded with `--spec-type draft-mtp`. 24 GB is not enough: the head costs extra GB and KVarN needs the rest for the cache. Without `--spec-type`, `nextn` tensors stay unloaded. Tiel's GGUF has no head.

## VRAM

```
RTX 5090 Laptop       24463 MiB
Desktop (Intel iGPU)  ~6 MiB on the 5090
fit-target            1024 MiB free
Rest                  weights + KVarN KV, all GPU (-ngl 99)
```

`-ctk kvarn6 -ctv kvarn6 --kv-tail-tokens 1024`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0`, `--no-mmproj`. 35B-A3B MoE: whole trunk on the GPU (stock placement).

Qwen3.8 only caches 16 of 64 layers (Gated Attention); the other 48 Gated DeltaNet layers keep a constant-size recurrent state. Native context is 262,144 tokens. `--fit on` shrinks `-c` if the KVarN pool plus graph scratch would miss `--fit-target`.

## omp / pi

```bash
omp --model qwen-local/qwen3.8-27b
omp --model qwen-local/qwen3.8-27b-uncensored
omp --model qwen-local/superqwen3.8-27b
omp --model cybertiel-local/cyber-tiel-coder-35b
omp --model tiel-local/tiel-coder-35b
```

Set sampling in omp to `-1` so the YAML defaults apply. Default thinking is **xhigh** (`--reasoning-effort xhigh` on the server; `defaultThinkingLevel: xhigh` in omp and pi). Qwen3.8 only accepts `low` / `medium` / `xhigh` — not `high`.

## Performance

`./scripts/compare.sh` compares Qwen NVFP4 and CyberTiel. `--fit-target 1024` leaves ~1 GB free; leftover VRAM is the KVarN cache.

Previous buun-llama-cpp VBR run (2026-08-23, `7d30a72`) for reference only — this tree now serves BeeLlama KVarN:

| | Qwen 27B dense | Ornith 35B-A3B MoE |
|---|---|---|
| Decode | **38.7 t/s** | **199.5 t/s** |
| Prefill ~1.9k | 2331 t/s | 5010 t/s |
| VRAM after load | 16553 / 24463 MiB | 19049 / 24463 MiB |
| VBR KV (auto) | 8389 MiB | 5755 MiB |

Raw data: `scripts/speed-results/summary-20260823-141820.txt`

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
- https://omp.sh
