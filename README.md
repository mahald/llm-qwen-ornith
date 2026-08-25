# Local LLMs — buun-llama-cpp on the RTX 5090 Laptop (24 GB)

One image, four models, one port. Runs on the **NVIDIA GeForce RTX 5090 Laptop** (Blackwell `sm_120`, **24 GB VRAM**). The Intel Arrow Lake iGPU drives the desktop, so almost all 24 GB on the 5090 is available for inference.

Compose shows what is running:

```bash
docker compose ls
docker compose -f docker-compose.yml -f qwen.yml ps    # or qwen-uncensored.yml / superqwen.yml / ornith.yml
```

- **GPU:** RTX 5090 Laptop, 24 GB
- **CPU:** Intel Core Ultra 9 275HX, image built with `-march=native`
- **Engine:** [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) only (`7d30a72`)

Weights + KV **entirely in VRAM**, ~1 GB headroom (`--fit-target 1024`).

## Qwen HIGH — the quality model

The default is **Qwen3.8-27B NVFP4 HIGH** (`./llm qwen`). In the [esatapedico family](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF) the compact tiers share the same 448-tensor NVFP4 backbone; they differ only in the ten extra tensors (LM head, embeddings, MTP draft).

On **HIGH the LM head (`output.weight`) stays BF16** — the same type as in the source conversion (`ORIG`), so it is left unchanged. The LM head maps hidden states to the vocabulary; leaving it unquantized is the biggest quality lever on this ladder. HIGH should therefore be the highest-quality variant that still runs well on **24 GB**: 17.57 GB weights, Blackwell NVFP4, and enough leftover VRAM for the VBR KV cache.

Ornith (35B-A3B MoE) is the fast second model (~200 t/s decode); Qwen HIGH is the accurate one. `./llm qwen-uncensored` is the Huihui abliterated NVFP4 sibling — same 27B dense layout, refusals stripped. `./llm superqwen` is SuperQwen3.8-27B abliterated **Q4_K_M** (text-only target; MTP draft and mmproj stay on Hugging Face).

## Quick start

```bash
cd ~/LLM
./llm download
./llm build
./llm qwen              # NVFP4 HIGH on :8080
./llm qwen-uncensored   # Huihui abliterated NVFP4 on :8080
./llm superqwen         # SuperQwen abliterated Q4_K_M on :8080
./llm ornith            # Ornith on :8080
./llm stop
```

- API: `http://127.0.0.1:8080/v1` (no key) · UI: http://127.0.0.1:8080
- Names: `qwen3.8-27b` / `qwen3.8-27b-uncensored` / `superqwen3.8-27b` / `ornith-1.5-35b`

Both think. The answer is in `message.content`, reasoning in `message.reasoning_content`.

## Layout

| | |
|---|---|
| `docker-compose.yml` | Image, GPU (`--gpus all`), port 8080, `NVIDIA_REQUIRE_CUDA` |
| `qwen.yml` / `qwen-uncensored.yml` / `superqwen.yml` / `ornith.yml` | Model, slots, sampling, VBR — **tune here** |
| `llm` | `qwen` / `qwen-uncensored` / `superqwen` / `ornith` / `stop` / `build` / `download` |
| `Dockerfile` | Native build, CUDA **13.3.1** |
| `models/` | The GGUFs (not in git — `./llm download`) |
| `scripts/` | Benchmarks (`compare.sh`, `speed-results/`) |

## Models

| | Qwen NVFP4 **HIGH** | Qwen uncensored | SuperQwen | Ornith |
|---|---|---|---|---|
| File | `Qwen3.8-27B-NVFP4-MTP-HIGH.gguf` (17.57 GB) | `Qwen3.8-27B-huihui-NVFP4.gguf` (19.65 GB) | `SuperQwen3.8-27b-abliterated-Q4_K_M.gguf` (15.41 GiB) | `Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf` (20.2 GB) |
| | NVFP4 backbone, **LM head BF16** | Huihui abliterated NVFP4 | Abliterated Q4_K_M | 35B-A3B MoE NVFP4 |
| Slots | 3 | 3 | 3 | 2 |
| Source | [esatapedico/…](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF) | [renketong/…](https://huggingface.co/renketong/Huihui-Qwen3.8-27B-abliterated-NVFP4-GGUF) | [Jiunsong/…](https://huggingface.co/Jiunsong/SuperQwen3.8-27b-abliterated-GGUF) | [Avifenesh/…](https://huggingface.co/Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF) |

## No MTP

**MTP stays off.** The GGUFs still contain the draft head; it is only loaded with `--spec-type draft-mtp`. 24 GB is not enough: the head costs several GB, Ornith is already tight, and VBR needs the rest for the cache. Without `--spec-type`, `nextn` tensors stay unloaded.

## VRAM

```
RTX 5090 Laptop       24463 MiB
Desktop (Intel iGPU)  ~6 MiB on the 5090
fit-target            1024 MiB free
Rest                  weights + VBR KV, all GPU (-ngl 99)
```

`--vbr-vram auto`, `-ct vbr` (full ladder down to turbo1_tcq), `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0`, `--no-mmproj`. Ornith: whole trunk on the GPU (MoE stock placement).

## omp / pi

```bash
omp --model qwen-local/qwen3.8-27b
omp --model qwen-local/qwen3.8-27b-uncensored
omp --model qwen-local/superqwen3.8-27b
omp --model ornith-local/ornith-1.5-35b
```

Set sampling in omp to `-1` so the YAML defaults apply.

## Performance

`./scripts/compare.sh` compares Qwen NVFP4 and Ornith. `--vbr-vram auto` + `--fit-target 1024` put leftover VRAM into the KV cache.

Run 2026-08-23, buun `7d30a72`:

| | Qwen 27B dense | Ornith 35B-A3B MoE |
|---|---|---|
| Decode | **38.7 t/s** | **199.5 t/s** |
| Prefill ~1.9k | 2331 t/s | 5010 t/s |
| VRAM after load | 16553 / 24463 MiB | 19049 / 24463 MiB |
| VBR KV (auto) | 8389 MiB | 5755 MiB |

Raw data: `scripts/speed-results/summary-20260823-141820.txt`

## Build

`./llm build` sets `CMAKE_CUDA_ARCHITECTURES=120`. Image: CUDA **13.3.1**. Host driver 595.84 speaks CUDA 13.2, so `NVIDIA_REQUIRE_CUDA=cuda>=13.2` (otherwise the 13.3.1 image will not start). `GGML_CUDA_FA_ALL_QUANTS=ON` is required for VBR/TCQ. GPU access is via NVIDIA Container Toolkit (`--gpus all`).

```bash
docker run --rm --gpus all -e NVIDIA_REQUIRE_CUDA="cuda>=13.2" --entrypoint nvidia-smi buun-llama:native
```

## Links

- https://github.com/spiritbuun/buun-llama-cpp
- https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF
- https://huggingface.co/renketong/Huihui-Qwen3.8-27B-abliterated-NVFP4-GGUF
- https://huggingface.co/Jiunsong/SuperQwen3.8-27b-abliterated-GGUF
- https://huggingface.co/Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF
- https://omp.sh
