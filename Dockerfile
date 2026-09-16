# Native BeeLlama.cpp for THIS machine.
#
#   GGML_NATIVE=ON          CPU backend with -march=native
#   CMAKE_CUDA_ARCHITECTURES  only the host GPU (RTX 5090 Laptop = sm_120)
#   GGML_CUDA_KVARN=ON      KVarN store + native CUDA FlashAttention
#   GGML_SCHED_MAX_COPIES=1 single-GPU: skip pipeline-parallel input copies (default 4)
#
# GGML_CUDA_FA_ALL_QUANTS compiles every standard pair and all 36 KVarN
# ordered pairs (needed for kvarn6/kvarn6 plus the F16 precision tail).
#
# --allow-shlib-undefined: CUDA VMM pool calls the driver API
# (cuMemAddressFree etc.). libcuda.so only exists at runtime.
#
# Host driver reports CUDA 13.2 (595.x). The 13.3.1 image default
# NVIDIA_REQUIRE_CUDA is cuda>=13.3 and would refuse to start. We override
# to cuda>=13.2. Use ./llm build — a plain `docker build` cannot resolve
# "native" CUDA architectures unless the daemon's default runtime is nvidia.

ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.3.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG LLAMACPP_REPO=https://github.com/Anbeeld/beellama.cpp
ARG LLAMACPP_REF=main
ARG CUDA_DOCKER_ARCH=native
ARG GGML_NATIVE=ON

RUN apt-get update && \
    apt-get install -y gcc-14 g++-14 build-essential cmake python3 git libssl-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /app
RUN git init -q . && \
    git remote add origin ${LLAMACPP_REPO} && \
    git fetch --depth 1 origin ${LLAMACPP_REF} && \
    git checkout -q FETCH_HEAD && \
    git log -1 --oneline

# SM120 dense-FP8 CUTLASS FFN does not compile with CUDA 13.3.1 + gcc-14
# (Base::Arguments is not a class-name). Served GGUFs are NVFP4 / Q4_K_M.
# BeeLlama may glob the .cu unconditionally; skip the file if present.
RUN if [ -f ggml/src/ggml-cuda/fp8-cutlass-ffn-sm120.cu ]; then \
        mv ggml/src/ggml-cuda/fp8-cutlass-ffn-sm120.cu \
           ggml/src/ggml-cuda/fp8-cutlass-ffn-sm120.cu.skip; \
    fi && \
    if grep -q 'CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8 AND' \
            ggml/src/ggml-cuda/CMakeLists.txt; then \
        sed -i 's/if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8 AND/if (FALSE AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8 AND/' \
            ggml/src/ggml-cuda/CMakeLists.txt; \
    fi

RUN if [ "${CUDA_DOCKER_ARCH}" = "native" ] && ! command -v nvidia-smi >/dev/null 2>&1; then \
        echo "ERROR: CUDA_DOCKER_ARCH=native, but no GPU is visible inside the build container." >&2; \
        echo "       Use ./llm build (it detects the compute capability on the host)," >&2; \
        echo "       or pass it yourself, e.g. --build-arg CUDA_DOCKER_ARCH=120." >&2; \
        exit 1; \
    fi && \
    cmake -B build \
        -DGGML_NATIVE=${GGML_NATIVE} \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH} \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_CUDA_KVARN=ON \
        -DGGML_SCHED_MAX_COPIES=1 \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS server

RUN apt-get update && \
    apt-get install -y libgomp1 curl && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /app/lib/ /app
COPY --from=build /app/build/bin/llama-server /app

# Host driver is CUDA 13.2; image default is cuda>=13.3.
ENV NVIDIA_REQUIRE_CUDA="cuda>=13.2"
ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_REASONING=on
ENV LLAMA_ARG_REASONING_EFFORT=xhigh
ENV LLAMA_ARG_REASONING_PRESERVE=1
WORKDIR /app
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]
ENTRYPOINT [ "/app/llama-server" ]
