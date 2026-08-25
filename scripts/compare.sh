#!/usr/bin/env bash
# Sequential Qwen vs Ornith speed check. MTP off. Uses the same compose
# overlays as ./llm qwen|ornith, on port 8080 (the live stack is stopped).
#
#   ./scripts/compare.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE=buun-llama:native
PORT=8080
HEALTH_TIMEOUT=240
WARMUP_MAX_TOKENS=16
DECODE_MAX_TOKENS=128
PREFILL_MAX_TOKENS=64
DECODE_REPEATS=2

DECODE_PROMPT='Write a numbered list of eight common programming languages. One short line each. No extra commentary.'
PREFILL_PROMPT="$(python3 - <<'PY'
body = (
    "The gated-delta hybrid in Qwen3.8 keeps most layers cheap and only "
    "every fourth block uses full attention. That makes long context cheap "
    "in VRAM, which is why KV-cache quantization is the interesting knob.\n"
)
print("Summarize the following notes in two sentences.\n\n" + body * 40)
PY
)"

RESULTS_DIR="$ROOT/scripts/speed-results"
mkdir -p "$RESULTS_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY="$RESULTS_DIR/summary-$STAMP.txt"
RAW_JSONL="$RESULTS_DIR/raw-$STAMP.jsonl"

log() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }

down_all() {
    local y
    for y in qwen.yml qwen-uncensored.yml superqwen.yml ornith.yml; do
        docker compose -f docker-compose.yml -f "$y" down --remove-orphans --timeout 20 >/dev/null 2>&1 || true
    done
    docker rm -f qwen qwen-uncensored superqwen ornith llm-compare >/dev/null 2>&1 || true
}

gpu_mem() {
    nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits | head -1
}

chat() {
    local model="$1" prompt="$2" max_tokens="$3"
    python3 - "$model" "$prompt" "$max_tokens" "$PORT" <<'PY'
import json, sys, urllib.request
model, prompt, max_tokens, port = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=600) as resp:
    sys.stdout.write(resp.read().decode())
PY
}

fmt_timings() {
    python3 -c '
import json, sys
t = (json.load(sys.stdin).get("timings") or {})
print(
    "prompt {pn} tok @ {pps:.1f} t/s, gen {gn} tok @ {gps:.1f} t/s "
    "(prompt {pms:.0f} ms, gen {gms:.0f} ms)".format(
        pn=t.get("prompt_n") or 0,
        pps=t.get("prompt_per_second") or 0.0,
        gn=t.get("predicted_n") or 0,
        gps=t.get("predicted_per_second") or 0.0,
        pms=t.get("prompt_ms") or 0.0,
        gms=t.get("predicted_ms") or 0.0,
    )
)
'
}

record_raw() {
    local label="$1" phase="$2" rep="$3" file="$4"
    python3 -c '
import json, sys
label, phase, rep, path = sys.argv[1:5]
data = json.load(open(path))
json.dump({
    "config": label,
    "phase": phase,
    "rep": int(rep),
    "timings": data.get("timings") or {},
    "usage": data.get("usage") or {},
}, sys.stdout)
print()
' "$label" "$phase" "$rep" "$file" >>"$RAW_JSONL"
}

run_config() {
    local label="$1" yml="$2" alias="$3"

    log ""
    log "=== ${label} ==="
    log "overlay: ${yml}"
    log "alias: ${alias}"

    down_all
    sleep 2

    local logf="$RESULTS_DIR/${label}-$STAMP.log"
    docker compose -f docker-compose.yml -f "$yml" up -d --wait --wait-timeout "$HEALTH_TIMEOUT"
    docker compose -f docker-compose.yml -f "$yml" logs >"$logf" 2>&1

    log "vram_after_load_mib (used,total,free): $(gpu_mem)"
    grep -E 'VBR |common_fit|MoE cache fit|n_ctx_slot|n_slots' "$logf" | head -20 | tee -a "$SUMMARY" || true

    local tmp
    tmp=$(mktemp)
    trap 'rm -f "$tmp"' RETURN

    if ! chat "$alias" "Hello." "$WARMUP_MAX_TOKENS" >"$tmp"; then
        log "FAIL: warmup request"
        down_all
        return 1
    fi
    record_raw "$label" warmup 1 "$tmp"

    local i gen_sum=0 prefill_sum=0
    for i in $(seq 1 "$DECODE_REPEATS"); do
        if ! chat "$alias" "$DECODE_PROMPT" "$DECODE_MAX_TOKENS" >"$tmp"; then
            log "FAIL: decode request $i"
            down_all
            return 1
        fi
        record_raw "$label" decode "$i" "$tmp"
        log "  decode[$i]: $(fmt_timings <"$tmp")"
        gen_sum=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1])).get("timings") or {}; print(float(sys.argv[2])+float(t.get("predicted_per_second") or 0))' "$tmp" "$gen_sum")
        prefill_sum=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1])).get("timings") or {}; print(float(sys.argv[2])+float(t.get("prompt_per_second") or 0))' "$tmp" "$prefill_sum")
    done
    log "  decode_avg_gen_t/s: $(python3 -c "print(round($gen_sum/$DECODE_REPEATS, 2))")"
    log "  decode_avg_prefill_t/s: $(python3 -c "print(round($prefill_sum/$DECODE_REPEATS, 2))")"

    if ! chat "$alias" "$PREFILL_PROMPT" "$PREFILL_MAX_TOKENS" >"$tmp"; then
        log "FAIL: prefill request"
        down_all
        return 1
    fi
    record_raw "$label" prefill 1 "$tmp"
    log "  long-prefill: $(fmt_timings <"$tmp")"
    log "vram_after_requests_mib (used,total,free): $(gpu_mem)"

    down_all
    sleep 3
}

: >"$SUMMARY"
log "buun-llama-cpp  Qwen3.8-27B NVFP4 HIGH vs Ornith-1.5 35B-A3B  (no MTP)"
log "stamp: $STAMP"
log "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
log "image: $IMAGE"
log "decode prompt max_tokens=${DECODE_MAX_TOKENS} repeats=${DECODE_REPEATS}"
log "MTP: off. --vbr-vram auto + --fit-target 1024: leftover VRAM goes to KV."

run_config qwen qwen.yml qwen3.8-27b || log "SKIP/FAIL qwen"
run_config ornith ornith.yml ornith-1.5-35b || log "SKIP/FAIL ornith"

log ""
log "raw timings: $RAW_JSONL"
log "summary: $SUMMARY"
log "done."
echo
echo "Summary: $SUMMARY"
echo "Stack is down. Start with ./llm qwen | qwen-uncensored | superqwen | ornith"
