#!/usr/bin/env bash
# Speed check through the same /v1/chat/completions path omp/pi use.
# MTP off. Uses the live compose overlays on port 8080 (the stack is stopped).
#
#   ./scripts/compare.sh                         # bee + bun + main, all models
#   ./scripts/compare.sh bee                     # BeeLlama kvarn6 only
#   ./scripts/compare.sh bun                     # buun VBR only
#   ./scripts/compare.sh main                    # llama.cpp master q5_1 only
#   ./scripts/compare.sh superqwen tiel-bun      # subset of overlay names
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

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

# overlay → compose file, API alias
declare -A YML ALIAS
register() {
    local label="$1" yml="$2" alias="$3"
    YML[$label]="$yml"
    ALIAS[$label]="$alias"
}
BEE_MODELS=(qwen qwen-uncensored superqwen cybertiel tiel genesis)
register qwen              qwen.yml              qwen3.8-27b
register qwen-uncensored   qwen-uncensored.yml   qwen3.8-27b-uncensored
register superqwen         superqwen.yml         superqwen3.8-27b
register cybertiel         cybertiel.yml         cyber-tiel-coder-35b
register tiel              tiel.yml              tiel-coder-35b
register genesis           genesis.yml           tiel-coder-35b-genesis
for m in "${BEE_MODELS[@]}"; do
    register "${m}-bun"  "${m}-bun.yml"  "${ALIAS[$m]}"
    register "${m}-main" "${m}-main.yml" "${ALIAS[$m]}"
done
ALL_MODELS=()
for m in "${BEE_MODELS[@]}"; do
    ALL_MODELS+=("$m" "${m}-bun" "${m}-main")
done

RESULTS_DIR="$ROOT/scripts/speed-results"
mkdir -p "$RESULTS_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY="$RESULTS_DIR/summary-$STAMP.txt"
RAW_JSONL="$RESULTS_DIR/raw-$STAMP.jsonl"

log() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }

down_all() {
    local y names=()
    for y in *.yml; do
        [[ "$y" == docker-compose.yml ]] && continue
        docker compose -f docker-compose.yml -f "$y" down --remove-orphans --timeout 20 >/dev/null 2>&1 || true
        names+=("${y%.yml}")
    done
    docker rm -f "${names[@]}" ornith llm-compare >/dev/null 2>&1 || true
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
data = json.load(sys.stdin)
t = data.get("timings") or {}
u = (data.get("usage") or {}).get("completion_tokens_details") or {}
print(
    "prompt {pn} tok @ {pps:.1f} t/s, gen {gn} tok @ {gps:.1f} t/s "
    "(prompt {pms:.0f} ms, gen {gms:.0f} ms; reasoning={rt}, visible={vt})".format(
        pn=t.get("prompt_n") or 0,
        pps=t.get("prompt_per_second") or 0.0,
        gn=t.get("predicted_n") or 0,
        gps=t.get("predicted_per_second") or 0.0,
        pms=t.get("prompt_ms") or 0.0,
        gms=t.get("predicted_ms") or 0.0,
        rt=u.get("reasoning_tokens", "?"),
        vt=u.get("visible_tokens", "?"),
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
    local label="$1"
    local yml="${YML[$label]}"
    local alias="${ALIAS[$label]}"

    log ""
    log "=== ${label} ==="
    log "overlay: ${yml}"
    log "alias: ${alias}"

    down_all
    sleep 2

    local logf="$RESULTS_DIR/${label}-$STAMP.log"
    if ! docker compose -f docker-compose.yml -f "$yml" up -d --wait --wait-timeout "$HEALTH_TIMEOUT"; then
        docker compose -f docker-compose.yml -f "$yml" logs >"$logf" 2>&1 || true
        log "FAIL: container did not become healthy"
        tail -40 "$logf" | tee -a "$SUMMARY" || true
        down_all
        return 1
    fi
    docker compose -f docker-compose.yml -f "$yml" logs >"$logf" 2>&1

    log "vram_after_load_mib (used,total,free): $(gpu_mem)"
    grep -E 'unused tensor|KVarN|kvarn|common_fit|n_ctx_slot|n_slots|cache type|kv-tail|VBR |vbr|q8_0|q5_1|type_k|type_v' "$logf" \
        | grep -v 'unused tensor' | head -30 | tee -a "$SUMMARY" || true
    python3 - "$logf" >>"$SUMMARY" <<'PY' || true
import re, sys
s = n = 0
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"unused tensor (\S+) \(size = (\d+) bytes\)", line)
    if m:
        n += 1
        s += int(m.group(2))
print(f"unused_mtp_tensors: {n}  unused_mtp_bytes: {s} ({s/1024**2:.1f} MiB)")
PY

    local tmp
    tmp=$(mktemp)
    trap 'rm -f "$tmp"' RETURN

    if ! chat "$alias" "Hello." "$WARMUP_MAX_TOKENS" >"$tmp"; then
        log "FAIL: warmup request"
        down_all
        return 1
    fi
    record_raw "$label" warmup 1 "$tmp"
    log "  warmup: $(fmt_timings <"$tmp")"

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
}

expand_engine() {
    local engine="$1" m
    case "$engine" in
        bee)  for m in "${BEE_MODELS[@]}"; do printf '%s\n' "$m"; done ;;
        bun)  for m in "${BEE_MODELS[@]}"; do printf '%s\n' "${m}-bun"; done ;;
        main) for m in "${BEE_MODELS[@]}"; do printf '%s\n' "${m}-main"; done ;;
        *)    return 1 ;;
    esac
}

MODELS=()
if [[ $# -eq 0 ]]; then
    MODELS=("${ALL_MODELS[@]}")
else
    for m in "$@"; do
        if expand_engine "$m" >/dev/null; then
            while IFS= read -r x; do MODELS+=("$x"); done < <(expand_engine "$m")
        elif [[ -n "${YML[$m]:-}" ]]; then
            MODELS+=("$m")
        else
            echo "unknown model: $m (want: bee|bun|main or ${ALL_MODELS[*]})" >&2
            exit 1
        fi
    done
fi

image_line() {
    local img="$1"
    if docker image inspect "$img" >/dev/null 2>&1; then
        docker image inspect "$img" --format '{{.Id}} {{.Created}}'
    else
        echo "(missing)"
    fi
}

: >"$SUMMARY"
log "engines: bee (kvarn6) / bun (VBR) / main (q5_1)   models: ${MODELS[*]}"
log "stamp: $STAMP"
log "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
log "beellama:native: $(image_line beellama:native)"
log "bun:native:      $(image_line bun:native)"
log "llamacpp:native: $(image_line llamacpp:native)"
log "decode prompt max_tokens=${DECODE_MAX_TOKENS} repeats=${DECODE_REPEATS}"
log "MTP off. --fit-target 1024. Same /v1/chat/completions path as omp/pi."
log "thinking: server default (xhigh)."

for m in "${MODELS[@]}"; do
    run_config "$m" || log "SKIP/FAIL $m"
done

down_all
log ""
if [[ -s "$RAW_JSONL" ]]; then
    python3 - "$RAW_JSONL" <<'PY' | tee -a "$SUMMARY"
import json, sys
from collections import defaultdict
rows = defaultdict(dict)
for line in open(sys.argv[1]):
    rec = json.loads(line)
    t = rec.get("timings") or {}
    key = rec["config"]
    phase = rec["phase"]
    gps = float(t.get("predicted_per_second") or 0)
    pps = float(t.get("prompt_per_second") or 0)
    pn = int(t.get("prompt_n") or 0)
    if phase == "decode":
        rows[key].setdefault("decode_gen", []).append(gps)
        rows[key].setdefault("decode_prefill", []).append(pps)
    elif phase == "prefill":
        rows[key]["prefill_tok"] = pn
        rows[key]["prefill_t/s"] = pps
        rows[key]["prefill_gen"] = gps

def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0

print()
print("=== table (decode gen t/s = mean of repeats; prefill = long prompt) ===")
print(f"{'config':<28} {'decode t/s':>10} {'prefill t/s':>12} {'prefill tok':>12} {'prefill-gen':>12}")
for key in sorted(rows):
    r = rows[key]
    print(f"{key:<28} {avg(r.get('decode_gen', [])):>10.2f} {r.get('prefill_t/s', 0):>12.1f} {r.get('prefill_tok', 0):>12d} {r.get('prefill_gen', 0):>12.1f}")
PY
fi
log "raw timings: $RAW_JSONL"
log "summary: $SUMMARY"
log "done."
echo
echo "Summary: $SUMMARY"
echo "Stack is down. Start with ./llm qwen  (or qwen-bun / qwen-main, …)"
