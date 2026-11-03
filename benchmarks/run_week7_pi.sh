#!/usr/bin/env bash
# Week 7 Pi benchmarks (Requirements v0.3.7): is the ONNX or INT8 separator faster on the Pi 4?
#   1. Alone, k = 1: PyTorch vs ONNX FP32 vs ONNX INT8 (1 thread on core 3; ONNX also at 4 threads)
#   2. Two-core S3 layout, k = 1: SED on cores 0-1 (one window every 2 s) while USS runs on core 3
#   3. Two-core S3 layout, k = 2 (engine + jackhammer, the class-aware setting) with the fastest backend
#   4. Which INT8 operators ONNX Runtime really runs on this CPU (needs the onnx package; skipped otherwise)
# About 3-4 hours including cool-downs. Run from the benchmarks folder on the Pi, inside the venv:
#   nohup bash run_week7_pi.sh > week7_pi.log 2>&1 &
# Step 3 uses the fastest valid variant from step 2. To choose it yourself:
#   FAST=fp32 nohup bash run_week7_pi.sh > week7_pi.log 2>&1 &      (torch, fp32 or int8)
# INT8 is eligible: O1 is met in the class-aware configuration (Requirements v0.3.7, O1 Week 6 results;
# server report results/o1_frozen_int8_v2.json from experiments/o1_frozen.py, INT8 sha256 6307849497...).
# Needs in this folder: resunet30_at_soft_fp32.pth, resunet30_at_soft_fp32.onnx,
# resunet30_at_soft_int8_v2.onnx, query_conditions/{engine,jackhammer,beeper}.npy, bench_clip_32k_mono.wav
set -e
PY="python -u"
SED_PID=""
cleanup() { [ -n "$SED_PID" ] && kill "$SED_PID" 2>/dev/null || true; rm -f s3_ready; }
trap cleanup EXIT
FP32=resunet30_at_soft_fp32.onnx
INT8=resunet30_at_soft_int8_v2.onnx
CONDS=query_conditions

echo "== Checks =="
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c
for f in resunet30_at_soft_fp32.pth $FP32 $INT8 $CONDS/engine.npy $CONDS/jackhammer.npy $CONDS/beeper.npy \
         bench_clip_32k_mono.wav; do
  [ -f "$f" ] || { echo "Missing $f"; exit 1; }
done
$PY -c "import onnxruntime as o; print('onnxruntime', o.__version__)"
if pgrep -f benchmark_pi.py > /dev/null; then
  echo "Another benchmark_pi.py is still running:"; pgrep -af benchmark_pi.py; exit 1
fi

# backend arguments per variant
args_for() {
  case "$1" in
    torch) echo "--backend torch --conditions_dir $CONDS" ;;
    fp32)  echo "--backend onnx --onnx $FP32 --conditions_dir $CONDS" ;;
    int8)  echo "--backend onnx --onnx $INT8 --conditions_dir $CONDS" ;;
    *)     echo "unknown variant '$1' (use torch, fp32 or int8)" >&2; exit 1 ;;
  esac
}

echo "== 1. Alone, k = 1 =="
for V in torch fp32; do
  $PY benchmark_pi.py --stage sep $(args_for $V) --threads 1 --cpus 3 --k 1 --tag w7alone
done
$PY benchmark_pi.py --stage sep $(args_for int8) --threads 1 --cpus 3 --k 1 --tag w7alone \
    --save_optimized int8_optimized_pi.onnx
for V in fp32 int8; do
  $PY benchmark_pi.py --stage sep $(args_for $V) --threads 4 --k 1 --tag w7alone
done

if ! ls results/sed_t2_c01_sed2core_*_summary.json > /dev/null 2>&1; then
  echo "== No 2-core SED baseline yet: running it =="
  $PY benchmark_pi.py --stage sed --threads 2 --cpus 0,1 --tag sed2core
fi
SED2_P95=$($PY -c "import json,glob; f=sorted(glob.glob('results/sed_t2_c01_sed2core_*_summary.json'))[-1]; print(json.load(open(f))['p95_ms'])")
echo "Using SED (2 cores) p95 = ${SED2_P95} ms"

s3_run() {   # $1 = variant, $2 = k
  rm -f s3_ready
  $PY benchmark_pi.py --stage sed --threads 2 --cpus 0,1 --pace_s 2.0 --repeats 100000 --warmup 2 \
      --ready_file s3_ready --tag "w7s3bg_$1_k$2" > "w7_s3_sed_bg_$1_k$2.log" 2>&1 &
  SED_PID=$!
  for i in $(seq 1 700); do
    [ -f s3_ready ] && break
    kill -0 "$SED_PID" 2>/dev/null || { echo "Background SED died, see w7_s3_sed_bg_$1_k$2.log"; exit 1; }
    sleep 1
  done
  [ -f s3_ready ] || { echo "Background SED never became ready"; exit 1; }
  $PY benchmark_pi.py --stage sep $(args_for $1) --threads 1 --cpus 3 --k "$2" --sed_p95_ms "$SED2_P95" \
      --class_ids 343 420 --cool_to_c 100 --tag "w7s3"
  kill -0 "$SED_PID" 2>/dev/null || { echo "Background SED stopped early; run invalid"; exit 1; }
  kill -TERM "$SED_PID"; wait "$SED_PID" 2>/dev/null || true; SED_PID=""
}

echo "== 2. Two-core S3 layout, k = 1 =="
for V in torch fp32 int8; do s3_run $V 1; done

if [ -z "${FAST:-}" ]; then
AUTO=$($PY - <<'EOF'
import glob, json
newest = {}                                    # newest VALID run per variant (names end in a timestamp)
for f in sorted(glob.glob("results/sep*_k1_t1_c3_w7s3_*_summary.json")):
    j = json.load(open(f))
    if j.get("valid_run") is False:
        continue
    v = "torch" if j.get("backend") == "torch" else j.get("precision")
    newest[v] = j["median_ms"]
if not newest:
    raise SystemExit("no valid k = 1 S3 run")
print(min(newest, key=newest.get))
EOF
)
  FAST=$AUTO
else
  AUTO="(not computed: FAST set by hand)"
fi
case "$FAST" in torch|fp32|int8) ;; *) echo "FAST must be torch, fp32 or int8, not '$FAST'"; exit 1 ;; esac
echo "== 3. Two-core S3 layout, k = 2 (engine + jackhammer): $FAST (automatic choice: $AUTO) =="
s3_run "$FAST" 2

echo "== 4. INT8 operators actually run by ONNX Runtime on this CPU =="
if [ ! -f int8_optimized_pi.onnx ]; then
  echo "int8_optimized_pi.onnx was not written by the INT8 run in step 1; check that run's output"
elif ! $PY -c "import onnx" 2>/dev/null; then
  echo "(skipped: pip install onnx to see this)"
else
  $PY - <<'EOF'
from collections import Counter
import onnx
g = onnx.load("int8_optimized_pi.onnx").graph
made = {o: n.op_type for n in g.node for o in n.output}
by_design = ("stft", "pre_conv", "after_conv")          # kept FP32 on purpose by int8_check.py defaults
fused, fallback, design, other = Counter(), Counter(), Counter(), Counter()
for n in g.node:
    if "QLinear" in n.op_type or "Integer" in n.op_type:
        fused[n.op_type] += 1
    elif "Conv" in n.op_type:                          # Conv, ConvTranspose, FusedConv, NhwcFusedConv, ...
        if len(n.input) > 1 and made.get(n.input[1]) == "DequantizeLinear":   # the WEIGHTS are INT8
            fallback[n.op_type] += 1
        elif any(k in n.name.lower() for k in by_design):
            design[n.op_type] += 1
        else:
            other[n.op_type] += 1
print("INT8 kernels (QLinear* / *Integer):", dict(fused) or "none")
print("Conv-type ops with INT8 weights but FP32 compute (fusion failed):", dict(fallback) or "none")
print("FP32 by design (STFT, inverse STFT, pre_conv, after_conv, matched by node name):", dict(design) or "none")
print("Other FP32 Conv-type ops (decoder ConvTranspose expected; names may change when ORT fuses):",
      dict(other) or "none")
EOF
fi

echo "== Summary (median / p95 per USS job; SED overruns from its background log) =="
$PY - <<'EOF'
import glob, json
rows = []
for f in sorted(glob.glob("results/*w7*_summary.json")):
    j = json.load(open(f))
    rows.append((f.split("/")[-1].rsplit("_", 1)[0], j.get("median_ms"), j.get("p95_ms"), j.get("peak_rss_mb"),
                 j.get("hop_overruns", ""), j.get("valid_run"), j.get("temp_max_c")))
print(f"{'run':52}{'median ms':>11}{'p95 ms':>9}{'RSS MB':>8}{'SED overruns':>14}{'valid':>7}{'max °C':>8}")
for r in rows:
    print(f"{r[0]:52}{r[1]:>11}{r[2]:>9}{r[3]:>8}{str(r[4]):>14}{str(r[5]):>7}{r[6]:>8}")
EOF