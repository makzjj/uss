#!/usr/bin/env bash
# Week 7: can an FP32 ONNX SED close the P1 gap (stage-1 alert <= 3 s = 2 s window + SED p95 <= ~1.0 s)?
# INT8 SED failed its frozen-threshold check (results/o1_frozen_sedint8.json), so this tests FP32 ONNX,
# which matches PyTorch to ~1e-6 and needs no accuracy check beyond the export check.
#   0. Export the ONNX SED on this Pi (export_sed_onnx_pi.py), if not already there
#   1. SED alone on cores 0-1: PyTorch vs ONNX (unpaced, as Week 5's 0.97 s)
#   2. SED on cores 0-1, one window every 2 s, while INT8 USS runs two-class jobs nonstop on core 3
#      (the deployed layout at its busiest): PyTorch vs ONNX. P1 estimate = 2 s + SED p95.
# About 40 minutes. Run from the benchmarks folder on the Pi, inside the venv:
#   nohup bash run_week7_sed_pi.sh > week7_sed_pi.log 2>&1 &
set -e
PY="python -u"
BG_PID=""
cleanup() { [ -n "$BG_PID" ] && kill "$BG_PID" 2>/dev/null || true; rm -f sedtest_ready; }
trap cleanup EXIT
SED_ONNX=cnn14_fp32_pi.onnx
USS_INT8=resunet30_at_soft_int8_v2.onnx

echo "== Checks =="
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c
for f in $USS_INT8 query_conditions/engine.npy query_conditions/jackhammer.npy bench_clip_32k_mono.wav \
         export_sed_onnx_pi.py; do
  [ -f "$f" ] || { echo "Missing $f"; exit 1; }
done
$PY -c "import onnxruntime as o; print('onnxruntime', o.__version__)"
if pgrep -f benchmark_pi.py > /dev/null; then
  echo "Another benchmark_pi.py is still running:"; pgrep -af benchmark_pi.py; exit 1
fi

echo "== 0. ONNX SED =="
SED_JSON="${SED_ONNX%.onnx}.json"
if [ ! -f "$SED_ONNX" ]; then
  $PY export_sed_onnx_pi.py --out "$SED_ONNX"
fi
$PY - "$SED_JSON" <<'EOF'
import json, sys
try:
    ok = json.load(open(sys.argv[1]))["passed"] is True
except (OSError, KeyError, ValueError) as e:
    sys.exit(f"Cannot read the export check {sys.argv[1]} ({e}); delete the .onnx and re-run")
if not ok:
    sys.exit(f"Export check in {sys.argv[1]} did not pass; delete the .onnx and re-run")
print(f"Export check passed ({sys.argv[1]})")
EOF

echo "== 1. SED alone, 2 cores =="
$PY benchmark_pi.py --stage sed --backend torch --threads 2 --cpus 0,1 --tag w7sedalone
$PY benchmark_pi.py --stage sed --backend onnx --onnx "$SED_ONNX" --threads 2 --cpus 0,1 --tag w7sedalone

busy_run() {   # $1 = backend: SED timed on cores 0-1 (paced 2 s) while INT8 USS (k = 2) runs on core 3
  rm -f sedtest_ready
  $PY benchmark_pi.py --stage sep --backend onnx --onnx "$USS_INT8" --conditions_dir query_conditions \
      --threads 1 --cpus 3 --k 2 --class_ids 343 420 --repeats 100000 --warmup 1 --ready_file sedtest_ready \
      --tag "w7sedbg_$1" > "w7sed_uss_bg_$1.log" 2>&1 &          # default cool-down first, as the old script
  BG_PID=$!
  for i in $(seq 1 700); do
    [ -f sedtest_ready ] && break
    kill -0 "$BG_PID" 2>/dev/null || { echo "Background USS died, see w7sed_uss_bg_$1.log"; exit 1; }
    sleep 1
  done
  [ -f sedtest_ready ] || { echo "Background USS never became ready"; exit 1; }
  if [ "$1" = "onnx" ]; then ARGS="--backend onnx --onnx $SED_ONNX"; else ARGS="--backend torch"; fi
  $PY benchmark_pi.py --stage sed $ARGS --threads 2 --cpus 0,1 --pace_s 2.0 --cool_to_c 100 --tag w7sedbusy
  kill -0 "$BG_PID" 2>/dev/null || { echo "Background USS stopped early; run invalid"; exit 1; }
  kill -TERM "$BG_PID"; wait "$BG_PID" 2>/dev/null || true; BG_PID=""
}

echo "== 2. SED with INT8 USS (two classes per job) running on core 3 =="
busy_run torch
busy_run onnx

echo "== Summary =="
$PY - <<'EOF'
import glob, json
newest = {}                                    # newest run per backend and mode (names end in a timestamp)
for f in sorted(glob.glob("results/sed*_w7sed*_summary.json")):
    j = json.load(open(f))
    newest[(j.get("backend"), "busy" if "w7sedbusy" in f else "alone")] = f
print(f"{'run':46}{'median s':>10}{'p95 s':>8}{'P1 = 2 s + p95':>16}{'meets 3 s':>11}{'overruns':>10}{'valid':>7}")
for f in sorted(newest.values()):
    j = json.load(open(f))
    p95 = j["p95_ms"] / 1000
    busy = "w7sedbusy" in f
    p1 = f"{2 + p95:.2f} s" if busy else "-"
    meets = ("yes" if 2 + p95 <= 3.0 else "no") if busy else "-"
    print(f"{f.split('/')[-1].rsplit('_', 1)[0]:46}{j['median_ms'] / 1000:>10.3f}{p95:>8.3f}{p1:>16}{meets:>11}"
          f"{str(j.get('hop_overruns', '')):>10}{str(j.get('valid_run')):>7}")
print("P1 counts the 2 s window plus SED time only; UI time is extra (Requirements P1).")
EOF