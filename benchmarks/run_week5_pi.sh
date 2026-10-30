#!/usr/bin/env bash
# Week 5 Pi benchmarks (Requirements v0.3.5): S1, S2, S3.
# Run from the benchmarks folder on the Pi, inside the venv:  bash run_week5_pi.sh
set -e
PY=python
SED_PID=""
cleanup() { [ -n "$SED_PID" ] && kill "$SED_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "== Governor check =="
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c

echo "== S1: single model, 4 threads and 1 thread =="
$PY benchmark_pi.py --stage sed --threads 4
$PY benchmark_pi.py --stage sed --threads 1
SED1_P95=$($PY -c "import json,glob; f=sorted(glob.glob('results/sed_t1_*_summary.json'))[-1]; print(json.load(open(f))['p95_ms'])")
echo "SED p95 at 1 thread: ${SED1_P95} ms (used in the USS deadline)"
$PY benchmark_pi.py --stage sep --threads 4 --k 1 --sed_p95_ms "$SED1_P95"
$PY benchmark_pi.py --stage sep --threads 1 --k 1 --sed_p95_ms "$SED1_P95"

echo "== S2: k = 2, 3 serial queries per job, 4 threads =="
for K in 2 3; do
  $PY benchmark_pi.py --stage sep --threads 4 --k $K --sed_p95_ms "$SED1_P95"
done

echo "== S3: SED on core 0 (one window every 2 s) while USS runs on cores 1-3 =="
for K in 1 3; do
  rm -f s3_ready
  $PY benchmark_pi.py --stage sed --threads 1 --cpus 0 --pace_s 2.0 --repeats 100000 --warmup 2 \
      --ready_file s3_ready --tag s3bg_k$K > s3_sed_bg_k$K.log 2>&1 &
  SED_PID=$!
  for i in $(seq 1 120); do            # wait up to 120 s for SED warm-up
    [ -f s3_ready ] && break
    kill -0 "$SED_PID" 2>/dev/null || { echo "Background SED died, see s3_sed_bg_k$K.log"; exit 1; }
    sleep 1
  done
  [ -f s3_ready ] || { echo "Background SED never became ready"; exit 1; }
  $PY benchmark_pi.py --stage sep --threads 3 --cpus 1,2,3 --k $K --sed_p95_ms "$SED1_P95" \
      --cool_to_c 100 --tag s3
  kill -0 "$SED_PID" 2>/dev/null || { echo "Background SED stopped during USS run; S3 invalid"; exit 1; }
  kill -TERM "$SED_PID"; wait "$SED_PID" 2>/dev/null || true; SED_PID=""
done
rm -f s3_ready

echo "== Summary table =="
$PY summarize_pi.py