#!/usr/bin/env bash
# S3 variant: SED on cores 0-1 (2 threads, one window every 2 s) while USS runs
# on core 3 (1 thread). Does SED stay near 1 s with USS beside it?
# Run from the benchmarks folder on the Pi, inside the venv:  bash run_s3_2core.sh
set -e
PY=python
SED_PID=""
cleanup() { [ -n "$SED_PID" ] && kill "$SED_PID" 2>/dev/null || true; }
trap cleanup EXIT

if pgrep -f benchmark_pi.py > /dev/null; then
  echo "Another benchmark_pi.py is still running:"; pgrep -af benchmark_pi.py; exit 1
fi

SED2_P95=$($PY -c "import json,glob; f=sorted(glob.glob('results/sed_t2_c01_sed2core_*_summary.json'))[-1]; print(json.load(open(f))['p95_ms'])")
echo "Using SED (2 cores) p95 = ${SED2_P95} ms"

rm -f s3_ready
$PY benchmark_pi.py --stage sed --threads 2 --cpus 0,1 --pace_s 2.0 --repeats 100000 --warmup 2 \
    --ready_file s3_ready --tag s3bg_2core > s3_sed_bg_2core.log 2>&1 &
SED_PID=$!
for i in $(seq 1 120); do
  [ -f s3_ready ] && break
  kill -0 "$SED_PID" 2>/dev/null || { echo "Background SED died, see s3_sed_bg_2core.log"; exit 1; }
  sleep 1
done
[ -f s3_ready ] || { echo "Background SED never became ready"; exit 1; }

$PY benchmark_pi.py --stage sep --threads 1 --cpus 3 --k 1 --sed_p95_ms "$SED2_P95" \
    --cool_to_c 100 --tag s3_2core
kill -0 "$SED_PID" 2>/dev/null || { echo "Background SED stopped early; run invalid"; exit 1; }
kill -TERM "$SED_PID"; wait "$SED_PID" 2>/dev/null || true; SED_PID=""
rm -f s3_ready

echo "== Results =="
$PY summarize_pi.py | grep -E "file|2core"