#!/usr/bin/env bash
# S3 "take turns" variant: SED (core 0, one window every 2 s) PAUSES USS while it
# works and resumes it afterwards, so the two never use the memory system at the
# same time. USS uses all 4 cores whenever SED is idle.
# Run from the benchmarks folder on the Pi, inside the venv:
#   bash run_s3_pause.sh        (k = 1)
#   bash run_s3_pause.sh 3      (k = 3)
set -e
PY=python
K=${1:-1}
USS_PID=""; SED_PID=""; TAIL_PID=""
cleanup() {
  [ -n "$USS_PID" ] && kill -CONT "$USS_PID" 2>/dev/null || true   # never leave USS frozen
  for P in $SED_PID $USS_PID $TAIL_PID; do kill "$P" 2>/dev/null || true; done
}
trap cleanup EXIT

if pgrep -f benchmark_pi.py > /dev/null; then
  echo "Another benchmark_pi.py is still running:"; pgrep -af benchmark_pi.py
  echo "Stop it first (kill <PID>), then rerun."; exit 1
fi

SED1_P95=$($PY -c "import json,glob; f=sorted(glob.glob('results/sed_t1_2*_summary.json'))[-1]; print(json.load(open(f))['p95_ms'])")
echo "Using SED p95 = ${SED1_P95} ms"

echo "== Starting USS (4 threads, k = $K) =="
$PY benchmark_pi.py --stage sep --threads 4 --k "$K" --sed_p95_ms "$SED1_P95" \
    --tag s3pause > s3_uss_pause_k$K.log 2>&1 &
USS_PID=$!
sleep 2
kill -0 "$USS_PID" 2>/dev/null || { echo "USS failed to start, see s3_uss_pause_k$K.log"; exit 1; }

echo "== Starting SED on core 0, pausing USS while each window runs =="
$PY benchmark_pi.py --stage sed --threads 1 --cpus 0 --pace_s 2.0 --repeats 100000 --warmup 2 \
    --pause_pid "$USS_PID" --cool_to_c 100 --tag s3bg_pause_k$K > s3_sed_pause_k$K.log 2>&1 &
SED_PID=$!

tail -f s3_uss_pause_k$K.log &
TAIL_PID=$!

wait "$USS_PID" || { echo "USS run failed, see s3_uss_pause_k$K.log"; exit 1; }
USS_PID=""
kill -0 "$SED_PID" 2>/dev/null || { echo "SED stopped before USS finished; run invalid (see s3_sed_pause_k$K.log)"; exit 1; }
wait "$SED_PID" 2>/dev/null || true      # SED notices USS has finished and saves its summary
SED_PID=""
kill "$TAIL_PID" 2>/dev/null || true; TAIL_PID=""

echo "== Results (SED row: median_ms under 2000 and hop_overruns near 0 = protected) =="
$PY summarize_pi.py | grep -E "file|pause"