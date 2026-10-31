#!/usr/bin/env bash
# S3 variants: can SED be protected while USS runs?
#   nice     : USS on cores 1-3 at lowest priority (nice 19)
#   onecore  : USS on core 3 only, 1 thread
# SED runs in the background on core 0, one window every 2 s, and its summary
# (including hop_overruns) is saved when the USS run finishes.
# Run from the benchmarks folder on the Pi, inside the venv:  bash run_s3_variants.sh
set -e
PY=python
SED_PID=""
cleanup() { [ -n "$SED_PID" ] && kill "$SED_PID" 2>/dev/null || true; }
trap cleanup EXIT

SED1_P95=$($PY -c "import json,glob; f=sorted(glob.glob('results/sed_t1_2*_summary.json'))[-1]; print(json.load(open(f))['p95_ms'])")
echo "Using SED p95 = ${SED1_P95} ms"

echo "== Checking nothing else is running =="
if pgrep -f benchmark_pi.py > /dev/null; then
  echo "Another benchmark_pi.py is still running:"; pgrep -af benchmark_pi.py
  echo "Stop it first (kill <PID>), then rerun."; exit 1
fi

run_variant () {
  NAME=$1; shift
  rm -f s3_ready
  $PY benchmark_pi.py --stage sed --threads 1 --cpus 0 --pace_s 2.0 --repeats 100000 --warmup 2 \
      --ready_file s3_ready --tag s3bg_$NAME > s3_sed_bg_$NAME.log 2>&1 &
  SED_PID=$!
  for i in $(seq 1 120); do
    [ -f s3_ready ] && break
    kill -0 "$SED_PID" 2>/dev/null || { echo "Background SED died, see s3_sed_bg_$NAME.log"; exit 1; }
    sleep 1
  done
  [ -f s3_ready ] || { echo "Background SED never became ready"; exit 1; }
  echo "== Variant: $NAME =="
  "$@" --sed_p95_ms "$SED1_P95" --cool_to_c 100 --tag s3$NAME
  kill -0 "$SED_PID" 2>/dev/null || { echo "Background SED stopped early; variant invalid"; exit 1; }
  kill -TERM "$SED_PID"; wait "$SED_PID" 2>/dev/null || true; SED_PID=""
  rm -f s3_ready
}

run_variant nice    nice -n 19 $PY benchmark_pi.py --stage sep --threads 3 --cpus 1,2,3 --k 1
run_variant onecore $PY benchmark_pi.py --stage sep --threads 1 --cpus 3 --k 1

echo "== Results (SED rows: look at median_ms and hop_overruns) =="
$PY summarize_pi.py | grep -E "file|s3bg|s3nice|s3onecore"