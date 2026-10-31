"""Raspberry Pi 4 CPU benchmark for Gate 2 (Requirements v0.3.5, S1-S3).
Times ONE stage per process.

  --stage sed : Cnn14 audio tagging on one 2 s segment (the always-on stage).
                Also saves condition.npy (Cnn14 output masked to the top class),
                the same at_soft condition format the stock pipeline builds.
  --stage sep : ResUNet30 separation of the same segment. With --k > 1, one
                "job" = k serial queries (S2), each with its own condition
                vector (one per target class). Condition values only steer the
                FiLM layers; they do not change the amount of computation.

Locked settings it checks/records: cool-down before start, SIGTERM-safe summary, performance governor, thread count,
CPU cores used (--cpus), >= 50 timed runs after warm-up, clock per run,
temperature, throttle flags, peak RAM, swap, model load time.

Examples (fresh process each time):
  python benchmark_pi.py --stage sed --threads 4                    # S1 (SED)
  python benchmark_pi.py --stage sep --threads 4                    # S1 (USS, k=1)
  python benchmark_pi.py --stage sep --threads 4 --k 3              # S2
  # S3 (contention): see run_week5_pi.sh
"""
import argparse
import csv
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--stage", choices=["sed", "sep"], required=True)
p.add_argument("--audio", default="bench_clip_32k_mono.wav", help="32 kHz mono WAV")
p.add_argument("--offset", type=float, default=0.0, help="start of the 2 s segment, seconds")
p.add_argument("--sed_weights", default=str(Path.home() / ".cache/panns/Cnn14_mAP=0.431.pth"))
p.add_argument("--sep_weights", default="resunet30_at_soft_fp32.pth")
p.add_argument("--condition", default="condition.npy")
p.add_argument("--repo", default=str(Path.home() / "uss"), help="USS repo root (for uss.models.resunet)")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--cpus", default="", help="cores to pin to, e.g. '0' or '1,2,3' (empty = no pinning)")
p.add_argument("--k", type=int, default=1, help="queries per USS job (sep stage only)")
p.add_argument("--class_ids", type=int, nargs="+", default=[343, 319, 420],
               help="target classes used for the k condition vectors")
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--repeats", type=int, default=50)
p.add_argument("--pace_s", type=float, default=0.0,
               help="start one run every pace_s seconds (S3 background SED: 2.0 = one window per hop)")
p.add_argument("--tag", default="", help="extra label in the output file names, e.g. s3")
p.add_argument("--sed_p95_ms", type=float, default=0.0,
               help="SED p95 from S1, added to the sep deadline (window fill + SED + k queries)")
p.add_argument("--cool_to_c", type=float, default=60.0, help="wait until the SoC is below this before starting")
p.add_argument("--ready_file", default="", help="touch this file once warm-up is done (used by S3)")
p.add_argument("--pause_pid", type=int, default=0,
               help="sed stage: pause this process (USS) while each SED window runs, resume after")
p.add_argument("--allow_governor", action="store_true", help="run even if governor is not 'performance'")
p.add_argument("--outdir", default="results")
args = p.parse_args()

# Pin to cores BEFORE torch starts its thread pool.
if args.cpus:
    os.sched_setaffinity(0, {int(c) for c in args.cpus.split(",")})


def read_sys(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "n/a"


governor = read_sys("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
if governor != "performance" and not args.allow_governor:
    sys.exit(f"CPU governor is '{governor}', locked config needs 'performance'.\n"
             "Set it with: sudo cpufreq-set -r -g performance   (or pass --allow_governor)")

import torch  # noqa: E402

torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SR = 32000            # at_soft config: data.sample_rate
SEG_S = 2.0           # at_soft config: data.segment_seconds
SEG = int(SR * SEG_S)


def cpu_temp_c():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except OSError:
        return float("nan")


def cur_freq_mhz():
    try:
        return int(read_sys("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")) / 1000
    except ValueError:
        return float("nan")


def get_throttled():
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return "n/a"


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def swap_used_mb():
    info = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        info[k] = int(v.split()[0])
    return (info.get("SwapTotal", 0) - info.get("SwapFree", 0)) / 1024


def mem_available_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return float("nan")


STOP = False


def _stop(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, _stop)

# Cool down before starting, so back-to-back runs do not start on a hot SoC
cool_wait = 0.0
t_cool = time.perf_counter()
while cpu_temp_c() > args.cool_to_c and time.perf_counter() - t_cool < 600:
    time.sleep(5)
cool_wait = round(time.perf_counter() - t_cool, 1)

env = {
    "cool_to_c": args.cool_to_c,
    "cool_wait_s": cool_wait,
    "mem_available_before_mb": round(mem_available_mb(), 1),
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "model": read_sys("/proc/device-tree/model").rstrip("\x00"),
    "os": read_sys("/etc/os-release").splitlines()[0] if Path("/etc/os-release").exists() else "n/a",
    "kernel": platform.release(),
    "arch": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "governor": governor,
    "threads": torch.get_num_threads(),
    "interop_threads": torch.get_num_interop_threads(),
    "cpus": sorted(os.sched_getaffinity(0)),
    "stage": args.stage,
    "k": args.k if args.stage == "sep" else None,
    "batch_size": 1,
    "precision": "fp32",
    "segment_seconds": SEG_S,
    "sample_rate": SR,
    "warmup": args.warmup,
    "repeats": args.repeats,
    "pace_s": args.pace_s,
    "temp_before_c": cpu_temp_c(),
    "throttled_before": get_throttled(),
    "swap_before_mb": round(swap_used_mb(), 1),
}

# ---------- fixed input, loaded before any timing ----------
audio, sr = sf.read(args.audio, dtype="float32")
if audio.ndim != 1:
    sys.exit("Expected a mono file.")
if sr != SR:
    sys.exit(f"Expected {SR} Hz, got {sr} Hz. Resample on the server first.")
start = int(args.offset * SR)
seg = audio[start:start + SEG]
if len(seg) < SEG:
    sys.exit("Clip too short for a full 2 s segment at this offset.")
x = torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0)  # (1, 64000)

# ---------- build the stage (load time measured) ----------
t_load0 = time.perf_counter()
if args.stage == "sed":
    from panns_inference.models import Cnn14
    model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                  mel_bins=64, fmin=50, fmax=14000, classes_num=527)  # same args as uss.utils.load_pretrained_panns
    ckpt = torch.load(args.sed_weights, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def forward():
        return model(x)["clipwise_output"]      # includes log-mel front end
else:
    sys.path.insert(0, args.repo)
    from uss.models.resunet import ResUNet30
    model = ResUNet30(input_channels=1, output_channels=1, condition_size=527)
    model.load_state_dict(torch.load(args.sep_weights, map_location="cpu", weights_only=True))
    model.eval()
    base = np.load(args.condition).astype(np.float32).reshape(527)
    conds = []
    for j in range(args.k):
        c = np.zeros(527, dtype=np.float32)
        cid = args.class_ids[j % len(args.class_ids)]
        c[cid] = max(float(base.max()), 0.5)   # one target class per query
        conds.append(torch.from_numpy(c).reshape(1, 527))
    mix = x.unsqueeze(1)                        # (1, 1, 64000)

    def forward():
        out = None
        for c in conds:                         # k serial queries = one job
            out = model(input_dict={"mixture": mix, "condition": c})["waveform"]  # includes STFT and iSTFT
        return out
env["model_load_s"] = round(time.perf_counter() - t_load0, 2)

# Take-turns mode: SED pauses USS while it works, so the two never share the memory system
if args.stage == "sed" and args.pause_pid:
    _plain_forward = forward

    def forward():
        global STOP
        try:
            os.kill(args.pause_pid, signal.SIGSTOP)
        except ProcessLookupError:
            STOP = True                         # USS has finished; stop after this window
            return _plain_forward()
        try:
            return _plain_forward()
        finally:
            try:
                os.kill(args.pause_pid, signal.SIGCONT)
            except ProcessLookupError:
                STOP = True
    env["pause_pid"] = args.pause_pid
env["params_millions"] = round(sum(t.numel() for t in model.parameters()) / 1e6, 2)
env["rss_after_load_mb"] = round(rss_mb(), 1)

# ---------- warm-up, then timed runs ----------
times, temps, freqs, mems = [], [], [], []
with torch.inference_mode():
    for _ in range(args.warmup):
        out = forward()
    if args.ready_file:
        Path(args.ready_file).touch()
    next_start = time.perf_counter()
    for i in range(args.repeats):
        if STOP:
            break
        if args.pace_s > 0:                     # keep a steady rate, like one window per hop
            wait = next_start - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            next_start += args.pace_s
        t0 = time.perf_counter()
        out = forward()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        temps.append(cpu_temp_c())
        freqs.append(cur_freq_mhz())
        mems.append(mem_available_mb())
        print(f"run {i + 1:3d}/{args.repeats}: {(t1 - t0) * 1000:8.1f} ms   "
              f"{temps[-1]:.1f} °C   {freqs[-1]:.0f} MHz", flush=True)

if not times:
    sys.exit("No timed runs completed.")
t = np.array(times)
throttled = get_throttled()
summary = {
    "runs_completed": len(times),
    "median_ms": round(float(np.median(t)) * 1000, 1),
    "p95_ms": round(float(np.percentile(t, 95)) * 1000, 1),
    "mean_ms": round(float(t.mean()) * 1000, 1),
    "min_ms": round(float(t.min()) * 1000, 1),
    "max_ms": round(float(t.max()) * 1000, 1),
    "rtf_median": round(float(np.median(t)) / SEG_S, 3),
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    "temp_max_c": max(temps),
    "temp_after_c": cpu_temp_c(),
    "freq_min_mhz": min(freqs),
    "freq_max_mhz": max(freqs),
    "throttled_after": throttled,
    "valid_run": throttled in ("throttled=0x0", "n/a"),
    "mem_available_min_mb": round(min(mems), 1),
    "swap_after_mb": round(swap_used_mb(), 1),
}
if args.stage == "sep":
    # S3 deadline: window fill (2 s) + SED (p95 from S1) + k serial queries (p95) must be <= 6 s
    summary["sed_p95_ms_used"] = args.sed_p95_ms
    summary["deadline_p95_s"] = round(SEG_S + args.sed_p95_ms / 1000 + float(np.percentile(t, 95)), 2)
    summary["meets_6s_realtime"] = summary["deadline_p95_s"] <= 6.0
else:
    # SED must finish each 2 s window within one hop
    hop = args.pace_s if args.pace_s > 0 else SEG_S
    summary["hop_s"] = hop
    summary["hop_overruns"] = int(sum(x > hop for x in times))
    summary["meets_2s_hop"] = float(np.percentile(t, 95)) <= hop

if args.stage == "sed":
    probs = out.squeeze(0).numpy()
    top = int(probs.argmax())
    cond_vec = np.zeros_like(probs)
    cond_vec[top] = probs[top]
    np.save(args.condition, cond_vec)
    try:
        from panns_inference.config import labels
        summary["top_class"] = f"{top} ({labels[top]}, p={probs[top]:.2f})"
    except Exception:
        summary["top_class"] = f"{top} (p={probs[top]:.2f})"

# ---------- save ----------
outdir = Path(args.outdir)
outdir.mkdir(exist_ok=True)
ktag = f"_k{args.k}" if args.stage == "sep" else ""
cpus = f"_c{args.cpus.replace(',', '')}" if args.cpus else ""
extra = f"_{args.tag}" if args.tag else ""
tag = f"{args.stage}{ktag}_t{args.threads}{cpus}{extra}_{datetime.now():%Y%m%d-%H%M%S}"
with open(outdir / f"{tag}_runs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "latency_ms", "cpu_temp_c", "freq_mhz", "mem_available_mb"])
    for i, (lat, tc, fq, mb) in enumerate(zip(times, temps, freqs, mems), 1):
        w.writerow([i, round(lat * 1000, 2), tc, fq, round(mb, 1)])
(outdir / f"{tag}_summary.json").write_text(json.dumps({**env, **summary}, indent=2))

print("\n" + json.dumps(summary, indent=2))
print(f"Saved {outdir}/{tag}_runs.csv and _summary.json")