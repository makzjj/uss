"""Gate 1 CPU benchmark for the Raspberry Pi 4: times ONE stage per process.
 
  --stage sed : Cnn14 audio tagging on one 2 s segment (the always-on stage).
                Also saves condition.npy (Cnn14 output masked to the top class),
                which is exactly the at_soft condition the stock pipeline builds.
  --stage sep : ResUNet30 separation of the same segment with that condition
                (the stage selective activation is meant to run less often).
 
Run each stage / thread count in a FRESH process, e.g.
  python benchmark_pi.py --stage sed --threads 4
  python benchmark_pi.py --stage sep --threads 4
"""
import argparse
import csv
import json
import platform
import resource
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
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--repeats", type=int, default=50)
p.add_argument("--outdir", default="results")
args = p.parse_args()
 
import torch  # noqa: E402
 
# Thread settings must be fixed before the first parallel op runs.
torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
 
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
 
SR = 32000                  # from the at_soft config (data.sample_rate)
SEG_S = 2.0                 # from the at_soft config (data.segment_seconds)
SEG = int(SR * SEG_S)       # 64 000 samples
 
 
# ---------- environment / health helpers ----------
def cpu_temp_c():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except OSError:
        return float("nan")
 
 
def get_throttled():
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return "n/a"
 
 
def read_sys(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "n/a"
 
 
def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")
 
 
def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KB
 
 
env = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "model": read_sys("/proc/device-tree/model").rstrip("\x00"),
    "os": read_sys("/etc/os-release").splitlines()[0] if Path("/etc/os-release").exists() else "n/a",
    "kernel": platform.release(),
    "arch": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "governor": read_sys("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
    "cur_freq_khz": read_sys("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
    "threads": torch.get_num_threads(),
    "interop_threads": torch.get_num_interop_threads(),
    "stage": args.stage,
    "batch_size": 1,
    "precision": "fp32",
    "segment_seconds": SEG_S,
    "sample_rate": SR,
    "warmup": args.warmup,
    "repeats": args.repeats,
    "temp_before_c": cpu_temp_c(),
    "throttled_before": get_throttled(),
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
 
# ---------- build the stage ----------
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
    cond = torch.from_numpy(np.load(args.condition)).float().reshape(1, 527)
    input_dict = {"mixture": x.unsqueeze(1), "condition": cond}   # (1,1,64000), (1,527)
 
    def forward():
        return model(input_dict=input_dict)["waveform"]   # includes STFT and iSTFT
 
env["params_millions"] = round(sum(t.numel() for t in model.parameters()) / 1e6, 2)
env["rss_after_load_mb"] = round(rss_mb(), 1)
 
# ---------- warm-up, then timed runs ----------
times, temps = [], []
with torch.inference_mode():
    for _ in range(args.warmup):
        out = forward()
    for i in range(args.repeats):
        t0 = time.perf_counter()
        out = forward()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        temps.append(cpu_temp_c())
        print(f"run {i + 1:3d}/{args.repeats}: {(t1 - t0) * 1000:8.1f} ms   {temps[-1]:.1f} °C", flush=True)
 
t = np.array(times)
summary = {
    "median_ms": round(float(np.median(t)) * 1000, 1),
    "p95_ms": round(float(np.percentile(t, 95)) * 1000, 1),
    "mean_ms": round(float(t.mean()) * 1000, 1),
    "min_ms": round(float(t.min()) * 1000, 1),
    "max_ms": round(float(t.max()) * 1000, 1),
    "rtf_median": round(float(np.median(t)) / SEG_S, 3),
    "peak_rss_mb": round(peak_rss_mb(), 1),
    "temp_max_c": max(temps),
    "temp_after_c": cpu_temp_c(),
    "throttled_after": get_throttled(),
}
 
# SED run also produces the condition the sep run uses (top-1 class, masked like separate_by_class_ids).
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
tag = f"{args.stage}_t{args.threads}_{datetime.now():%Y%m%d-%H%M%S}"
with open(outdir / f"{tag}_runs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "latency_ms", "cpu_temp_c"])
    for i, (lat, tc) in enumerate(zip(times, temps), 1):
        w.writerow([i, round(lat * 1000, 2), tc])
(outdir / f"{tag}_summary.json").write_text(json.dumps({**env, **summary}, indent=2))
 
print("\n" + json.dumps(summary, indent=2))
print(f"Saved results/{tag}_runs.csv and _summary.json")
 