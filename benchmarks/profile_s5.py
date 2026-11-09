"""Week 13: S5 baseline profile of the deployed models on the Pi (Requirements S5: reported, no pass mark).

For each model, in its own process and pinned as deployed (SED: 2 threads on cores 0-1; USS: 1 thread on core 3):
  - time per 2 s window (median, p95) over --runs timed runs after warm-up, with the profiler OFF
  - front-end share, from a SEPARATE profiled session (profiling slows operations, so its absolute times are not
    used): operations whose names contain stft / istft / spectrogram / logmel, timed runs only (warm-up excluded);
    the front-end time is estimated as share x the unprofiled median
  - CPU use: process CPU time / wall time during the timed runs (1.0 = one core fully busy)
  - peak resident memory of the process (VmHWM), and memory before loading
  - model file size and parameter count
If no operation matches the front-end names, the script says so and lists the most expensive operation names, so the
pattern can be adjusted instead of reporting a wrong split.

Usage (benchmarks folder on the Pi, venv active, governor at performance, nothing else running):
  python profile_s5.py
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HOP = 64000
FRONT = re.compile(r"stft|istft|spectrogram|logmel", re.IGNORECASE)


def status_kb(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return None


def make_session(path, threads, profile_prefix=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if profile_prefix:
        so.enable_profiling = True
        so.profile_file_prefix = profile_prefix
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def run_model(name, path, threads, cores, runs, conditions, no_pin, q):
    try:
        if not no_pin:
            os.sched_setaffinity(0, cores)
        rss0 = status_kb("VmRSS")
        rng = np.random.default_rng(0)
        x = (0.05 * rng.standard_normal(HOP)).astype(np.float32)
        if name == "sed":
            feed = {"audio": x[None, :]}
        else:
            cond = np.load(Path(conditions) / "jackhammer.npy").astype(np.float32).reshape(1, 527)
            feed = {"mixture": x[None, None, :], "condition": cond}

        # pass 1: timing with the profiler OFF (comparable with Week 7 and S4)
        sess = make_session(path, threads)
        for _ in range(2):
            sess.run(None, feed)                                  # warm-up, untimed
        t_wall0, c0 = time.monotonic(), os.times()
        times = []
        for _ in range(runs):
            t = time.monotonic()
            sess.run(None, feed)
            times.append(time.monotonic() - t)
        t_wall, c1 = time.monotonic() - t_wall0, os.times()
        cpu = ((c1.user - c0.user) + (c1.system - c0.system)) / t_wall
        del sess

        # pass 2: a separate profiled session, only for the front-end share (its absolute times are inflated)
        n_prof = min(runs, 10)
        psess = make_session(path, threads, f"/tmp/s5_{name}")
        for _ in range(2):
            psess.run(None, feed)                                 # warm-up, excluded below
        for _ in range(n_prof):
            psess.run(None, feed)
        prof = psess.end_profiling()
        events = json.loads(Path(prof).read_text())
        run_ev = sorted((e for e in events if e.get("cat") == "Session" and e.get("name") == "model_run"),
                        key=lambda e: e["ts"])[-n_prof:]          # the timed runs only, not the warm-up
        windows = [(e["ts"], e["ts"] + e["dur"]) for e in run_ev]
        inside = lambda e: any(a <= e["ts"] <= b for a, b in windows)  # noqa: E731
        node = [e for e in events if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time") and inside(e)]
        front = sum(e["dur"] for e in node if FRONT.search(e["name"])) / n_prof
        total = sum(e["dur"] for e in node) / n_prof
        by_name = {}
        for e in node:
            k = e["name"].replace("_kernel_time", "")
            by_name[k] = by_name.get(k, 0) + e["dur"]
        top = sorted(by_name.items(), key=lambda kv: -kv[1])[:8]
        Path(prof).unlink(missing_ok=True)
        try:
            import onnx
            m = onnx.load(path)
            params = int(sum(np.prod(t.dims) for t in m.graph.initializer))
        except Exception:
            params = None
        share = (front / total) if total else None
        med = float(np.median(times))
        q.put({"model": name, "file": path, "file_mb": round(Path(path).stat().st_size / 1e6, 1), "params": params,
               "threads": threads, "cores": sorted(cores), "runs": runs,
               "median_s": med, "p95_s": float(np.percentile(times, 95)),
               "front_end_share": share, "front_end_s_est": (share * med) if share is not None else None,
               "front_end_matched": any(FRONT.search(k) for k in by_name), "profiled_runs": n_prof,
               "top_operations": [(k, round(v / 1e6 / n_prof, 4)) for k, v in top],
               "cpu_cores_busy": round(cpu, 2), "rss_before_mb": round(rss0 / 1024, 1) if rss0 else None,
               "peak_rss_mb": round(status_kb("VmHWM") / 1024, 1)})
    except Exception as e:                                         # report, do not hang the parent
        q.put({"model": name, "error": str(e).splitlines()[0]})


def read_sys(path, default=""):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sed_onnx", default="cnn14_fp32_pi.onnx")
    ap.add_argument("--uss_onnx", default="resunet30_at_soft_int8_v2.onnx")
    ap.add_argument("--conditions", default="query_conditions")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--no_pin", action="store_true", help="testing only")
    ap.add_argument("--allow_governor", action="store_true", help="testing only")
    a = ap.parse_args()
    gov = sorted({read_sys(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor") for i in range(4)})
    if gov != ["performance"] and not a.allow_governor:
        sys.exit(f"CPU governor {gov}: set performance first (echo performance | sudo tee .../scaling_governor)")
    thr0 = os.popen("vcgencmd get_throttled 2>/dev/null").read().strip()
    ctx = mp.get_context("fork")
    res = []
    for name, path, threads, cores in (("sed", a.sed_onnx, 2, {0, 1}), ("uss", a.uss_onnx, 1, {3})):
        if not Path(path).exists():
            sys.exit(f"Missing {path}")
        q = ctx.Queue()
        p = ctx.Process(target=run_model, args=(name, path, threads, cores, a.runs, a.conditions, a.no_pin, q))
        p.start()
        got = None
        deadline = time.monotonic() + 1800
        while got is None and time.monotonic() < deadline:
            try:
                got = q.get(timeout=5)
            except Exception:
                if not p.is_alive():                               # killed (e.g. out of memory) or crashed
                    sys.exit(f"{name}: the worker died (exit code {p.exitcode}) without a result; check dmesg / memory")
        p.join(timeout=30)
        if got is None:
            p.kill()
            sys.exit(f"{name}: no result within 30 minutes; stopped")
        res.append(got)
        if "error" in got:
            sys.exit(f"{name}: {got['error']}")
    print(f"S5 profile ({a.runs} timed runs per model, pinned as deployed; one 2 s window per run)")
    print(f"   {'model':5} {'file MB':>8} {'params M':>9} {'median s':>9} {'p95 s':>7} {'front-end share':>16} "
          f"{'≈ front-end s':>14} {'CPU cores':>10} {'peak RSS MB':>12}")
    for r in res:
        share = f"{100 * r['front_end_share']:.1f}%" if r["front_end_share"] is not None else "-"
        fe = f"{r['front_end_s_est']:.3f}" if r["front_end_s_est"] is not None else "-"
        params = f"{r['params'] / 1e6:.1f}" if r["params"] else "-"
        print(f"   {r['model']:5} {r['file_mb']:>8} {params:>9} {r['median_s']:>9.3f} {r['p95_s']:>7.3f} "
              f"{share:>16} {fe:>14} {r['cpu_cores_busy']:>10} {r['peak_rss_mb']:>12}")
        if not r["front_end_matched"]:
            print(f"      no front-end operation matched; most expensive operations: {r['top_operations'][:4]}")
    thr1 = os.popen("vcgencmd get_throttled 2>/dev/null").read().strip()
    valid = all(t.endswith("0x0") for t in (thr0, thr1) if t)
    print("   timings: profiler OFF (comparable with Week 7 and S4). Front-end share: from a separate profiled session, "
          "timed runs only (warm-up excluded); front-end = STFT and log-mel (SED) / STFT and inverse STFT (USS); "
          "≈ front-end s = share × median")
    print(f"   throttling before/after: {thr0 or 'n/a'} / {thr1 or 'n/a'}; valid run: {valid}")
    for r in res:
        r["throttled"], r["valid_run"], r["governor"] = [thr0, thr1], valid, gov
    total_peak = sum(r["peak_rss_mb"] for r in res)
    print(f"   both model processes together: about {total_peak:.0f} MB peak resident memory (H3 headroom 2 GB)")
    out = Path("results") / f"s5_profile_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()