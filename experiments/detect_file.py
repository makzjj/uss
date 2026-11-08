"""Week 12 prototype: the frozen cascade (Requirements v0.3.12) on any prerecorded audio file, on the Pi.

The same pipeline as s4_pi.py (ONNX FP32 SED on cores 0-1, frozen class-aware gate, jackhammer-only INT8 USS
on core 3, separated outputs scored on the SED cores between windows, two thresholds), read from the frozen
design file, so it cannot drift from what was evaluated:
  stage-1 alert     SED score >= t1 for a class, shown at once (recovery mode: never withdrawn)
  USS-recovered     a USS result reaching t2 in time, for a window SED alone did not alert on
Modes:
  --mode realtime   windows released as their audio would arrive from a microphone; bounded queue (one running,
                    max_wait waiting) and job deadline (ttl_s from window start); the deployed behaviour, and the
                    mode P1 is measured in. P1 = window start to the complete alert line written and flushed to
                    stdout; with --quiet nothing is printed and P1 is window start to the alert event received
                    (headless). run.json says which one was measured; the two are not interchangeable.
  --mode fast       no pacing and no queue limits: every gated window is separated (offline batch analysis;
                    not a deployment measurement, so no delays are reported)
A final performance run (realtime, time_scale 1, pinned) applies S4's checks: the CPU governor must be at
performance (or --allow_governor), and temperature, clocks and throttling are monitored as in s4_pi.py;
run.json "valid_run" is false if the governor was not performance, the Pi throttled or a clock dropped below
95% of the board maximum. Do not quote P1 from a run that is not valid.
Outputs (results/detect_<file>_<time>/): events.csv (one row per alerting window: time, class, stage, score,
threshold, delays), windows.jsonl and jobs.jsonl (as s4_pi.py), uss_events.jsonl, monitor.jsonl (realtime),
run.json (settings, hashes, versions, counters, P1, throttling).
--compare_s4 <S4 run folder> checks equivalence on the same audio: stage-1 alerts and gated job selections must
match S4 exactly (both depend only on SED scores, which are deterministic), else the exit code is nonzero; USS
recoveries are reported, allowing for queue-timing differences.
If the SED or USS process dies, the run stops with a nonzero exit code instead of hanging.

Usage (benchmarks folder on the Pi, venv active, governor at performance):
  python detect_file.py --audio some_clip.wav --frozen frozen_design_v0_3_10.json
  python detect_file.py --audio s4_heldout/stream.flac --frozen frozen_design_v0_3_10.json \\
      --compare_s4 results/s4_sed_cores_20261107-040541
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SR, HOP = 32000, 64000
CLASS_IDS = {"beeper": 319, "engine": 343, "jackhammer": 420}       # as run_gate1.SOUNDS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sys(path, default="0"):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def temp_c():
    return int(read_sys("/sys/class/thermal/thermal_zone0/temp", "0")) / 1000


def mem_available_mb():
    return next((int(l.split()[1]) / 1024 for l in read_sys("/proc/meminfo", "").splitlines()
                 if l.startswith("MemAvailable")), None)


def governors():
    return sorted({read_sys(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor", "n/a") for i in range(4)})


def vcgencmd_throttled():
    return os.popen("vcgencmd get_throttled 2>/dev/null").read().strip()


class Monitor(threading.Thread):
    """s4_pi.py's monitor: temperature, clocks, throttling and free memory each second, vcgencmd every 30 s,
    and its validity rule (no throttling flags, lowest clock >= 95% of the board maximum)."""

    def __init__(self, path, t0):
        super().__init__(daemon=True)
        self.log, self.t0, self.rows = Log(path), t0, []
        self.stop_evt = threading.Event()

    def run(self):
        last_vc, throttled = -1e9, ""
        while not self.stop_evt.is_set():
            if time.monotonic() - last_vc >= 30:
                throttled = vcgencmd_throttled()
                last_vc = time.monotonic()
            row = {"t": round(time.monotonic() - self.t0, 1), "temp_c": temp_c(),
                   "freq_mhz": [int(read_sys(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq", "0")) / 1000
                                for i in range(4)],
                   "throttled": throttled, "mem_available_mb": mem_available_mb()}
            self.rows.append(row)
            self.log(row)
            self.stop_evt.wait(1.0)

    def finish(self):
        self.stop_evt.set()
        self.join(5)
        seen = sorted({r["throttled"] for r in self.rows} | {vcgencmd_throttled()})
        fmax = max((int(read_sys(f"/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq", "0")) / 1000
                    for i in range(4)), default=0) or 1800.0
        fmin = min((min(r["freq_mhz"]) for r in self.rows), default=None)
        return {"temp_max_c": max((r["temp_c"] for r in self.rows), default=None), "throttled_seen": seen,
                "freq_min_mhz": fmin, "freq_board_max_mhz": fmax,
                "mem_available_min_mb": min((r["mem_available_mb"] for r in self.rows if r["mem_available_mb"]),
                                            default=None),
                "no_throttling": all(t.endswith("0x0") for t in seen if t) and (fmin or 0) >= 0.95 * fmax}


def pin(cores, enabled):
    if enabled:
        os.sched_setaffinity(0, cores)


def session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def load_audio(path):
    """Mono float32 at 32 kHz; resampled if needed (librosa, else scipy)."""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        try:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        except ImportError:
            try:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(SR, sr)
                x = resample_poly(x, SR // g, sr // g).astype(np.float32)
            except ImportError:
                sys.exit(f"{path} is {sr} Hz; install librosa or scipy, or convert first: "
                         f"ffmpeg -i {path} -ac 1 -ar 32000 out.wav")
    n = int(np.ceil(len(x) / HOP)) * HOP                      # pad the last window with silence
    return np.pad(x, (0, n - len(x))).astype(np.float32), sr


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", buffering=1)

    def __call__(self, rec):
        self.fh.write(json.dumps(rec) + "\n")


def sed_proc(a, fz, out, ready, start_evt, t0_val, jobs_q, outs_q, events_q, n_win, audio):
    pin({0, 1}, not a.no_pin)
    sed = session(a.sed_onnx, 2)
    t1, t2 = fz["t1"], fz["t2"]
    g = fz["args"]
    qcls = fz.get("final_query_classes", [s for s in CLASS_IDS if t2[s] is not None])
    gated = [s for s in CLASS_IDS if s not in g["exclude"]]
    hop_s = 2.0 * a.time_scale
    realtime = a.mode == "realtime"
    sed.run(None, {"audio": np.zeros((1, HOP), np.float32)})
    wlog = Log(out / "windows.jsonl")
    ready.set()
    start_evt.wait()
    t0 = t0_val.value
    uss_done = False

    def score_output(block):
        nonlocal uss_done
        try:
            item = outs_q.get(timeout=0.5 if block else 0.0)
        except queue.Empty:
            return False
        if item is None:
            uss_done = True
            return False
        c, f = sed.run(None, {"audio": item["audio"][None, :]})
        s = item["class"]
        score = float(c[0, CLASS_IDS[s]])
        now = time.monotonic() - t0
        in_time = (not realtime) or now <= item["deadline"]
        events_q.put({"kind": "uss", "window": item["window"], "class": s, "score": score, "t2": t2[s],
                      "recovered": in_time and t2[s] is not None and score >= t2[s], "late": not in_time,
                      "scored_at": now, "frames": [round(float(v), 4) for v in f[0, :, CLASS_IDS[s]]]})
        return True

    for k in range(n_win):
        win = audio[k * HOP:(k + 1) * HOP]
        avail = t0 + (k + 1) * hop_s
        if realtime:
            now = time.monotonic()
            if now < avail:
                time.sleep(avail - now)
        t_start = time.monotonic()
        c, f = sed.run(None, {"audio": win[None, :]})
        t_end = time.monotonic()
        scores = {s: float(c[0, CLASS_IDS[s]]) for s in CLASS_IDS}
        alerts = [s for s in CLASS_IDS if scores[s] >= t1[s]]
        events_q.put({"kind": "sed", "window": k, "alerts": alerts, "scores": scores, "t1": t1,
                      "avail": avail - t0 if realtime else None, "sed_done": t_end - t0})
        over = [s for s in gated if scores[s] >= g["lo"]]
        want = (gated if over else []) if g["policy"] == "kmax" else over
        fired = [s for s in want if s in qcls and scores[s] < t1[s]]          # skip classes already alerting at t1
        if fired:
            jobs_q.put({"job": k, "window": k, "classes": fired, "arrive": t_end - t0,
                        "deadline": k * hop_s + g["ttl_s"] * a.time_scale, "audio": win.copy()})
        extra = 0.0
        t3 = time.monotonic()
        if score_output(block=False):
            extra = time.monotonic() - t3
        wlog({"window": k, "start_delay_ms": round((t_start - avail) * 1000, 2) if realtime else None,
              "sed_ms": round((t_end - t_start) * 1000, 1), "extra_score_ms": round(extra * 1000, 1),
              "scores": {s: round(v, 6) for s, v in scores.items()}, "job_submitted": fired})
    jobs_q.put(None)
    while not uss_done:                    # if the USS process dies, main notices and terminates this process
        score_output(block=True)
    events_q.put(None)


def uss_proc(a, fz, out, ready, start_evt, t0_val, jobs_q, outs_q):
    pin({3}, not a.no_pin)
    uss = session(a.uss_onnx, 1)
    conds = {s: np.load(Path(a.conditions) / f"{s}.npy").astype(np.float32).reshape(1, 527) for s in CLASS_IDS}
    g = fz["args"]
    ts = a.time_scale
    realtime = a.mode == "realtime"
    plan_sep, hop_s, sed_p95 = g["t_plan"] * ts, 2.0 * ts, g["sed_s"] * ts
    core_plan = lambda j: len(j["classes"]) * plan_sep                    # noqa: E731
    delay = lambda j: hop_s + 2 * sed_p95                                # noqa: E731
    uss.run(None, {"mixture": np.zeros((1, 1, HOP), np.float32), "condition": conds["jackhammer"]})
    jlog = Log(out / "jobs.jsonl")
    ready.set()
    start_evt.wait()
    t0 = t0_val.value
    lock = threading.Condition()
    state = {"busy_until": None, "waiting": [], "closed": False}

    def now():
        return time.monotonic() - t0

    def dispatcher():
        while True:
            job = jobs_q.get()
            with lock:
                if job is None:
                    state["closed"] = True
                    lock.notify_all()
                    return
                rec = {k: job[k] for k in ("job", "window", "classes", "arrive", "deadline")}
                if realtime:
                    t = now()
                    busy = state["busy_until"]
                    ahead = max((busy or t) - t, 0.0) + sum(core_plan(w) for w in state["waiting"])
                    idle = busy is None and not state["waiting"]
                    if not idle and len(state["waiting"]) >= g["max_wait"]:
                        jlog({**rec, "state": "dropped", "reason": "wait slot full"})
                        continue
                    if t + ahead + core_plan(job) + delay(job) > job["deadline"]:
                        jlog({**rec, "state": "dropped", "reason": "planned finish after deadline"})
                        continue
                state["waiting"].append({**job, "rec": rec})
                lock.notify_all()

    threading.Thread(target=dispatcher, daemon=True).start()
    while True:
        with lock:
            while not state["waiting"] and not state["closed"]:
                lock.wait()
            if not state["waiting"]:
                break
            job = state["waiting"].pop(0)
            t = now()
            if realtime and t + core_plan(job) + delay(job) > job["deadline"]:
                jlog({**job["rec"], "state": "expired"})
                continue
            state["busy_until"] = t + core_plan(job)
        start = now()
        for s in job["classes"]:
            q1 = time.monotonic()
            y = uss.run(None, {"mixture": job["audio"][None, None, :], "condition": conds[s]})[0].reshape(-1)
            if a.test_sep_s:
                time.sleep(max(0.0, a.test_sep_s * ts - (time.monotonic() - q1)))
            outs_q.put({"window": job["window"], "class": s, "audio": y.astype(np.float32),
                        "deadline": job["deadline"]})
        with lock:
            state["busy_until"] = None
        jlog({**job["rec"], "state": "completed", "started": start, "finished": now(),
              "wait_s": round(start - job["arrive"], 3)})
    outs_q.put(None)


def compare(out, run_dir, fz):
    """Stage-1 alerts and gated job selections must equal S4's exactly (both depend only on the SED scores, which
    are deterministic); USS recoveries are reported, allowing for queue timing. Returns True if equivalent."""
    t1, t2 = fz["t1"], fz["t2"]
    run_dir = Path(run_dir)
    mine = [json.loads(l) for l in open(out / "windows.jsonl")]
    s4 = [json.loads(l) for l in open(run_dir / "windows.jsonl")]
    if len(mine) != len(s4):
        print(f"   compare: {len(mine)} windows here vs {len(s4)} in S4: not the same audio")
        return False
    a1 = {(w["window"], s) for w in mine for s, v in w["scores"].items() if v >= t1[s]}
    a2 = {(w["window"], s) for w in s4 for s, v in w["scores"].items() if v >= t1[s]}
    worst = max(abs(w["scores"][s] - x["scores"][s]) for w, x in zip(mine, s4) for s in w["scores"])
    same_alerts = a1 == a2
    print(f"   compare with S4 {run_dir.name}: stage-1 alerts {len(a1)} here, {len(a2)} in S4 -> "
          f"{'IDENTICAL' if same_alerts else 'DIFFERENT'} (max SED score difference {worst:.1e})")
    if not same_alerts:
        print(f"     only here: {sorted(a1 - a2)[:10]}; only in S4: {sorted(a2 - a1)[:10]}")
    j1 = {w["window"]: sorted(w["job_submitted"]) for w in mine if w["job_submitted"]}
    j2 = {w["window"]: sorted(w["job_submitted"]) for w in s4 if w["job_submitted"]}
    diff = sorted(k for k in set(j1) | set(j2) if j1.get(k) != j2.get(k))
    print(f"   gated job selections: {len(j1)} windows here, {len(j2)} in S4 -> "
          f"{'IDENTICAL' if not diff else f'DIFFERENT at {len(diff)} windows, first {diff[:10]}'}")
    rec_mine = {(e["window"], e["class"]) for e in map(json.loads, open(out / "uss_events.jsonl")) if e["recovered"]}
    s4_out = [json.loads(l) for l in open(run_dir / "outputs.jsonl")]
    rec_s4 = {(o["window"], o["class"]) for o in s4_out if o["in_time"] and t2[o["class"]] is not None
              and o["score"] >= t2[o["class"]]}
    jm = sum(1 for l in open(out / "jobs.jsonl") if '"completed"' in l)
    js = sum(1 for l in open(run_dir / "jobs.jsonl") if '"completed"' in l)
    print(f"   USS recoveries: {len(rec_mine)} here, {len(rec_s4)} in S4, {len(rec_mine & rec_s4)} in both "
          f"(queue timing may differ slightly; not part of the pass/fail check); "
          f"jobs completed {jm} here vs {js} in S4")
    return same_alerts and not diff


def stop_all(procs, msg):
    for p in procs.values():
        if p.is_alive():
            p.terminate()
    for p in procs.values():
        p.join(5)
    sys.exit(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--frozen", default="frozen_design_v0_3_10.json")
    ap.add_argument("--mode", choices=["realtime", "fast"], default="realtime")
    ap.add_argument("--sed_onnx", default="cnn14_fp32_pi.onnx")
    ap.add_argument("--uss_onnx", default="resunet30_at_soft_int8_v2.onnx")
    ap.add_argument("--conditions", default="query_conditions")
    ap.add_argument("--compare_s4", default="")
    ap.add_argument("--quiet", action="store_true",
                    help="no per-event lines (summary only); P1 is then measured headless, to the event received")
    ap.add_argument("--allow_governor", action="store_true", help="run a final measurement without the "
                    "performance governor (recorded; the run is then not valid)")
    ap.add_argument("--time_scale", type=float, default=1.0, help="testing only")
    ap.add_argument("--test_sep_s", type=float, default=0.0, help="testing only")
    ap.add_argument("--no_pin", action="store_true", help="testing only")
    a = ap.parse_args()

    fz = json.loads(Path(a.frozen).read_text())
    for f in (a.audio, a.sed_onnx, a.uss_onnx):
        if not Path(f).exists():
            sys.exit(f"Missing {f}")
    realtime = a.mode == "realtime"
    final = realtime and a.time_scale == 1.0 and not a.no_pin and not a.test_sep_s   # a performance measurement
    gov = governors()
    if final and gov != ["performance"] and not a.allow_governor:
        sys.exit(f"CPU governor {gov}, locked configuration needs performance (as S4)")
    audio, orig_sr = load_audio(a.audio)
    n_win = len(audio) // HOP
    out = Path("results") / f"detect_{Path(a.audio).stem}_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    pin({2}, not a.no_pin)
    ctx = mp.get_context("fork")
    r1, r2, start_evt = ctx.Event(), ctx.Event(), ctx.Event()
    t0_val = ctx.Value("d", 0.0)
    jobs_q, outs_q, events_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    p_sed = ctx.Process(target=sed_proc, args=(a, fz, out, r1, start_evt, t0_val, jobs_q, outs_q, events_q,
                                               n_win, audio))
    p_uss = ctx.Process(target=uss_proc, args=(a, fz, out, r2, start_evt, t0_val, jobs_q, outs_q))
    procs = {"SED": p_sed, "USS": p_uss}
    p_sed.start()
    p_uss.start()

    def check_workers():
        bad = [f"{n} exit code {p.exitcode}" for n, p in procs.items() if p.exitcode not in (None, 0)]
        if bad:
            stop_all(procs, f"Worker process failed ({', '.join(bad)}); see the traceback above. Partial logs in {out}")

    load_deadline = time.monotonic() + 600
    while not (r1.is_set() and r2.is_set()):
        check_workers()
        if time.monotonic() > load_deadline:
            stop_all(procs, "Models did not load within 10 minutes")
        (r2 if r1.is_set() else r1).wait(0.5)
    t0_val.value = time.monotonic() + 1.0
    mon = Monitor(out / "monitor.jsonl", t0_val.value) if realtime else None
    if mon:
        mon.start()
    start_evt.set()
    print(f"{Path(a.audio).name}: {n_win * 2} s ({n_win} windows), mode {a.mode}, frozen design {a.frozen}")
    print(f"{'time':>8}  {'class':10} {'stage':14} {'score':>7} {'thr':>7}  delay (USS; stage-1 P1 in events.csv)")

    def next_event():
        """The next event; stops the run with an error if a worker died (instead of waiting forever)."""
        while True:
            try:
                return events_q.get(timeout=1.0)
            except queue.Empty:
                pass
            check_workers()
            if not p_sed.is_alive():           # exited cleanly but the end marker never came
                try:
                    return events_q.get(timeout=5.0)
                except queue.Empty:
                    stop_all(procs, "SED process ended without finishing the stream")

    t0 = t0_val.value
    p1_kind = "headless: alert event received, nothing printed" if a.quiet else \
        "terminal: complete alert line written and flushed to stdout"
    rows, p1, ulog = [], [], Log(out / "uss_events.jsonl")
    alerted = set()
    counts = {"stage-1 alert": 0, "USS-recovered": 0, "USS late": 0}
    while True:
        ev = next_event()
        if ev is None:
            break
        k = ev["window"]
        if ev["kind"] == "sed":
            for s in ev["alerts"]:
                alerted.add((k, s))
                if not a.quiet:
                    print(f"{2 * k:>7.0f}s  {s:10} {'stage-1 alert':14} {ev['scores'][s]:>7.4f} {ev['t1'][s]:>7.4f}",
                          flush=True)
                shown = time.monotonic() - t0             # after the flush (terminal) or on receipt (headless)
                p1_s = (shown - (k * 2.0 * a.time_scale)) / a.time_scale if realtime else None
                if p1_s is not None:
                    p1.append(p1_s)
                counts["stage-1 alert"] += 1
                rows.append({"time_s": 2 * k, "class": s, "stage": "stage-1 alert", "score": round(ev["scores"][s], 6),
                             "threshold": ev["t1"][s], "delay_s": round(p1_s, 3) if p1_s is not None else ""})
        else:
            ulog(ev)
            if ev["late"]:
                counts["USS late"] += 1
            if ev["recovered"] and (k, ev["class"]) not in alerted:
                d = (ev["scored_at"] - k * 2.0 * a.time_scale) / a.time_scale if realtime else None
                if not a.quiet:
                    print(f"{2 * k:>7.0f}s  {ev['class']:10} {'USS-recovered':14} {ev['score']:>7.4f} {ev['t2']:>7.4f}"
                          + (f"  {d:.1f} s after window start" if d is not None else ""), flush=True)
                counts["USS-recovered"] += 1
                rows.append({"time_s": 2 * k, "class": ev["class"], "stage": "USS-recovered",
                             "score": round(ev["score"], 6), "threshold": ev["t2"],
                             "delay_s": round(d, 2) if d is not None else ""})
    for p in procs.values():
        p.join(60)
    check_workers()
    perf = mon.finish() if mon else None
    rows.sort(key=lambda r: (r["time_s"], r["stage"]))
    with open(out / "events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["time_s", "class", "stage", "score", "threshold", "delay_s"])
        w.writeheader()
        w.writerows(rows)
    jobs = [json.loads(l) for l in open(out / "jobs.jsonl")]
    st = {s: sum(j["state"] == s for j in jobs) for s in ("completed", "dropped", "expired")}
    p1a = np.array(p1)
    valid = bool(final and gov == ["performance"] and perf and perf["no_throttling"])
    import onnxruntime as ort
    summary = {"audio": a.audio, "audio_sha256": sha256(a.audio), "original_sr": orig_sr, "seconds": n_win * 2,
               "mode": a.mode, "frozen": a.frozen, "frozen_sha256": sha256(a.frozen), "events": counts,
               "jobs_submitted": len(jobs), **st,
               "p1_s": {"measured_to": p1_kind, "median": float(np.median(p1a)), "p95": float(np.percentile(p1a, 95)),
                        "max": float(p1a.max())} if p1 else None,
               "final_measurement": final, "governor": gov, "monitor": perf, "valid_run": valid,
               "args": vars(a),
               "hashes": {"sed_onnx": sha256(a.sed_onnx), "uss_onnx": sha256(a.uss_onnx),
                          **{f"condition_{s}": sha256(Path(a.conditions) / f"{s}.npy") for s in CLASS_IDS}},
               "versions": {"onnxruntime": ort.__version__, "python": sys.version.split()[0]}}
    (out / "run.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone: {counts['stage-1 alert']} stage-1 alerts, {counts['USS-recovered']} USS-recovered; USS jobs "
          f"{len(jobs)} submitted, {st['completed']} completed, {st['dropped']} dropped, {st['expired']} expired, "
          f"{counts['USS late']} late results")
    if p1:
        what = "headless, to alert event received" if a.quiet else "to alert line flushed to stdout"
        print(f"P1 ({what}): median {np.median(p1a):.2f} s, p95 {np.percentile(p1a, 95):.2f} s, "
              f"max {p1a.max():.2f} s (target 3 s)")
    if perf:
        print(f"Temperature max {perf['temp_max_c']} C; throttled {perf['throttled_seen']}; lowest clock "
              f"{perf['freq_min_mhz']} MHz (board max {perf['freq_board_max_mhz']:.0f}); governor {gov}; "
              f"valid run: {valid}" + ("" if valid or not final else "  -> do not quote P1 from this run")
              + ("" if final else " (testing settings, not a performance measurement)"))
    print(f"Saved {out}/events.csv and run.json")
    if a.compare_s4 and not compare(out, a.compare_s4, fz):
        sys.exit("--compare_s4: NOT equivalent to S4 (stage-1 alerts or gated job selections differ)")


if __name__ == "__main__":
    main()
