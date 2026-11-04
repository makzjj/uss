"""S4: the deployed pipeline on the Pi, streamed in real time (Requirements v0.3.7, Contingency Specification).

Plays results/s4/stream.flac (the 220 dev clips end to end) as if a microphone heard it, and runs:
  SED process, cores 0-1   ONNX FP32 Cnn14, one 2 s window every 2 s, started when that window's audio has
                           fully arrived. Logs each window's start delay (for the P1 tail), its SED time and
                           its clip and frame scores. Applies the gate from s4_config.json (class-aware floor,
                           never-gated classes, skip classes already at their frozen threshold) and submits jobs.
  USS process, core 3      ONNX INT8 separator. Queue rules: one job running, at most max_wait waiting,
                           deadline = window start + TTL. On arrival a job is dropped if the wait slot is full or
                           its PLANNED finish is past the deadline; before starting, a waiting job expires if its
                           planned finish is now past the deadline; a running job is never cancelled, and a
                           result after the deadline is late (logged, not counted).
  --score_on core3         the USS process also runs the SED (1 thread) on each separated output
  --score_on sed_cores     each separated output goes to the SED process as soon as it is separated; the SED
                           scores at most one per hop, right after that hop's main window (so the main window is
                           never delayed). Planned delay after the last separation: one hop + 2 x SED p95 (~3.3 s).
  monitor, core 2          temperature, clock speed, throttling and free memory every 5 s

Writes results/s4_<design>_<time>/: run.json (settings, versions, hashes, summary), windows.jsonl,
jobs.jsonl, outputs.jsonl, monitor.jsonl, and s4_<design>_k1_query_times.csv (per-query times in the
latency_ms format schedule_sim.py --t_query_csv reads; separation plus, for core3, its SED pass).
Score on the server with experiments/s4_score.py.

Run from the benchmarks folder on the Pi, in the venv, governor at performance, nothing else running:
  nohup python -u s4_pi.py --score_on core3 > s4_core3.log 2>&1 &
  nohup python -u s4_pi.py --score_on sed_cores > s4_sed_cores.log 2>&1 &
Testing off the Pi: --time_scale 0.1 --no_pin --max_windows 40 (10x faster, no pinning, short stream).
"""
import argparse
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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sys(path, default="n/a"):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


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


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", buffering=1)

    def __call__(self, rec):
        self.fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- SED process
def sed_proc(a, cfg, out, ready, start_evt, t0_val, jobs_q, outs_q, n_win):
    pin({0, 1}, not a.no_pin)
    sed = session(a.sed_onnx, 2)
    ids = cfg["class_ids"]
    names = list(ids)
    gate, thr = cfg["gate"], cfg["thresholds_frozen"]
    hop_s = cfg["stream"]["hop_s"] * a.time_scale
    import soundfile as sf
    sfile = sf.SoundFile(str(Path(a.s4_dir) / "stream.flac"))
    sed.run(None, {"audio": np.zeros((1, HOP), np.float32)})      # warm-up, untimed
    wlog = Log(out / "windows.jsonl")
    olog = Log(out / "outputs.jsonl") if a.score_on == "sed_cores" else None
    ready.set()
    start_evt.wait()
    t0 = t0_val.value
    uss_done = False

    def score_one_output(block):
        """sed_cores design: score one separated output, if one is waiting."""
        nonlocal uss_done
        try:
            item = outs_q.get(timeout=0.0 if not block else 0.5)
        except queue.Empty:
            return False
        if item is None:
            uss_done = True
            return False
        if olog is None:
            return False
        t1 = time.monotonic()
        c, f = sed.run(None, {"audio": item["audio"][None, :]})
        t2 = time.monotonic()
        s = item["class"]
        olog({"job": item["job"], "window": item["window"], "class": s, "score": float(c[0, ids[s]]),
              "frames": [round(float(v), 4) for v in f[0, :, ids[s]]], "score_ms": round((t2 - t1) * 1000, 1),
              "scored_on": "sed_cores", "sep_done_at": item["sep_done_at"], "scored_at": t2 - t0,
              "deadline": item["deadline"], "in_time": t2 - t0 <= item["deadline"]})
        return True

    for k in range(n_win):
        sfile.seek(k * HOP)                                    # read during idle time, before the window is due
        win = sfile.read(HOP, dtype="float32")
        avail = t0 + (k + 1) * hop_s                           # the window's audio has fully arrived
        now = time.monotonic()
        if now < avail:
            time.sleep(avail - now)
        t1 = time.monotonic()
        c, f = sed.run(None, {"audio": win[None, :]})
        t2 = time.monotonic()
        scores = {s: float(c[0, ids[s]]) for s in names}
        gated_cls = [s for s in names if s not in gate["never_gated"]]
        over = [s for s in gated_cls if scores[s] >= gate["lo"]]          # as schedule_sim.gate_masks
        want = (gated_cls if over else []) if gate["policy"] == "kmax" else over
        fired = [s for s in want if not (gate["skip_alerting"] and scores[s] >= thr[s])]   # skip: already alerting
        if fired:
            jobs_q.put({"job": k, "window": k, "classes": fired, "arrive": t2 - t0,
                        "deadline": k * hop_s + cfg["queue"]["ttl_s"] * a.time_scale,
                        "audio": win})
        extra = 0.0
        if a.score_on == "sed_cores":
            t3 = time.monotonic()
            if score_one_output(block=False):
                extra = time.monotonic() - t3
        wlog({"window": k, "start_delay_ms": round((t1 - avail) * 1000, 2), "sed_ms": round((t2 - t1) * 1000, 1),
              "p1_s": round((t2 - t0) - k * hop_s, 4), "extra_score_ms": round(extra * 1000, 1),
              "scores": {s: round(scores[s], 6) for s in names},
              "frames": {s: [round(float(v), 4) for v in f[0, :, ids[s]]] for s in names},
              "job_submitted": fired})
    jobs_q.put(None)                                           # no more arrivals
    while a.score_on == "sed_cores" and not uss_done:          # keep scoring until the worker is finished
        score_one_output(block=True)


# ---------------------------------------------------------------- USS process
def uss_proc(a, cfg, out, ready, start_evt, t0_val, jobs_q, outs_q):
    pin({3}, not a.no_pin)
    uss = session(a.uss_onnx, 1)
    sed1 = session(a.sed_onnx, 1) if a.score_on == "core3" else None
    ids = cfg["class_ids"]
    conds = {s: np.load(Path("query_conditions") / f"{s}.npy").astype(np.float32).reshape(1, 527) for s in ids}
    q = cfg["queue"]
    ts = a.time_scale
    plan_sep, plan_score = q["t_plan_sep_per_query_s"] * ts, q["t_plan_score_per_output_s"] * ts
    hop_s = cfg["stream"]["hop_s"] * ts
    sed_p95 = cfg["sed_p95_s"] * ts

    def core_plan(job):      # planned time the job holds core 3
        n = len(job["classes"])
        return n * (plan_sep + plan_score) if a.score_on == "core3" else n * plan_sep

    def result_delay(job):   # sed_cores: outputs are handed over as each is separated, so the last one waits
        # for the next window (<= 1 hop), that window's main pass (SED p95) and its own scoring pass (~SED p95)
        return 0.0 if a.score_on == "core3" else hop_s + 2 * sed_p95

    uss.run(None, {"mixture": np.zeros((1, 1, HOP), np.float32), "condition": conds[next(iter(conds))]})
    if sed1:
        sed1.run(None, {"audio": np.zeros((1, HOP), np.float32)})
    jlog = Log(out / "jobs.jsonl")
    olog = Log(out / "outputs.jsonl") if a.score_on == "core3" else None
    qtimes = open(out / f"s4_{a.score_on}_k1_query_times.csv", "w", buffering=1)
    qtimes.write("run,latency_ms\n")
    ready.set()
    start_evt.wait()
    t0 = t0_val.value

    lock = threading.Condition()
    state = {"running_plan_until": None, "waiting": [], "closed": False, "nq": 0}

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
                t = now()
                busy = state["running_plan_until"]
                ahead = max((busy or t) - t, 0.0) + sum(core_plan(w) for w in state["waiting"])
                rec = {"job": job["job"], "window": job["window"], "classes": job["classes"], "arrive": job["arrive"],
                       "deadline": job["deadline"], "planned_core_s": round(core_plan(job), 3),
                       "planned_result_delay_s": round(result_delay(job), 3),
                       "queue_depth_at_arrival": len(state["waiting"]) + (busy is not None)}
                idle = busy is None and not state["waiting"]
                if not idle and len(state["waiting"]) >= q["max_wait"]:
                    jlog({**rec, "state": "dropped", "reason": "wait slot full"})
                elif t + ahead + core_plan(job) + result_delay(job) > job["deadline"]:
                    jlog({**rec, "state": "dropped", "reason": "planned finish after deadline"})
                else:
                    state["waiting"].append({**job, "rec": rec})
                    lock.notify_all()

    th = threading.Thread(target=dispatcher, daemon=True)
    th.start()
    while True:
        with lock:
            while not state["waiting"] and not state["closed"]:
                lock.wait()
            if not state["waiting"] and state["closed"]:
                break
            job = state["waiting"].pop(0)
            t = now()
            if t + core_plan(job) + result_delay(job) > job["deadline"]:
                jlog({**job["rec"], "state": "expired", "start_check_at": t})
                continue
            state["running_plan_until"] = t + core_plan(job)
        start = now()
        per_query, last_counts = [], start
        for s in job["classes"]:
            q1 = time.monotonic()
            y = uss.run(None, {"mixture": job["audio"][None, None, :], "condition": conds[s]})[0].reshape(-1)
            if a.test_sep_s:
                time.sleep(max(0.0, a.test_sep_s * ts - (time.monotonic() - q1)))
            q2 = time.monotonic()
            entry = {"class": s, "sep_ms": round((q2 - q1) * 1000, 1)}
            if sed1 is not None:                               # core3 design: score here, raw output
                c, f = sed1.run(None, {"audio": y[None, :].astype(np.float32)})
                q3 = time.monotonic()
                entry["score_ms"] = round((q3 - q2) * 1000, 1)
                last_counts = q3 - t0
                olog({"job": job["job"], "window": job["window"], "class": s, "score": float(c[0, ids[s]]),
                      "frames": [round(float(v), 4) for v in f[0, :, ids[s]]], "score_ms": entry["score_ms"],
                      "scored_on": "core3", "scored_at": q3 - t0, "deadline": job["deadline"],
                      "in_time": q3 - t0 <= job["deadline"]})
            else:                                              # sed_cores design: hand over the raw output
                outs_q.put({"job": job["job"], "window": job["window"], "class": s, "audio": y.astype(np.float32),
                            "sep_done_at": q2 - t0, "deadline": job["deadline"]})
            per_query.append(entry)
            state["nq"] += 1
            qtimes.write(f"{state['nq']},{(entry['sep_ms'] + entry.get('score_ms', 0.0)) / ts:.1f}\n")
        end = now()
        with lock:
            state["running_plan_until"] = None
        jlog({**job["rec"], "state": "completed", "started": start, "finished": end, "wait_s": round(start - job["arrive"], 3),
              "duration_s": round(end - start, 3), "queries": per_query,
              "late": (last_counts > job["deadline"]) if sed1 is not None else None})
    if a.score_on == "sed_cores":
        outs_q.put(None)


# ---------------------------------------------------------------- monitor + main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_on", choices=["core3", "sed_cores"], required=True)
    ap.add_argument("--s4_dir", default="s4", help="folder with stream.flac, s4_config.json, s4_order.json")
    ap.add_argument("--sed_onnx", default="cnn14_fp32_pi.onnx")
    ap.add_argument("--uss_onnx", default="resunet30_at_soft_int8_v2.onnx")
    ap.add_argument("--time_scale", type=float, default=1.0, help="testing only: < 1 runs faster than real time")
    ap.add_argument("--max_windows", type=int, default=0, help="testing only: stop after this many windows")
    ap.add_argument("--no_pin", action="store_true", help="testing only: no core pinning")
    ap.add_argument("--test_sep_s", type=float, default=0.0,
                    help="testing only: make each separation take this many (real-time) seconds")
    ap.add_argument("--allow_governor", action="store_true")
    ap.add_argument("--cool_to_c", type=float, default=60.0, help="wait for this temperature before starting")
    ap.add_argument("--min_free_mb", type=float, default=1500.0)
    a = ap.parse_args()

    import soundfile as sf
    s4 = Path(a.s4_dir)
    cfg = json.loads((s4 / "s4_config.json").read_text())
    order = json.loads((s4 / "s4_order.json").read_text())
    gov = sorted({read_sys(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor") for i in range(4)})
    if gov != ["performance"] and not a.allow_governor:
        sys.exit(f"CPU governor {gov}, locked configuration needs performance")
    for f in (a.sed_onnx, a.uss_onnx, s4 / "stream.flac"):
        if not Path(f).exists():
            sys.exit(f"Missing {f}")
    info = sf.info(str(s4 / "stream.flac"))
    if info.samplerate != SR or info.channels != 1 or info.frames != len(order["clips"]) * order["clip_samples"]:
        sys.exit("stream.flac does not match s4_order.json (rate, channels or length)")
    n_win = info.frames // HOP if a.max_windows <= 0 else min(a.max_windows, info.frames // HOP)
    mem_free = next((int(l.split()[1]) / 1024 for l in read_sys("/proc/meminfo").splitlines()
                     if l.startswith("MemAvailable")), None)
    if mem_free is not None and mem_free < a.min_free_mb:
        sys.exit(f"Only {mem_free:.0f} MB free (need {a.min_free_mb}); close other programs first")
    temp = lambda: int(read_sys("/sys/class/thermal/thermal_zone0/temp", "0")) / 1000  # noqa: E731
    waited = 0
    while temp() > a.cool_to_c and waited < 900:               # same start temperature for both designs
        if waited == 0:
            print(f"Cooling from {temp():.1f} C to {a.cool_to_c} C ...")
        time.sleep(10)
        waited += 10
    out = Path("results") / f"s4_{a.score_on}_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    pin({2}, not a.no_pin)

    ctx = mp.get_context("fork")
    ready_sed, ready_uss, start_evt = ctx.Event(), ctx.Event(), ctx.Event()
    t0_val = ctx.Value("d", 0.0)
    jobs_q, outs_q = ctx.Queue(), ctx.Queue()
    p_sed = ctx.Process(target=sed_proc, args=(a, cfg, out, ready_sed, start_evt, t0_val, jobs_q, outs_q, n_win))
    p_uss = ctx.Process(target=uss_proc, args=(a, cfg, out, ready_uss, start_evt, t0_val, jobs_q, outs_q))
    p_sed.start()
    p_uss.start()
    mem0 = read_sys("/proc/meminfo")
    if not (ready_sed.wait(600) and ready_uss.wait(600)):
        sys.exit("Models did not load within 10 minutes")
    t0_val.value = time.monotonic() + 1.0
    start_evt.set()
    print(f"S4 {a.score_on}: {n_win} windows, about {n_win * 2 * a.time_scale / 60:.1f} min, "
          f"{mem_free:.0f} MB free at start, {temp():.1f} C. Log in {out}")

    mon = Log(out / "monitor.jsonl")
    last_vc, throttled = -1e9, ""
    while p_sed.is_alive() or p_uss.is_alive():                # light: /sys reads each second, vcgencmd every 30 s
        if time.monotonic() - last_vc >= 30:
            throttled = os.popen("vcgencmd get_throttled 2>/dev/null").read().strip()
            last_vc = time.monotonic()
        mon({"t": round(time.monotonic() - t0_val.value, 1), "temp_c": temp(),
             "freq_mhz": [int(read_sys(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq", "0")) / 1000
                          for i in range(4)],
             "throttled": throttled,
             "mem_available_mb": next((int(l.split()[1]) / 1024 for l in read_sys("/proc/meminfo").splitlines()
                                       if l.startswith("MemAvailable")), None)})
        time.sleep(1.0)
    throttled_end = os.popen("vcgencmd get_throttled 2>/dev/null").read().strip()
    p_sed.join()
    p_uss.join()
    if p_sed.exitcode or p_uss.exitcode:
        sys.exit(f"A process failed (SED exit {p_sed.exitcode}, USS exit {p_uss.exitcode}); see the log")

    # ---------- summary ----------
    rd = lambda n: [json.loads(l) for l in open(out / n)] if (out / n).exists() else []  # noqa: E731
    wins, jobs, outs, mon_rows = rd("windows.jsonl"), rd("jobs.jsonl"), rd("outputs.jsonl"), rd("monitor.jsonl")
    delays = np.array([w["start_delay_ms"] for w in wins])
    p1 = np.array([w["p1_s"] for w in wins]) / a.time_scale
    states = {s: sum(j["state"] == s for j in jobs) for s in ("completed", "dropped", "expired")}
    late = sum(1 for o in outs if not o["in_time"])
    diffs = [abs(w["scores"][s] - order["windows"][w["window"]]["server_before"][s])
             for w in wins for s in w["scores"]]
    import onnxruntime as ort
    summary = {
        "design": a.score_on, "windows": len(wins), "jobs_submitted": len(jobs), **states,
        "outputs_scored": len(outs), "outputs_late": late,
        "p1_s": {"median": float(np.median(p1)), "p95": float(np.percentile(p1, 95)),
                 "p99": float(np.percentile(p1, 99)), "max": float(p1.max())},
        "sed_start_delay_ms": {"p99": float(np.percentile(delays, 99)), "max": float(delays.max())},
        "alignment_max_abs_vs_server": float(max(diffs)) if diffs else None,
        "alignment_tolerance": order["expected_alignment_tolerance"],
        "temp_max_c": max(m["temp_c"] for m in mon_rows) if mon_rows else None,
        "throttled_seen": sorted({m["throttled"] for m in mon_rows} | {throttled_end}),
        "freq_min_mhz": min((min(m["freq_mhz"]) for m in mon_rows), default=None),
        "mem_available_min_mb": min((m["mem_available_mb"] for m in mon_rows if m["mem_available_mb"]), default=None),
    }
    clean = [t for t in summary["throttled_seen"] if t]
    summary["valid_run"] = all(t.endswith("0x0") for t in clean) and (summary["freq_min_mhz"] or 0) >= 1700
    run = {"summary": summary, "args": vars(a), "config": cfg, "time_scale": a.time_scale,
           "hashes": {"sed_onnx": sha256(a.sed_onnx), "uss_onnx": sha256(a.uss_onnx),
                      "stream": sha256(s4 / "stream.flac"), "config": sha256(s4 / "s4_config.json")},
           "versions": {"onnxruntime": ort.__version__, "python": sys.version.split()[0]},
           "governor": gov, "mem_before": next((l for l in mem0.splitlines() if l.startswith("MemAvailable")), "")}
    (out / "run.json").write_text(json.dumps(run, indent=2))
    ok_align = summary["alignment_max_abs_vs_server"] is not None and \
        summary["alignment_max_abs_vs_server"] <= order["expected_alignment_tolerance"]
    print(f"\nDone. Windows {len(wins)}; jobs {len(jobs)}: completed {states['completed']}, dropped {states['dropped']}, "
          f"expired {states['expired']}; outputs scored {len(outs)}, late {late}")
    print(f"P1 (window start to alert): median {summary['p1_s']['median']:.2f} s, p95 {summary['p1_s']['p95']:.2f} s, "
          f"p99 {summary['p1_s']['p99']:.2f} s, max {summary['p1_s']['max']:.2f} s (target 3 s); "
          f"SED start delay p99 {summary['sed_start_delay_ms']['p99']:.0f} ms, "
          f"max {summary['sed_start_delay_ms']['max']:.0f} ms")
    print(f"Alignment vs server scores: max {summary['alignment_max_abs_vs_server']:.1e} "
          f"({'OK' if ok_align else 'NOT OK: windows may be misaligned; do not score this run'})")
    print(f"Temperature max {summary['temp_max_c']} C; throttled {summary['throttled_seen']}; lowest clock "
          f"{summary['freq_min_mhz']} MHz; lowest free memory {summary['mem_available_min_mb']:.0f} MB; "
          f"valid run: {summary['valid_run']}")
    print(f"Copy {out} to the server and run experiments/s4_score.py on it.")


if __name__ == "__main__":
    main()