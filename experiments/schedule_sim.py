"""Scheduled recall: does a gate's offline gain survive the Pi's USS queue?

Offline results assume every USS job finishes. On the Pi, one core runs one job at a time, so in
target-rich periods jobs are dropped, expire, or finish late. This replays the dev clips as one
continuous stream through the queue rules of the Contingency Specification and keeps the USS
result only for jobs that finish within their deadline. Uses saved scores only (exact for the
same reason as gate_analysis.py: USS treats each 2 s window on its own).

Stream: clips laid end to end (10 s each, 5 hops of 2 s). Random orders (seeded, --orders) give a
spread; one "grouped" order (clips sorted by target sound) gives a bursty worst case.

Queue rules (Contingency Specification, Week 5):
  - a job is created when the gate fires on a window, at window end + SED p95 (--sed_s)
  - job duration = queries x per-query time (--t_query, Pi median beside SED; --t_query_csv to
    sample from a Pi runs.csv instead)
  - one job runs at a time; at most one may wait (--max_wait)
  - deadline = window start + TTL (--ttl_s), standing in for event onset
  - on arrival: dropped if the wait slot is full or its predicted finish is after the deadline
  - before starting: expired if its predicted finish is now after the deadline
  - while running: never cancelled; a result after the deadline is late and ignored
Windows are never merged: the USS window equals the 2 s hop.

Scoring. Thresholds are those of the OFFLINE gated pipeline (every job completes), set at the
SED-only false-alarm count as in Gate 1, and applied unchanged to the scheduled pipeline, as a
deployed system would run (main number). The re-matched result is shown alongside.
--skip_alerting: windows whose original SED score already reaches that frozen threshold get no
job (they alert anyway), which frees the core for windows that might be recovered.

In-sample on the dev set; job times are an input, so re-run with the Week 7 Pi timings.

Usage (inside the container, from the uss folder):
  python experiments/schedule_sim.py                               # lead gate, Week 5 timing
  python experiments/schedule_sim.py --t_query 18.0 --skip_alerting
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gate_analysis import gated, load  # noqa: E402
from score_events import CONDITIONS, evaluate, match  # noqa: E402
from detect_baseline import USS_THRESHOLD  # noqa: E402
from run_gate1 import SOUNDS, TA  # noqa: E402

NAMES = list(SOUNDS)
HOP = 2.0
BUDGET = 0.7
STAT_KEYS = ("submitted", "completed", "dropped", "expired", "late", "max_queue")


def gate_masks(B, clips, lo, policy, gate_classes):
    """Floor gate as in gate_analysis.py. Returns per-class masks: fired[s][clip][k]."""
    fired = {s: {c: [s in gate_classes and b[0] >= lo for b in B[s][c]] for c in clips} for s in NAMES}
    if policy == "kmax":
        hop = {c: [any(fired[s][c][k] for s in NAMES) for k in range(len(B[NAMES[0]][c]))] for c in clips}
        fired = {s: (hop if s in gate_classes else {c: [False] * len(v) for c, v in hop.items()}) for s in NAMES}
    return fired


def simulate(order, queries, n_hops, t_query, t_plan, ttl, sed_s, max_wait, rng, sample_times, score_delay=0.0):
    """Run the queue over one stream order. queries[(clip, k)] = classes to query for that window.
    Returns the set of (clip, k, class) whose result arrived in time, and job statistics."""
    jobs = []
    t_clip = 0.0
    for c in order:
        for k in range(n_hops[c]):
            q = queries.get((c, k))
            if q:
                start = t_clip + k * HOP
                dur = sum(sample_times(rng) if sample_times else t_query for _ in q)   # actual
                jobs.append({"clip": c, "k": k, "q": q, "arrive": start + HOP + sed_s,
                             "deadline": start + ttl, "dur": dur, "plan": t_plan * len(q)})  # planned (p95)
        t_clip += n_hops[c] * HOP
    stats = Counter({k: 0 for k in STAT_KEYS})
    stats["submitted"] = len(jobs)
    done, waits, verdict = set(), [], []
    busy_until, plan_until, waiting = 0.0, 0.0, []   # running job's actual and planned finish; queue

    def start(w, t0):
        nonlocal busy_until, plan_until
        busy_until = t0 + w["dur"]
        plan_until = t0 + w["plan"]          # all the scheduler can know about the running job
        waits.append(t0 - w["arrive"])
        if busy_until + score_delay <= w["deadline"]:     # result counts once the SED cores have scored it
            verdict.append(busy_until + score_delay - (w["deadline"] - ttl))
            done.update((w["clip"], w["k"], s) for s in w["q"])
            stats["completed"] += 1
        else:
            stats["late"] += 1             # ran over its plan: result logged, alert unchanged

    def run_queue(until):
        while waiting and busy_until <= until:
            w = waiting.pop(0)
            t0 = max(busy_until, w["arrive"])
            if t0 + w["plan"] + score_delay > w["deadline"]:
                stats["expired"] += 1
                continue
            start(w, t0)

    for j in sorted(jobs, key=lambda x: x["arrive"]):
        now = j["arrive"]
        run_queue(now)
        if busy_until <= now:                # idle core (run_queue has emptied the queue)
            if now + j["plan"] + score_delay > j["deadline"]:
                stats["dropped"] += 1
            else:
                start(j, now)
            continue
        ahead = max(plan_until - now, 0.0) + sum(w["plan"] for w in waiting)   # planned, never actual
        if len(waiting) >= max_wait or now + ahead + j["plan"] + score_delay > j["deadline"]:
            stats["dropped"] += 1
        else:
            waiting.append(j)
            stats["max_queue"] = max(stats["max_queue"], len(waiting))
    run_queue(float("inf"))
    return done, stats, waits, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--pieces", default="")
    ap.add_argument("--policy", choices=["k1", "kmax"], default="kmax")
    ap.add_argument("--lo", type=float, default=0.1002)
    ap.add_argument("--exclude", nargs="*", default=["beeper"], choices=NAMES)
    ap.add_argument("--t_query", type=float, default=19.4, help="seconds per query on the USS core (Week 5)")
    ap.add_argument("--t_query_csv", default="", help="Pi runs.csv (latency_ms per k=1 job) to sample from")
    ap.add_argument("--t_plan", type=float, default=19.8,
                    help="per-query time the queue plans with (Week 5 p95); with --t_query_csv, its p95")
    ap.add_argument("--ttl_s", type=float, default=60.0)
    ap.add_argument("--sed_s", type=float, default=1.21,
                    help="SED p95 with USS running (Week 5 PyTorch: 1.21; Week 7 ONNX: 0.66)")
    ap.add_argument("--score_delay_s", type=float, default=0.0,
                    help="detector-core design: delay before a separated output is scored (up to one hop + "
                         "SED time); counts against the deadline without occupying the USS core")
    ap.add_argument("--max_wait", type=int, default=1)
    ap.add_argument("--skip_alerting", action="store_true")
    ap.add_argument("--orders", type=int, default=200)
    ap.add_argument("--order_file", default="",
                    help="s4_order.json: simulate only this exact clip order (the Pi's S4 order)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--onset_rel", type=float, default=0.5)
    a = ap.parse_args()

    ts = Path(a.test_set)
    manifest, seq, gts = load(ts, Path(a.pieces or f"results/{ts.name}_scores_pieces.csv"), a.onset_rel)
    clips = sorted(manifest)
    B = {s: seq[(s, "before")] for s in NAMES}
    U = {s: seq[(s, "after")] for s in NAMES}
    gate_classes = [s for s in NAMES if s not in a.exclude]
    anchor = {s: sum(evaluate(B[s], gts[s], USS_THRESHOLD)[1].values()) for s in NAMES}
    base_hits = {s: match(B[s], gts[s], anchor[s])[1] for s in NAMES}

    sample_times = None
    if a.t_query_csv:
        if "_k1_" not in Path(a.t_query_csv).name:
            sys.exit(f"{a.t_query_csv}: needs a k = 1 runs.csv (per-query times); jobs multiply by queries")
        with open(a.t_query_csv) as fh:
            pool = np.array([float(r["latency_ms"]) / 1000 for r in csv.DictReader(fh)])
        sample_times = lambda rng: float(rng.choice(pool))  # noqa: E731
        a.t_plan = float(np.percentile(pool, 95))
        t_desc = (f"sampled from {a.t_query_csv} (median {np.median(pool):.1f} s, mean {pool.mean():.1f} s, "
                  f"planned at p95 {a.t_plan:.1f} s)")
    else:
        t_desc = f"{a.t_query:.1f} s per query, planned at {a.t_plan:.1f} s"

    # offline gated pipeline: every job completes; its thresholds are frozen for scheduling
    mask = gate_masks(B, clips, a.lo, a.policy, gate_classes)
    off = {s: gated(B[s], U[s], mask[s]) for s in NAMES}
    thr = {s: match(off[s], gts[s], anchor[s])[0] for s in NAMES}
    off_hits = {s: evaluate(off[s], gts[s], thr[s])[0] for s in NAMES}
    queries = {}
    for c in clips:
        for k in range(len(B[NAMES[0]][c])):
            q = [s for s in NAMES if mask[s][c][k] and not (a.skip_alerting and B[s][c][k][0] >= thr[s])]
            if q:
                queries[(c, k)] = q

    events = [(s, c) for s in NAMES for c in gts[s]]

    def summarise(hits):
        rec = {}
        for cond in CONDITIONS:
            sel = [(s, c) for s, c in events if manifest[c]["condition"] == cond]
            rec[cond] = (100 * (np.mean([hits[s][c] for s, c in sel]) - np.mean([base_hits[s][c] for s, c in sel]))
                         if sel else float("nan"))
        f = sum(hits[s][c] and not base_hits[s][c] for s, c in events)
        r = sum(base_hits[s][c] and not hits[s][c] for s, c in events)
        return rec, f, r

    def scheduled(done):
        m = {s: {c: [(c, k, s) in done for k in range(len(B[s][c]))] for c in clips} for s in NAMES}
        d = {s: gated(B[s], U[s], m[s]) for s in NAMES}
        frozen = {s: evaluate(d[s], gts[s], thr[s]) for s in NAMES}
        fa = {s: sum(frozen[s][1].values()) for s in NAMES}
        rematched = {s: match(d[s], gts[s], anchor[s])[1] for s in NAMES}
        return {s: frozen[s][0] for s in NAMES}, fa, rematched

    n_jobs = len(queries)
    n_hops = {c: len(B[NAMES[0]][c]) for c in clips}
    stream_s = sum(n_hops.values()) * HOP
    hours = stream_s / 3600
    load_pk = sum(len(q) for q in queries.values()) * (pool.mean() if sample_times else a.t_query) / stream_s
    print(f"Scheduled recall: {ts.name}, {len(clips)} clips as one {hours * 60:.0f} min stream; gate {a.policy}, "
          f"lo {a.lo:g}, never gated: {', '.join(a.exclude) or 'none'}" + ("; skipping windows that already alert"
                                                                          if a.skip_alerting else ""))
    print(f"Job time {t_desc}; TTL {a.ttl_s:g} s; SED p95 {a.sed_s:g} s; one running + {a.max_wait} waiting"
          + (f"; outputs scored on the SED cores, {a.score_delay_s:g} s after separation" if a.score_delay_s else ""))
    print(f"Jobs wanted: {n_jobs} windows ({n_jobs / sum(n_hops.values()):.1%} of hops), USS core demand "
          f"{load_pk:.2f} of real time, from mean job time (the core can supply 1.0; the planning budget is "
          f"{BUDGET:g})")
    rec, f, r = summarise(off_hits)
    print("\nOffline (every job completes):   d = " + " / ".join(f"{rec[c]:+.1f}" for c in CONDITIONS)
          + f" pp   fixes/regressions {f}/{r}")

    rng = np.random.default_rng(a.seed)          # stream orders
    rng_t = np.random.default_rng(a.seed + 1)    # job times, so orders and times vary independently
    runs = []
    if a.order_file:
        fixed = json.loads(Path(a.order_file).read_text())["clips"]
        if sorted(fixed) != clips:
            sys.exit(f"{a.order_file} does not hold exactly this test set's {len(clips)} clips")
        orders = [("S4 order", fixed)]
    else:
        orders = [("grouped (worst case)", sorted(clips, key=lambda c: (manifest[c]["contains"], c)))]
        orders += [(f"random {i}", list(rng.permutation(clips))) for i in range(a.orders)]
    for name, order in orders:
        done, st, waits, verdict = simulate(order, queries, n_hops, a.t_query, a.t_plan, a.ttl_s, a.sed_s,
                                            a.max_wait, rng_t, sample_times, a.score_delay_s)
        hits, fa, rm = scheduled(done)
        rec_s, f_s, r_s = summarise(hits)
        rec_r, _, _ = summarise(rm)
        runs.append({"order": name, "rec": rec_s, "f": f_s, "r": r_s, "rec_rm": rec_r, "fa": fa, "st": st,
                     "wait_med": float(np.median(waits)) if waits else 0.0,
                     "wait_max": float(max(waits)) if waits else 0.0,
                     "verdict_max": float(max(verdict)) if verdict else 0.0})

    def line(x):
        st = x["st"]
        return (" / ".join(f"{x['rec'][c]:+.1f}" for c in CONDITIONS) + f" pp  f/r {x['f']}/{x['r']}  "
                f"jobs {st['submitted']}: done {st['completed']}, dropped {st['dropped']}, "
                f"expired {st['expired']}, late {st['late']}  (re-matched " + " / ".join(f"{x['rec_rm'][c]:+.1f}" for c in CONDITIONS)
                + ")")

    print(f"Scheduled, {'S4 order' if a.order_file else 'grouped worst case'}:   d = {line(runs[0])}")
    rand = runs[1:]
    if rand:
        mean_d = {c: np.mean([x["rec"][c] for x in rand]) for c in CONDITIONS}
        lo_d = {c: np.percentile([x["rec"][c] for x in rand], 5) for c in CONDITIONS}
        hi_d = {c: np.percentile([x["rec"][c] for x in rand], 95) for c in CONDITIONS}
        comp = [x["st"]["completed"] / max(x["st"]["submitted"], 1) for x in rand]
        print(f"Scheduled, {len(rand)} random orders: d mean " + " / ".join(f"{mean_d[c]:+.1f}" for c in CONDITIONS)
              + "  [5-95%: " + " / ".join(f"{lo_d[c]:+.1f}..{hi_d[c]:+.1f}" for c in CONDITIONS) + "]")
        print(f"   jobs completed in time: median {np.median(comp):.0%} (range {min(comp):.0%}-{max(comp):.0%}); "
              f"fixes/regressions median {np.median([x['f'] for x in rand]):.0f}/"
              f"{np.median([x['r'] for x in rand]):.0f}")
        print(f"   queue wait: median {np.median([x['wait_med'] for x in rand]):.1f} s, worst "
              f"{max(x['wait_max'] for x in rand):.1f} s; worst window-start-to-result "
              f"{max(x['verdict_max'] for x in rand):.1f} s (TTL {a.ttl_s:g} s)")
        fa_max = {s: max(x["fa"][s] for x in rand) for s in NAMES}
        print("   false alarms at the frozen thresholds, worst order: "
              + ", ".join(f"{s} {fa_max[s]} (allowed {anchor[s]})" for s in NAMES))
    print("\nIn-sample on the dev set. Main number = frozen thresholds.")

    timing = Path(a.t_query_csv).stem.replace("_runs", "") if a.t_query_csv else f"tq{a.t_query:g}-plan{a.t_plan:g}"
    extra = (("" if a.ttl_s == 60 else f"_ttl{a.ttl_s:g}") + ("" if a.max_wait == 1 else f"_wait{a.max_wait}")
             + ("" if a.sed_s == 1.21 else f"_sed{a.sed_s:g}")
             + (f"_scoredelay{a.score_delay_s:g}" if a.score_delay_s else ""))
    extra += "_s4order" if a.order_file else ""
    out = Path(f"results/{ts.name}_schedule_{a.policy}_lo{a.lo:g}" + ("_skip" if a.skip_alerting else "")
               + f"_{timing}{extra}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "jobs_wanted": n_jobs, "demand": round(float(load_pk), 3),
                               "offline": {"d": rec, "fixes": f, "regressions": r},
                               "runs": [{**{k: v for k, v in x.items() if k != "st"}, "stats": {k: x["st"][k] for k in STAT_KEYS}}
                                        for x in runs]}, indent=1, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()