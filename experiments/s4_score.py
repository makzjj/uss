"""Score an S4 Pi run (benchmarks/s4_pi.py) on the server: measured scheduled recall and job statistics.

Inputs: the run folder copied from the Pi (results/s4_<design>_<time>/) and results/s4/ (order, config).
Checks first: the run's config is the same file as results/s4/s4_config.json, its thresholds came from the
same Gate 1 scores file (sha256), the run is valid (no throttling) and its windows are aligned with the
server's (the Pi's own original-audio scores match the server's).

Recall, at the frozen thresholds from the config (deployment), for three versions of the same stream:
  Pi measured        the Pi's SED scores, plus the Pi's INT8 USS outputs scored by the Pi, for the jobs
                     that finished in time on the Pi
  Pi jobs, FP32 USS  the same finished jobs, but scored with the server's saved FP32 USS outputs
                     (Gate 1): isolates the effect of INT8 USS + ONNX SED from the effect of scheduling
  SED only           the Pi's SED scores at the SED-only thresholds (the baseline every delta is against)
plus the simulator command for the same order with the Pi's measured per-query times, which is the
prediction to compare against (its 200-order version compares the designs).

Usage (inside the container, from the uss folder):
  python experiments/s4_score.py --run results/s4_core3_20261104-230908
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gate_analysis import load  # noqa: E402
from score_events import CONDITIONS, evaluate, onset_time  # noqa: E402
from run_gate1 import SOUNDS, TA  # noqa: E402

NAMES = list(SOUNDS)
HOP = 2.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the S4 run folder copied from the Pi")
    ap.add_argument("--s4_dir", default="results/s4")
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--onset_rel", type=float, default=0.5)
    ap.add_argument("--force", action="store_true", help="score even if a check fails (give --reason)")
    ap.add_argument("--reason", default="", help="why a failed check is acceptable (saved with the result)")
    a = ap.parse_args()

    run_dir, s4 = Path(a.run), Path(a.s4_dir)
    run = json.loads((run_dir / "run.json").read_text())
    cfg = json.loads((s4 / "s4_config.json").read_text())
    order = json.loads((s4 / "s4_order.json").read_text())
    design = run["summary"]["design"]
    ts = Path(a.test_set)
    scores_csv = Path(f"results/{ts.name}_scores_pieces.csv")

    # ---------- checks ----------
    problems = []
    if run["hashes"]["config"] != sha256(s4 / "s4_config.json"):
        problems.append("the run used a different s4_config.json")
    if cfg["provenance"]["scores_sha256"] != sha256(scores_csv):
        problems.append(f"{scores_csv} changed since s4_prepare.py set the thresholds")
    if not run["summary"].get("valid_run"):
        problems.append("the run is not valid (throttling or a clock drop)")
    al, tol = run["summary"]["alignment_max_abs_vs_server"], order["expected_alignment_tolerance"]
    if al is None or al > tol:
        problems.append(f"windows misaligned (max score difference {al} > {tol})")
    if run["time_scale"] != 1.0:
        problems.append(f"time_scale {run['time_scale']} (a test run, not real time)")
    print(f"S4 {design}: {run_dir.name}")
    for p in problems:
        print(f"   CHECK FAILED: {p}")
    if problems and not a.force:
        sys.exit("Not scoring this run (use --force with --reason \"...\" only if the failure is acceptable)")
    if problems and a.force and not a.reason:
        sys.exit("--force needs --reason \"...\"; the reason is saved with the result")
    print(f"   checks passed: config and scores file match, valid run, alignment {al:.1e} (limit {tol:g})"
          if not problems else "   scoring anyway (--force)")

    # ---------- build detections ----------
    manifest, seq, gts = load(ts, scores_csv, a.onset_rel)
    U_srv = {s: seq[(s, "after")] for s in NAMES}
    win_map = {w["window"]: (w["clip"], w["piece"]) for w in order["windows"]}
    clips = sorted(manifest)
    n_hops = len(U_srv[NAMES[0]][clips[0]])
    B = {s: {c: [None] * n_hops for c in clips} for s in NAMES}
    for w in rows(run_dir / "windows.jsonl"):
        c, k = win_map[w["window"]]
        for s in NAMES:
            B[s][c][k] = (w["scores"][s], onset_time(k * HOP, " ".join(str(v) for v in w["frames"][s]), a.onset_rel))
    missing = sum(v is None for s in NAMES for c in clips for v in B[s][c])
    if missing:
        sys.exit(f"{missing} window scores missing from windows.jsonl: the run did not cover the whole stream")
    outs = rows(run_dir / "outputs.jsonl")
    done = {}                                                   # (clip, piece, class) -> (score, time), in time only
    late = 0
    for o in outs:
        if not o["in_time"]:
            late += 1
            continue
        c, k = win_map[o["window"]]
        done[(c, k, o["class"])] = (o["score"], onset_time(k * HOP, " ".join(str(v) for v in o["frames"]), a.onset_rel))

    def pipeline(uss_scores):
        return {s: {c: [(uss_scores(c, k, s) if (c, k, s) in done and uss_scores(c, k, s)[0] > B[s][c][k][0]
                         else B[s][c][k]) for k in range(n_hops)] for c in clips} for s in NAMES}

    measured = pipeline(lambda c, k, s: done[(c, k, s)])
    fp32 = pipeline(lambda c, k, s: U_srv[s][c][k])
    thr, thr_sed, allowed = cfg["thresholds_frozen"], cfg["thresholds_sed_only"], cfg["allowed_false_alarms"]
    base_hits = {s: evaluate(B[s], gts[s], thr_sed[s])[0] for s in NAMES}
    events = [(s, c) for s in NAMES for c in gts[s]]

    def score(dets, thresholds):
        hits, fa = {}, {}
        for s in NAMES:
            h, f = evaluate(dets[s], gts[s], thresholds[s])
            hits[s], fa[s] = h, sum(f.values())
        d = {cond: 100 * (np.mean([hits[s][c] for s, c in events if manifest[c]["condition"] == cond])
                          - np.mean([base_hits[s][c] for s, c in events if manifest[c]["condition"] == cond]))
             for cond in CONDITIONS}
        fx = sum(hits[s][c] and not base_hits[s][c] for s, c in events)
        rg = sum(base_hits[s][c] and not hits[s][c] for s, c in events)
        return d, fx, rg, fa

    res = {"Pi measured (INT8 USS, Pi SED)": score(measured, thr),
           "Pi jobs, FP32 USS outputs": score(fp32, thr)}
    sed_d, _, _, sed_fa = score(B, thr_sed)
    print(f"\nRecall change vs SED only (pp, ebr0 / ebr-5 / overlap) at the frozen thresholds; "
          f"f/r = fixes/regressions; false alarms vs allowed {allowed}")
    for name, (d, fx, rg, fa) in res.items():
        print(f"   {name:32} " + " / ".join(f"{d[c]:+.1f}" for c in CONDITIONS)
              + f"   f/r {fx}/{rg}   false alarms " + ", ".join(f"{s} {fa[s]}" for s in NAMES))
    sed_abs = {cond: 100 * np.mean([base_hits[s][c] for s, c in events if manifest[c]["condition"] == cond])
               for cond in CONDITIONS}
    print(f"   {'SED only (Pi), absolute recall':32} " + " / ".join(f"{sed_abs[c]:.1f}%" for c in CONDITIONS)
          + "   (baseline)   false alarms " + ", ".join(f"{s} {sed_fa[s]}" for s in NAMES))
    over = [s for s in NAMES if res["Pi measured (INT8 USS, Pi SED)"][3][s] > allowed[s]]
    if over:
        print(f"   NOTE: false alarms above the allowed count for {', '.join(over)}")

    # ---------- jobs and timing ----------
    jobs = rows(run_dir / "jobs.jsonl")
    st = defaultdict(int)
    for j in jobs:
        st[j["state"]] += 1
    comp = [j for j in jobs if j["state"] == "completed"]
    out_by_job = defaultdict(list)
    for o in outs:
        out_by_job[o["job"]].append(o["in_time"])
    in_time_jobs = sum(1 for j in comp if out_by_job[j["job"]] and all(out_by_job[j["job"]]))
    tsc = run["time_scale"]

    def stats(x):
        return (f"median {np.median(x):.2f} s, p95 {np.percentile(x, 95):.2f} s") if len(x) else "none"

    sep = np.array([q["sep_ms"] for j in comp for q in j["queries"]]) / 1000 / tsc
    scr = np.array([o["score_ms"] for o in outs]) / 1000 / tsc
    waits = np.array([j["wait_s"] for j in comp]) / tsc
    lat = np.array([(o["scored_at"] - o["window"] * HOP * tsc) / tsc for o in outs if o["in_time"]])
    print(f"\nJobs: {len(jobs)} submitted, {st['dropped']} dropped, {st['expired']} expired, {st['completed']} ran to the "
          f"end; of those, {in_time_jobs} had every result in time (= the simulator's 'done') and "
          f"{st['completed'] - in_time_jobs} had a late result (= the simulator's 'late'). Outputs {len(outs)}, late {late}")
    print(f"   per query: separation {stats(sep)}; scoring pass ({design}) {stats(scr)}")
    if design == "sed_cores":
        dl = np.array([(o["scored_at"] - o["sep_done_at"]) / tsc for o in outs if "sep_done_at" in o])
        print(f"   measured delay from separation to scored result: {stats(dl)} "
              f"(the simulator plans and counts {HOP + 2 * cfg['sed_p95_s']:.2f} s, so it is slightly pessimistic)")
    print(f"   queue wait {stats(waits)}" + (f", max {waits.max():.1f} s" if len(waits) else "")
          + f"; window start to counted (in-time) result {stats(lat)}"
          + (f", max {lat.max():.1f} s" if len(lat) else "") + f" (deadline {cfg['queue']['ttl_s']:g} s)")
    p1 = run["summary"]["p1_s"]
    print(f"   P1 (window start to alert): median {p1['median']:.2f} s, p95 {p1['p95']:.2f} s, p99 {p1['p99']:.2f} s, "
          f"max {p1['max']:.2f} s (target 3 s)")
    if design == "sed_cores":
        print("   Note: the simulator counts a two-sound job only if both results are in time; the Pi counts each "
              "output on its own. Rare, since admission includes the delay.")

    # ---------- the matching prediction ----------
    q_csv = run_dir / f"s4_{design}_k1_query_times.csv"
    qc = cfg["queue"]
    plan = qc["t_plan_sep_per_query_s"] + (qc["t_plan_score_per_output_s"] if design == "core3" else 0.0)
    delay = "" if design == "core3" else f" --score_delay_s {HOP + 2 * cfg['sed_p95_s']:.2f}"
    g = cfg["gate"]
    cmd = (f"python experiments/schedule_sim.py --order_file {s4}/s4_order.json --policy {g['policy']} --lo {g['lo']:g} "
           f"--exclude {' '.join(g['never_gated'])} --ttl_s {cfg['queue']['ttl_s']:g} --max_wait {cfg['queue']['max_wait']} "
           f"--sed_s {cfg['sed_p95_s']:g}" + (" --skip_alerting" if g["skip_alerting"] else "")
           + f" --t_query_csv {q_csv} --t_plan {plan:.2f}{delay}")   # plan exactly as the Pi queue did
    print("\nPrediction for this order with the Pi's measured per-query times (compare with the job counts above):")
    print(f"   {cmd}")
    print("   It runs the S4 order with 100 draws of the measured times: the Pi's 'every result in time' count should "
          "fall inside its 5-95% range.")
    print("   For the design comparison, run the same command without --order_file (200 random orders).")
    print("Note: thresholds were frozen on FP32 USS + PyTorch SED; this run used INT8 USS + ONNX SED. "
          "In-sample on the dev set.")

    out = Path(f"results/s4_score_{run_dir.name}.json")
    out.write_text(json.dumps({
        "run": str(run_dir), "design": design, "checks_failed": problems, "force_reason": a.reason,
        "recall": {k: {"delta_pp": v[0], "fixes": v[1], "regressions": v[2], "false_alarms": v[3]} for k, v in res.items()},
        "sed_only_false_alarms": sed_fa, "allowed_false_alarms": allowed,
        "jobs": dict(st), "jobs_all_results_in_time": in_time_jobs, "outputs": len(outs), "late": late,
        "sep_s": {"median": float(np.median(sep)), "p95": float(np.percentile(sep, 95)),
                  "mean": float(sep.mean())} if len(sep) else None,
        "score_pass_s": {"median": float(np.median(scr)), "p95": float(np.percentile(scr, 95))} if len(scr) else None,
        "sed_only_recall_pct": sed_abs,
        "p1_s": p1, "prediction_command": cmd}, indent=2, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()