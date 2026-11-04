"""Weeks 9-10: the two-threshold design, on development data (Requirements v0.3.8, Gate 2 follow-up).

Single threshold (S4 so far): one frozen threshold per class for the whole pipeline, set as if every USS job
completes. When jobs are dropped, windows keep their SED score but face that higher threshold, so the
pipeline can fall BELOW SED alone (regressions in bursty streams).

Two thresholds: per class,
  t1 (stage-1)        the SED-only threshold, unchanged: stage-1 alerts are exactly SED alone's
  t2 (USS-recovered)  a USS result counts only if its score reaches t2; t2 is the lowest value that keeps the
                      total false alarms within the Gate 1 budget (SED-only count at 0.2), set offline with
                      every gated job completed
A window detects if SED >= t1, or if a USS result arrived in time and USS >= t2 (the event time comes from
whichever crossed). Because stage-1 is SED alone and scheduling only removes USS results, the scheduled
pipeline can never lose an event SED alone catches (no regressions), and its false alarms never exceed the
offline count. What remains to measure is how many fixes survive at that t2.

Budget: t1 already uses the SED-only false-alarm count, so USS may add only the SLACK = budget - false alarms
at t1 (often 0): t2 then sits just above the highest USS score on a false-alarm window, and the result is
"fixes that survive with no added false alarms". The slack is printed per class.

The gate is kept as tuned for the single-threshold design (class-aware floor lo, never-gated classes); it may
not be the best gate for two thresholds. The skip rule uses the STAGE-1 threshold (skip a class whose SED score
already reaches t1). Compared offline and under the queue (schedule_sim.simulate, Pi-measured job times, 200
random orders, the same seed per order for both designs; the designs queue different jobs, so their time draws
are not paired), with the single-threshold design. t2 is set offline only, never from the queue runs.
In-sample: t2 is chosen on these clips, so the result is an upper bound; a sensitivity check raises each t2 one
step (to the next candidate score) and reports how many offline fixes depend on t2 sitting at that edge.

Usage (inside the container, from the uss folder):
  python experiments/two_threshold.py --t_query_csv results/s4_sed_cores_20261104-235255/s4_sed_cores_k1_query_times.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gate_analysis import gated, load  # noqa: E402
from schedule_sim import NAMES, gate_masks, simulate  # noqa: E402
from score_events import CONDITIONS, evaluate, match  # noqa: E402
from detect_baseline import USS_THRESHOLD  # noqa: E402
from run_gate1 import TA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--policy", choices=["k1", "kmax"], default="kmax")
    ap.add_argument("--lo", type=float, default=0.1002)
    ap.add_argument("--exclude", nargs="*", default=["beeper"], choices=NAMES)
    ap.add_argument("--t_query_csv", required=True, help="Pi-measured per-query times (an S4 _k1_ file)")
    ap.add_argument("--t_plan", type=float, default=9.46, help="planned time per query (S4 sed_cores design)")
    ap.add_argument("--score_delay_s", type=float, default=3.32, help="0 for the core3 design")
    ap.add_argument("--sed_s", type=float, default=0.66)
    ap.add_argument("--ttl_s", type=float, default=60.0)
    ap.add_argument("--max_wait", type=int, default=1)
    ap.add_argument("--orders", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--onset_rel", type=float, default=0.5)
    a = ap.parse_args()

    name = Path(a.t_query_csv).name
    if "_k1_" not in name:
        sys.exit(f"{name}: needs per-query (k = 1) times, such as an S4 *_k1_query_times.csv")
    if "core3" in name and a.score_delay_s:
        sys.exit("core3 times already include the scoring pass: use --score_delay_s 0 --t_plan 10.71")
    ts = Path(a.test_set)
    manifest, seq, gts = load(ts, Path(f"results/{ts.name}_scores_pieces.csv"), a.onset_rel)
    clips = sorted(manifest)
    B = {s: seq[(s, "before")] for s in NAMES}
    U = {s: seq[(s, "after")] for s in NAMES}
    n_hops = {c: len(B[NAMES[0]][c]) for c in clips}
    gate_classes = [s for s in NAMES if s not in a.exclude]
    anchor = {s: sum(evaluate(B[s], gts[s], USS_THRESHOLD)[1].values()) for s in NAMES}
    base_hits = {s: match(B[s], gts[s], anchor[s])[1] for s in NAMES}
    t1 = {s: match(B[s], gts[s], anchor[s])[0] for s in NAMES}
    mask = gate_masks(B, clips, a.lo, a.policy, gate_classes)

    # ---------- single-threshold design (as S4) ----------
    off_single = {s: gated(B[s], U[s], mask[s]) for s in NAMES}
    thr_single = {s: match(off_single[s], gts[s], anchor[s])[0] for s in NAMES}

    def queries_for(skip_thr):
        q = {}
        for c in clips:
            for k in range(n_hops[c]):
                cl = [s for s in NAMES if mask[s][c][k] and B[s][c][k][0] < skip_thr[s]]
                if cl:
                    q[(c, k)] = cl
        return q

    q_single, q_two = queries_for(thr_single), queries_for(t1)

    def single_dets(done):
        m = {s: {c: [(c, k, s) in done for k in range(n_hops[c])] for c in clips} for s in NAMES}
        return {s: gated(B[s], U[s], m[s]) for s in NAMES}

    # ---------- two-threshold design ----------
    def two_dets_one(s, done, t2s):
        """One class: per window the larger of SED/t1 and (if USS arrived) USS/t2; detects where that is >= 1."""
        out = {}
        for c in clips:
            row = []
            for k in range(n_hops[c]):
                b = B[s][c][k]
                cand = (b[0] / t1[s], b[1])
                if (c, k, s) in done and np.isfinite(t2s):
                    u = U[s][c][k]
                    if u[0] / t2s > cand[0]:
                        cand = (u[0] / t2s, u[1])
                row.append(cand)
            out[c] = row
        return out

    def two_dets(done, t2):
        return {s: two_dets_one(s, done, t2[s]) for s in NAMES}

    all_two = {(c, k, s) for (c, k), cl in q_two.items() for s in cl}
    t2, t2_next, cand_lists = {}, {}, {}
    for s in NAMES:
        cands = sorted({U[s][c][k][0] for (c, k, s2) in all_two if s2 == s})
        cands = [v for v in cands if v > 0] + [np.inf]
        lo_i, hi_i = 0, len(cands) - 1                 # false alarms fall as t2 rises: binary search
        while lo_i < hi_i:
            mid = (lo_i + hi_i) // 2
            fa = sum(evaluate(two_dets_one(s, all_two, cands[mid]), gts[s], 1.0)[1].values())
            if fa <= anchor[s]:
                hi_i = mid
            else:
                lo_i = mid + 1
        t2[s] = cands[lo_i]
        t2_next[s] = cands[min(lo_i + 1, len(cands) - 1)]      # one step stricter, for the sensitivity check
        cand_lists[s] = len(cands) - 1

    def summarise(dets, thrs):
        hits, fas = {}, {}
        for s in NAMES:
            h, f = evaluate(dets[s], gts[s], thrs[s])
            hits[s], fas[s] = h, sum(f.values())
        ev = [(s, c) for s in NAMES for c in gts[s]]
        d = {cond: 100 * (np.mean([hits[s][c] for s, c in ev if manifest[c]["condition"] == cond])
                          - np.mean([base_hits[s][c] for s, c in ev if manifest[c]["condition"] == cond]))
             for cond in CONDITIONS}
        fx = sum(hits[s][c] and not base_hits[s][c] for s, c in ev)
        rg = sum(base_hits[s][c] and not hits[s][c] for s, c in ev)
        return d, fx, rg, fas

    ones = {s: 1.0 for s in NAMES}
    print(f"Two-threshold design: {ts.name}, gate {a.policy} lo {a.lo:g}, never gated {a.exclude}; dev data, in-sample")
    fa_t1 = {s: sum(evaluate(B[s], gts[s], t1[s])[1].values()) for s in NAMES}
    print(f"   {'class':11}{'t1 = SED-only':>15}{'t2 = USS-recovered':>22}{'single threshold':>18}"
          f"{'FA budget':>11}{'FA at t1':>10}{'slack':>7}")
    for s in NAMES:
        if s in a.exclude:
            t2_txt = "not gated"
        elif not np.isfinite(t2[s]):
            t2_txt = "none (no FA room)"
        else:
            t2_txt = f"{t2[s]:.4f}"
        print(f"   {s:11}{t1[s]:>15.4f}{t2_txt:>22}{thr_single[s]:>18.4f}{anchor[s]:>11}{fa_t1[s]:>10}"
              f"{anchor[s] - fa_t1[s]:>7}")

    def show(label, res):
        d, fx, rg, fas = res
        return (f"   {label:34} d = " + " / ".join(f"{d[c]:+.1f}" for c in CONDITIONS) + f" pp   f/r {fx}/{rg}   "
                f"false alarms " + ", ".join(f"{s} {fas[s]}" for s in NAMES))

    all_single = {(c, k, s) for (c, k), cl in q_single.items() for s in cl}
    print("\nOffline (every gated job completes):")
    off_s = summarise(single_dets(all_single), thr_single)
    off_t = summarise(two_dets(all_two, t2), ones)
    off_n = summarise(two_dets(all_two, t2_next), ones)
    print(show("single threshold", off_s))
    print(show("two thresholds", off_t))
    print(show("two thresholds, t2 one step up", off_n) + "   <- in-sample sensitivity")

    with open(a.t_query_csv) as fh:
        pool = np.array([float(r["latency_ms"]) / 1000 for r in csv.DictReader(fh)])
    sample = lambda rng: float(rng.choice(pool))  # noqa: E731
    rng_o, rng_t = np.random.default_rng(a.seed), np.random.default_rng(a.seed + 1)
    runs = {"single threshold": [], "two thresholds": []}
    for _ in range(a.orders):
        order = list(rng_o.permutation(clips))
        state = rng_t.bit_generator.state
        for name, q in (("single threshold", q_single), ("two thresholds", q_two)):
            rng_t.bit_generator.state = state          # same job-time draws for both designs on this order
            done, st, _, _ = simulate(order, q, n_hops, 0.0, a.t_plan, a.ttl_s, a.sed_s, a.max_wait, rng_t,
                                      sample, a.score_delay_s, 0.0)
            res = summarise(single_dets(done), thr_single) if name == "single threshold" \
                else summarise(two_dets(done, t2), ones)
            runs[name].append((res, st["completed"], st["submitted"]))
    print(f"\nScheduled ({a.orders} random orders, Pi-measured job times, the same seed per order for both designs):")
    sched = {}
    for name, rs in runs.items():
        d = {c: np.array([r[0][0][c] for r in rs]) for c in CONDITIONS}
        fx = np.array([r[0][1] for r in rs])
        rg = np.array([r[0][2] for r in rs])
        fa = {s: max(r[0][3][s] for r in rs) for s in NAMES}
        print(f"   {name:18} d mean " + " / ".join(f"{d[c].mean():+.1f}" for c in CONDITIONS)
              + "  [5-95%: " + " / ".join(f"{np.percentile(d[c], 5):+.1f}..{np.percentile(d[c], 95):+.1f}"
                                          for c in CONDITIONS)
              + f"]   fixes mean {fx.mean():.1f}, regressions mean {rg.mean():.1f} (max {rg.max()})   "
              f"jobs done {np.mean([r[1] for r in rs]):.1f}/{rs[0][2]}   worst-order false alarms "
              + ", ".join(f"{s} {fa[s]}" for s in NAMES))
        sched[name] = {"delta_mean_pp": {c: float(d[c].mean()) for c in CONDITIONS},
                       "delta_p5_pp": {c: float(np.percentile(d[c], 5)) for c in CONDITIONS},
                       "delta_p95_pp": {c: float(np.percentile(d[c], 95)) for c in CONDITIONS},
                       "fixes_mean": float(fx.mean()), "regressions_mean": float(rg.mean()),
                       "regressions_max": int(rg.max()), "jobs_done_mean": float(np.mean([r[1] for r in rs])),
                       "jobs_wanted": rs[0][2], "false_alarms_worst_order": fa}
    print("\nTwo thresholds: regressions are 0 by construction (stage-1 = SED alone); false alarms cannot exceed "
          "the offline count. In-sample on the dev set: t2 is chosen on these clips.")
    out = Path(f"results/{ts.name}_two_threshold.json")
    res = lambda r: {"delta_pp": r[0], "fixes": r[1], "regressions": r[2], "false_alarms": r[3]}  # noqa: E731
    out.write_text(json.dumps({
        "t1": t1, "t2": {s: (t2[s] if np.isfinite(t2[s]) else None) for s in NAMES},
        "t2_one_step_up": {s: (t2_next[s] if np.isfinite(t2_next[s]) else None) for s in NAMES},
        "thr_single": thr_single, "allowed_false_alarms": anchor, "false_alarms_at_t1": fa_t1,
        "offline": {"single": res(off_s), "two": res(off_t), "two_t2_one_step_up": res(off_n)},
        "scheduled": sched, "args": vars(a)}, indent=2, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()