"""Week 11 evaluation (Requirements v0.3.10): every system in one pass, with every threshold frozen beforehand.

prepare   (development data only, before the held-out run)
          From the combined development set (devplus_v1), compute the comparators' thresholds by the SAME rules as
          the frozen design and save them next to it (data_notes/comparators_v0_3_10.json):
            always-on USS, single threshold   every window, every class queried; max fusion; the lowest threshold
                                              with false alarms <= the SED-only count at 0.2 (Gate 1 style)
            always-on USS, two thresholds     every window queried for every class not already alerting at t1;
                                              t2 per class by the two-threshold rule (separates the gate's effect
                                              from the threshold design's)
evaluate  (one test set; held-out needs --confirm_heldout, verifies heldout_lock.json, and refuses a second run
          unless --rerun_reason is given)
          Systems: SED alone (t1); frozen selective design; both always-on variants; random gating matched on the
          number of queries (same eligible windows: jackhammer below t1; --trials draws); selective under the queue
          (Pi-measured job times, --orders random orders). Reported on the main subset: recall per condition and
          class, change versus SED alone, fixes, regressions, false alarms (per class, all clips); jackhammer fixes
          per target recording; a bootstrap interval over target recordings for the selective design's pooled change.

Usage (inside the container, from the uss folder):
  python experiments/heldout_eval.py prepare --dev_set <TA>/testsets/devplus_v1 \\
      --frozen data_notes/frozen_design_v0_3_10.json --out data_notes/comparators_v0_3_10.json
  python experiments/heldout_eval.py evaluate --test_set <TA>/testsets/heldout_v1 --confirm_heldout \\
      --frozen data_notes/frozen_design_v0_3_10.json --comparators data_notes/comparators_v0_3_10.json \\
      --t_query_csv results/s4_sed_cores_20261104-235255/s4_sed_cores_k1_query_times.csv
Queue and onset settings are read from the frozen file. Held-out is detected from settings.json, the set name, or
target recordings that appear in heldout_lock.json. A held-out rerun (--rerun_reason) is saved to its own file.
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gate_analysis import load  # noqa: E402
from schedule_sim import NAMES, gate_masks, simulate  # noqa: E402
from score_events import CONDITIONS, evaluate  # noqa: E402
from detect_baseline import USS_THRESHOLD  # noqa: E402
from run_gate1 import TA  # noqa: E402

INF = float("inf")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fin(x):
    return INF if x is None else x


class Data:
    def __init__(self, ts, onset_rel):
        self.ts = Path(ts)
        self.manifest, seq, self.gts = load(self.ts, Path(f"results/{self.ts.name}_scores_pieces.csv"), onset_rel)
        self.clips = sorted(self.manifest)
        self.B = {s: seq[(s, "before")] for s in NAMES}
        self.U = {s: seq[(s, "after")] for s in NAMES}
        self.n = {c: len(self.B[NAMES[0]][c]) for c in self.clips}
        self.anchor = {s: sum(evaluate(self.B[s], self.gts[s], USS_THRESHOLD)[1].values()) for s in NAMES}

    def dets(self, queried, t1, t2=None, single=None):
        """Per class and clip, [(score ratio, time)]: two thresholds max(B/t1, U/t2) where queried; or, with
        single, max(B, U where queried)/single (one shared threshold)."""
        out = {}
        for s in NAMES:
            out[s] = {}
            for c in self.clips:
                row = []
                for k in range(self.n[c]):
                    b, u, q = self.B[s][c][k], self.U[s][c][k], (c, k, s) in queried
                    if single is not None:
                        best = u if (q and u[0] > b[0]) else b
                        row.append((best[0] / single[s], best[1]))
                    else:
                        cand = (b[0] / t1[s], b[1])
                        if q and np.isfinite(fin(t2[s])) and u[0] / fin(t2[s]) > cand[0]:
                            cand = (u[0] / fin(t2[s]), u[1])
                        row.append(cand)
                out[s][c] = row
        return out

    def score(self, d):
        hits, fa = {}, {}
        for s in NAMES:
            h, f = evaluate(d[s], self.gts[s], 1.0)
            hits[s], fa[s] = h, sum(f.values())
        return hits, fa


def t2_rule(D, queried, t1, s):
    """Lowest t2 keeping class s's false alarms within the SED-only budget (binary search; FA fall as t2 rises)."""
    cands = sorted({D.U[s][c][k][0] for (c, k, s2) in queried if s2 == s and D.U[s][c][k][0] > 0}) + [INF]
    lo, hi = 0, len(cands) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        t2 = {x: INF for x in NAMES}
        t2[s] = cands[mid]
        if D.score(D.dets(queried, t1, t2))[1][s] <= D.anchor[s]:
            hi = mid
        else:
            lo = mid + 1
    return cands[lo]


def single_rule(D, queried, s):
    cands = sorted({max(D.B[s][c][k][0], D.U[s][c][k][0] if (c, k, s) in queried else 0) for c in D.clips
                    for k in range(D.n[c])} | {1.01})
    lo, hi = 0, len(cands) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        th = {x: 1.0 for x in NAMES}
        th[s] = cands[mid]
        if D.score(D.dets(queried, None, single=th))[1][s] <= D.anchor[s]:
            hi = mid
        else:
            lo = mid + 1
    return cands[lo]


def prepare(a):
    fz = json.loads(Path(a.frozen).read_text())
    D = Data(a.dev_set, fz["args"].get("onset_rel", 0.5))
    if D.ts.name != Path(fz["args"]["test_set"]).name:
        sys.exit(f"--dev_set should be the set the design was frozen on ({Path(fz['args']['test_set']).name})")
    t1 = fz["t1"]
    all_q = {(c, k, s) for c in D.clips for k in range(D.n[c]) for s in NAMES}
    below_t1 = {(c, k, s) for (c, k, s) in all_q if D.B[s][c][k][0] < t1[s]}
    thr_always = {s: single_rule(D, all_q, s) for s in NAMES}
    t2_always = {s: t2_rule(D, below_t1, t1, s) for s in NAMES}
    out = {"made": datetime.now().isoformat(timespec="seconds"), "dev_set": str(D.ts), "frozen": a.frozen,
           "frozen_sha256": sha256(a.frozen), "anchor": D.anchor,
           "always_on_single_threshold": thr_always,
           "always_on_two_thresholds_t2": {s: (None if not np.isfinite(v) else v) for s, v in t2_always.items()}}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"Comparator thresholds from {D.ts.name} (same rules as the frozen design):")
    for s in NAMES:
        print(f"   {s:11} always-on single {thr_always[s]:.4f}   always-on two-threshold t2 "
              f"{'none' if not np.isfinite(t2_always[s]) else f'{t2_always[s]:.4f}'}")
    print(f"Saved {a.out}: commit it with the frozen design BEFORE the held-out run")


def evaluate_set(a):
    fz = json.loads(Path(a.frozen).read_text())
    cp = json.loads(Path(a.comparators).read_text())
    if cp["frozen_sha256"] != sha256(a.frozen):
        sys.exit(f"{a.comparators} was prepared from a different frozen file than {a.frozen}")
    ts = Path(a.test_set)
    st = json.loads((ts / "settings.json").read_text()) if (ts / "settings.json").exists() else {}
    fa_args = fz["args"]
    frozen_on = Path(fa_args["test_set"])
    # 4. refuse the frozen set and every set merged into it (in-sample, not independent evidence)
    fst = frozen_on / "settings.json"
    in_sample = {frozen_on.name} | ({Path(x).name for x in json.loads(fst.read_text()).get("merged_from", [])}
                                    if fst.exists() else set())
    if ts.name in in_sample:
        sys.exit(f"{ts.name} is development data the design was frozen on ({sorted(in_sample)}); not independent")
    # 2. held-out if settings say so, OR the name says so, OR its target recordings are locked held-out files
    lock_path = Path(a.lock)
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {"files": []}
    locked_names = {Path(f["file_32k"]).name for f in lock["files"]}
    with open(ts / "manifest.csv") as fh:
        srcs = {r.get("target_source", "") for r in csv.DictReader(fh)}
    heldout = (st.get("target_set") == "heldout" or st.get("background_set") == "heldout"
               or "heldout" in ts.name.lower() or bool(srcs & locked_names))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    first_path = Path(f"results/heldout_eval_{ts.name}.json")
    out_path = first_path
    if heldout:
        if not a.confirm_heldout:
            sys.exit("This is the held-out set: add --confirm_heldout (it is evaluated once)")
        if not lock["files"]:
            sys.exit(f"Held-out set but no lock at {lock_path}: cannot verify the files are unchanged")
        base = lock_path.resolve().parent                       # 5. relative lock paths resolve from the lock
        path = lambda f: Path(f) if Path(f).is_absolute() else (base / f).resolve()  # noqa: E731
        bad = [f["file_32k"] for f in lock["files"]
               if not path(f["file_32k"]).exists() or sha256(path(f["file_32k"])) != f["sha256_32k"]]
        if bad:
            sys.exit(f"{len(bad)} held-out files changed since locking, e.g. {bad[0]}")
        if first_path.exists():
            if not a.rerun_reason:
                sys.exit(f"{first_path} exists: the held-out set was already evaluated. A rerun needs "
                         f"--rerun_reason \"...\" and is saved to a separate file; the first result is never replaced")
            out_path = Path(f"results/heldout_eval_{ts.name}_rerun_{stamp}.json")     # 1. never overwrite
    elif first_path.exists():
        out_path = Path(f"results/heldout_eval_{ts.name}_{stamp}.json")
    # 3. queue and onset settings come from the frozen design, not from this script
    q = {k: fa_args[k] for k in ("t_plan", "score_delay_s", "sed_s", "ttl_s", "max_wait")}
    onset_rel = fa_args.get("onset_rel", 0.5)
    D = Data(ts, onset_rel)
    t1, t2 = fz["t1"], {s: fin(fz["t2"][s]) for s in NAMES}
    g = fz["args"]
    qcls = fz.get("final_query_classes", [s for s in NAMES if np.isfinite(t2[s])])
    gate_classes = [s for s in NAMES if s not in g["exclude"]]
    mask = gate_masks(D.B, D.clips, g["lo"], g["policy"], gate_classes)
    sel_q = {(c, k, s) for s in qcls for c in D.clips for k in range(D.n[c])
             if mask[s][c][k] and D.B[s][c][k][0] < t1[s]}
    all_q = {(c, k, s) for c in D.clips for k in range(D.n[c]) for s in NAMES}
    t2a = {s: fin(cp["always_on_two_thresholds_t2"][s]) for s in NAMES}
    two_q = {(c, k, s) for (c, k, s) in all_q if D.B[s][c][k][0] < t1[s] and np.isfinite(t2a[s])}

    subset = lambda c: D.manifest[c].get("target_subset") or "main"  # noqa: E731
    ev = [(s, c) for s in NAMES for c in D.gts[s] if subset(c) == "main"]
    systems = {"SED alone (t1)": D.dets(set(), t1, t2),
               "selective (frozen)": D.dets(sel_q, t1, t2),
               "always-on, single threshold": D.dets(all_q, None, single=cp["always_on_single_threshold"]),
               "always-on, two thresholds": D.dets(two_q, t1, t2a)}
    scored = {k: D.score(v) for k, v in systems.items()}
    base_h = scored["SED alone (t1)"][0]

    def pct(v):
        return 100 * float(np.mean(v)) if len(v) else float("nan")

    def summary(h, fa):
        rec = {c: pct([h[s][x] for s, x in ev if D.manifest[x]["condition"] == c]) for c in CONDITIONS}
        bcl = {s: pct([h[s][x] for s2, x in ev if s2 == s]) for s in NAMES}
        fx = [(s, x) for s, x in ev if h[s][x] and not base_h[s][x]]
        rg = [(s, x) for s, x in ev if base_h[s][x] and not h[s][x]]
        return rec, bcl, fx, rg, fa

    res = {k: summary(*v) for k, v in scored.items()}
    base_rec = res["SED alone (t1)"][0]
    hours = sum(D.n.values()) * 2.0 / 3600
    print(f"Evaluation on {ts.name}{' (HELD-OUT)' if heldout else ''}: {len(D.clips)} clips, {len(ev)} target events "
          f"(main subset), {hours * 60:.0f} min per class; frozen design {a.frozen}")
    print(f"   USS queries: selective {len(sel_q)}, always-on single {len(all_q)}, always-on two-threshold {len(two_q)}")
    print(f"\n   {'system':30}" + "".join(f"{c:>9}" for c in CONDITIONS) + "".join(f"{s:>12}" for s in NAMES)
          + "   fixes/regr   false alarms")
    for k, (rec, bcl, fx, rg, fa) in res.items():
        print(f"   {k:30}" + "".join(f"{rec[c]:>8.1f}%" for c in CONDITIONS) + "".join(f"{bcl[s]:>11.1f}%" for s in NAMES)
              + f"   {len(fx):>3}/{len(rg):<3}      " + ", ".join(f"{s} {fa[s]}" for s in NAMES))
    print("   change vs SED alone (pp):")
    for k, (rec, *_rest) in res.items():
        if k != "SED alone (t1)":
            print(f"      {k:27} " + " / ".join(f"{rec[c] - base_rec[c]:+.1f}" for c in CONDITIONS))

    # ---------- random gating, matched on the number of queries ----------
    rng = np.random.default_rng(a.seed)
    elig = sorted({(c, k, s) for s in qcls for c in D.clips for k in range(D.n[c]) if D.B[s][c][k][0] < t1[s]})
    n_sel = len(sel_q)
    rand_fx, rand_fa = [], []
    for _ in range(a.trials):
        pick = {elig[i] for i in rng.choice(len(elig), size=min(n_sel, len(elig)), replace=False)} if n_sel else set()
        rec, bcl, fx, rg, fa = summary(*D.score(D.dets(pick, t1, t2)))
        rand_fx.append(len(fx))
        rand_fa.append(sum(fa.values()))
    sel_fx = len(res["selective (frozen)"][2])
    p_ge = float(np.mean(np.array(rand_fx) >= sel_fx))
    print(f"\n   random gating ({a.trials} draws of {n_sel} queries among {len(elig)} eligible windows): fixes mean "
          f"{np.mean(rand_fx):.2f} (5-95% {np.percentile(rand_fx, 5):.0f}-{np.percentile(rand_fx, 95):.0f}), false "
          f"alarms mean {np.mean(rand_fa):.2f}; share of draws with at least the selective design's {sel_fx} fixes: "
          f"{p_ge:.2f}")

    # ---------- scheduled recall ----------
    with open(a.t_query_csv) as fh:
        pool = np.array([float(r["latency_ms"]) / 1000 for r in csv.DictReader(fh)])
    queries = {}
    for (c, k, s) in sel_q:
        queries.setdefault((c, k), []).append(s)
    rng_o, rng_t = np.random.default_rng(a.seed + 1), np.random.default_rng(a.seed + 2)
    sched_d, sched_fx, done_n = {c: [] for c in CONDITIONS}, [], []
    for _ in range(a.orders):
        order = list(rng_o.permutation(D.clips))
        done, stt, _, _ = simulate(order, queries, D.n, 0.0, q["t_plan"], q["ttl_s"], q["sed_s"], q["max_wait"],
                                   rng_t, lambda r: float(r.choice(pool)), q["score_delay_s"], 0.0)
        rec, bcl, fx, rg, fa = summary(*D.score(D.dets(done, t1, t2)))
        for c in CONDITIONS:
            sched_d[c].append(rec[c] - base_rec[c])
        sched_fx.append(len(fx))
        done_n.append(stt["completed"])
    print(f"   selective under the queue ({a.orders} orders, Pi-measured times): change "
          + " / ".join(f"{np.mean(sched_d[c]):+.1f}" for c in CONDITIONS) + " pp (mean), fixes mean "
          f"{np.mean(sched_fx):.2f}, jobs done {np.mean(done_n):.1f}/{len(queries)}")

    # ---------- per recording, bootstrap ----------
    fx_sel = res["selective (frozen)"][2]
    by_rec = {}
    for s, c in ev:
        r = D.manifest[c].get("target_source", "")
        by_rec.setdefault((s, r), []).append(c)
    print(f"\n   fixes per target recording (selective; queried classes {qcls}):")
    for (s, r), cs in sorted(by_rec.items()):
        if s in qcls:
            n_fx = sum(1 for s2, c in fx_sel if s2 == s and c in cs)
            base = sum(base_h[s][c] for c in cs)
            print(f"      {s:10} {r:40} events {len(cs):>3}, SED alone hits {base:>3}, fixes {n_fx}")
    sel_h = scored["selective (frozen)"][0]
    recs = sorted(by_rec)
    boots = []
    for _ in range(2000):
        draw = []
        for s in NAMES:
            rs = [r for r in recs if r[0] == s]
            draw += [rs[i] for i in rng.integers(0, len(rs), len(rs))]
        cl = [(s, c) for (s, r) in draw for c in by_rec[(s, r)]]
        boots.append(100 * (np.mean([sel_h[s][c] for s, c in cl]) - np.mean([base_h[s][c] for s, c in cl])))
    pooled = 100 * (np.mean([sel_h[s][c] for s, c in ev]) - np.mean([base_h[s][c] for s, c in ev]))
    print(f"\n   selective vs SED alone, pooled change {pooled:+.2f} pp; 95% bootstrap interval over target recordings "
          f"{np.percentile(boots, 2.5):+.2f} to {np.percentile(boots, 97.5):+.2f} pp")
    print("   Note: false alarms here are counted on target-rich paired mixtures, not a deployment rate; P2 (1 per hour) "
          "is measured on the target-free recordings.")
    out = {"test_set": str(ts), "heldout": heldout, "when": datetime.now().isoformat(timespec="seconds"),
           "rerun_reason": a.rerun_reason, "first_result": str(first_path), "queue_settings": q,
           "onset_rel": onset_rel, "frozen": a.frozen, "frozen_sha256": sha256(a.frozen),
           "comparators": a.comparators,
           "systems": {k: {"recall_pct": rec, "recall_by_class_pct": bcl, "change_pp": {c: rec[c] - base_rec[c]
                       for c in CONDITIONS}, "fixes": [list(x) for x in fx], "regressions": [list(x) for x in rg],
                           "false_alarms": fa} for k, (rec, bcl, fx, rg, fa) in res.items()},
           "queries": {"selective": len(sel_q), "always_on_single": len(all_q), "always_on_two": len(two_q)},
           "random_gating": {"trials": a.trials, "fixes_mean": float(np.mean(rand_fx)), "p_at_least_selective": p_ge},
           "scheduled": {c: float(np.mean(sched_d[c])) for c in CONDITIONS},
           "scheduled_fixes_mean": float(np.mean(sched_fx)), "jobs_done_mean": float(np.mean(done_n)),
           "pooled_change_pp": float(pooled), "bootstrap_95": [float(np.percentile(boots, 2.5)),
                                                               float(np.percentile(boots, 97.5))]}
    clean = lambda o: ({k: clean(v) for k, v in o.items()} if isinstance(o, dict) else  # noqa: E731
                       [clean(v) for v in o] if isinstance(o, list) else
                       (None if isinstance(o, float) and not np.isfinite(o) else o))
    out_path.write_text(json.dumps(clean(out), indent=2))
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dev_set", required=True)
    p.add_argument("--frozen", required=True)
    p.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--test_set", required=True)
    e.add_argument("--frozen", required=True)
    e.add_argument("--comparators", required=True)
    e.add_argument("--t_query_csv", required=True)
    e.add_argument("--orders", type=int, default=200)
    e.add_argument("--trials", type=int, default=500)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--confirm_heldout", action="store_true")
    e.add_argument("--lock", default=f"{TA}/heldout_lock.json")
    e.add_argument("--rerun_reason", default="")
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else evaluate_set(a)


if __name__ == "__main__":
    main()