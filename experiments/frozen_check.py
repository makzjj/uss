"""Apply the FROZEN design (Requirements v0.3.9) unchanged to a new test set: extra-dev now, held-out in Week 11.

Reads the frozen values from results/gate1_v2_two_threshold.json (written by two_threshold.py on the development
set): t1 per class, t2 per class, the gate (policy, lo, never-gated classes) and the query classes (finite t2).
Nothing is re-chosen on the new set. Needs results/<set>_scores_pieces.csv from run_gate1.py --test_set <set>.

Reports, against SED alone at the frozen t1 (the stage-1 alerts):
  - recall per condition and per class, for SED alone and for the frozen design (every queried job completing)
  - fixes, regressions (0 by construction) and false alarms; ADDED false alarms = design minus SED alone
  - the v0.3.9 criterion: the freeze stands if at least one fix survives and no false alarm is added

Usage (inside the container, from the uss folder):
  python experiments/frozen_check.py --test_set /home/zmak002/docker_share/test_audio/testsets/extradev_v1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gate_analysis import load  # noqa: E402
from schedule_sim import NAMES, gate_masks  # noqa: E402
from score_events import CONDITIONS, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set", required=True)
    ap.add_argument("--frozen", default="results/gate1_v2_two_threshold.json")
    ap.add_argument("--onset_rel", type=float, default=0.5)
    ap.add_argument("--reproduce_dev", action="store_true",
                    help="run on the development set only to confirm this script reproduces two_threshold.py")
    a = ap.parse_args()

    fz = json.loads(Path(a.frozen).read_text())
    t1 = fz["t1"]
    t2 = {s: (np.inf if fz["t2"][s] is None else fz["t2"][s]) for s in NAMES}
    g = fz["args"]
    query_classes = fz.get("final_query_classes", [s for s in NAMES if np.isfinite(t2[s])])
    ts = Path(a.test_set)
    dev_name = Path(fz["args"]["test_set"]).name
    if ts.name == dev_name and not a.reproduce_dev:
        sys.exit(f"{ts.name} is the development set the design was frozen on: give a new set, or --reproduce_dev")
    if a.reproduce_dev and ts.name != dev_name:
        sys.exit("--reproduce_dev is only for the development set")
    allowed, at_t1 = fz["allowed_false_alarms"], fz["false_alarms_at_t1"]
    slack = {s: allowed[s] - at_t1[s] for s in NAMES}
    manifest, seq, gts = load(ts, Path(f"results/{ts.name}_scores_pieces.csv"), a.onset_rel)
    clips = sorted(manifest)
    B = {s: seq[(s, "before")] for s in NAMES}
    U = {s: seq[(s, "after")] for s in NAMES}
    gate_classes = [s for s in NAMES if s not in g["exclude"]]
    mask = gate_masks(B, clips, g["lo"], g["policy"], gate_classes)
    done = {(c, k, s) for s in query_classes for c in clips for k in range(len(B[s][c]))
            if mask[s][c][k] and B[s][c][k][0] < t1[s]}

    def dets(use_uss):
        out = {}
        for s in NAMES:
            out[s] = {}
            for c in clips:
                row = []
                for k in range(len(B[s][c])):
                    b = B[s][c][k]
                    cand = (b[0] / t1[s], b[1])
                    if use_uss and (c, k, s) in done and np.isfinite(t2[s]) and U[s][c][k][0] / t2[s] > cand[0]:
                        cand = (U[s][c][k][0] / t2[s], U[s][c][k][1])
                    row.append(cand)
                out[s][c] = row
        return out

    res = {}
    for name, use in (("SED alone (t1)", False), ("frozen design", True)):
        d = dets(use)
        hits, fa = {}, {}
        for s in NAMES:
            h, f = evaluate(d[s], gts[s], 1.0)
            hits[s], fa[s] = h, sum(f.values())
        res[name] = (hits, fa)
    base_h, base_fa = res["SED alone (t1)"]
    des_h, des_fa = res["frozen design"]
    subset = lambda c: manifest[c].get("target_subset") or "main"  # noqa: E731
    ev_all = [(s, c) for s in NAMES for c in gts[s]]
    groups = {"main": [(s, c) for s, c in ev_all if subset(c) == "main"],
              "ambiguity": [(s, c) for s, c in ev_all if subset(c) != "main"]}
    added = {s: des_fa[s] - base_fa[s] for s in NAMES}

    def rec(h, ev, cond=None, cls=None):
        sel = [(s, c) for s, c in ev if (cond is None or manifest[c]["condition"] == cond) and (cls is None or s == cls)]
        return 100 * float(np.mean([h[s][c] for s, c in sel])) if sel else None

    fmt = lambda v: f"{v:>9.1f}%" if v is not None else f"{'-':>10}"  # noqa: E731
    hours = sum(len(B[NAMES[0]][c]) for c in clips) * 2.0 / 3600
    print(f"Frozen design on {ts.name}: {len(clips)} clips, {len(groups['main'])} target events in the main subset, "
          f"{len(groups['ambiguity'])} in the ambiguity subset, {hours * 60:.0f} min per class")
    print(f"   frozen values from {a.frozen}: t1 " + ", ".join(f"{s} {t1[s]:.4f}" for s in NAMES) + "; t2 "
          + ", ".join(f"{s} {'none' if not np.isfinite(t2[s]) else f'{t2[s]:.4f}'}" for s in NAMES)
          + f"; gate {g['policy']} lo {g['lo']:g}, never gated {g['exclude']}; queries {query_classes}")
    print("   false-alarm slack allowed per class (from development): " + ", ".join(f"{s} {slack[s]}" for s in NAMES))
    print(f"   USS jobs (every queried window): {len({(c, k) for c, k, _ in done})} windows, {len(done)} queries")
    summary = {}
    for gname, ev in groups.items():
        if not ev:
            continue
        fixes = [(s, c) for s, c in ev if des_h[s][c] and not base_h[s][c]]
        regs = [(s, c) for s, c in ev if base_h[s][c] and not des_h[s][c]]
        print(f"\n   [{gname} subset]{'':4}" + "".join(f"{c:>10}" for c in CONDITIONS) + "".join(f"{s:>11}" for s in NAMES))
        for name, (h, _) in res.items():
            print(f"   {name:20}" + "".join(fmt(rec(h, ev, cond=c)) for c in CONDITIONS)
                  + "".join(fmt(rec(h, ev, cls=s)) for s in NAMES))
        print(f"   fixes {len(fixes)}, regressions {len(regs)} (0 by construction)")
        for s, c in fixes:
            print(f"      fix: {s} in {c} ({manifest[c]['condition']})")
        summary[gname] = {"fixes": [list(x) for x in fixes], "regressions": [list(x) for x in regs],
                          "recall_pct": {name: {"by_condition": {c: rec(h, ev, cond=c) for c in CONDITIONS},
                                                "by_class": {s: rec(h, ev, cls=s) for s in NAMES}}
                                         for name, (h, _) in res.items()}}
    print("\n   false alarms (all clips): SED alone " + ", ".join(f"{s} {base_fa[s]}" for s in NAMES)
          + "; frozen design " + ", ".join(f"{s} {des_fa[s]}" for s in NAMES)
          + "; added " + ", ".join(f"{s} {added[s]:+d}" for s in NAMES))
    n_fix_main = len(summary.get("main", {}).get("fixes", []))
    holds = n_fix_main >= 1 and all(added[s] <= slack[s] for s in NAMES)
    if a.reproduce_dev:
        ref = fz["offline"].get("two_finite_t2_queries")
        same = ref is not None and n_fix_main == ref["fixes"] and len(summary["main"]["regressions"]) == ref["regressions"] \
            and all(des_fa[s] == ref["false_alarms"][s] for s in NAMES)
        print(f"\n   reproduce_dev: two_threshold.py reported fixes {ref and ref['fixes']}, false alarms "
              f"{ref and ref['false_alarms']}; this script: fixes {n_fix_main}, false alarms {des_fa} -> "
              f"{'MATCH' if same else 'MISMATCH: do not trust the extra-dev result until resolved'}")
    else:
        print(f"\n   v0.3.9 criterion (main subset: at least one fix; added false alarms within the development slack): "
              f"{'HOLDS' if holds else 'DOES NOT HOLD'}")
        if not holds:
            print("   The configuration may be adjusted ONCE on development data and re-frozen (recorded in the "
                  "Requirements); the held-out set stays untouched.")
    out = Path(f"results/frozen_check_{ts.name}.json")
    out.write_text(json.dumps({
        "test_set": str(ts), "frozen_from": a.frozen, "t1": t1, "t2": fz["t2"], "query_classes": query_classes,
        "slack": slack, "subsets": summary, "false_alarms": {name: f for name, (_, f) in res.items()},
        "added_false_alarms": added, "criterion_holds": None if a.reproduce_dev else holds,
        "reproduce_dev": a.reproduce_dev}, indent=2, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()