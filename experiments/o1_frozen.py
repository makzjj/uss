"""O1 recall and false-alarm check: INT8 scored at the FP32 thresholds, frozen.

A deployed INT8 separator would run at the operating point chosen for the system, not at a
threshold re-tuned for itself. So for each class and pipeline the threshold is set on the FP32
pieces (lowest threshold with false alarms <= the SED-only count at 0.2, as in Gate 1), then
applied unchanged to the INT8 pieces. Recall AND false alarms can then both move, and both are
checked. A re-matched comparison (INT8 given its own threshold) is printed as a second line.

O1 (Requirements v0.3.6): recall loss <= 2 pp; target-absent false alarms still inside P2.
Recall loss here = FP32 recall - INT8 recall over all low-EBR and overlap events, pooled across
classes (as the Gate 1 verdict is pooled); per condition and per class are shown alongside.
False alarms: INT8 count at the frozen threshold must not exceed the allowed count (the P2
operating point on this set). 36.7 min per class cannot validate 1 FP/h by itself; this only
shows INT8 does not add false alarms at the same operating point.

Pipelines checked:
  always-on USS            (lo = 0, all classes), the O1 comparison proper
  --lead                   the Week 6 lead: floor gate, policy and excluded classes as given
Calibration clips (from the INT8 report) are left out of scoring by default, since INT8 was
calibrated on them; --include_calib keeps them.

Usage (inside the container, from the uss folder):
  python experiments/o1_frozen.py --int8_pieces results/gate1_v2_int8_v2_scores_pieces.csv \\
      --int8_report resunet30_at_soft_int8_v2.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).parent))
from gate_analysis import gated, load  # noqa: E402
from score_events import CONDITIONS, evaluate, match  # noqa: E402
from detect_baseline import USS_THRESHOLD  # noqa: E402
from run_gate1 import SOUNDS, TA  # noqa: E402

O1_RECALL_PP = 2.0
SOUNDS_L = list(SOUNDS)


def masks(B, clips, lo, policy, gate_classes):
    """Floor gate, as in gate_analysis.py: class s fires where its original score >= lo."""
    fired = {s: {c: [s in gate_classes and b[0] >= lo for b in B[s][c]] for c in clips} for s in SOUNDS}
    if policy == "k1":
        return fired
    hop = {c: [any(fired[s][c][k] for s in SOUNDS) for k in range(len(B[SOUNDS_L[0]][c]))] for c in clips}
    never = {c: [False] * len(v) for c, v in hop.items()}
    return {s: hop if s in gate_classes else never for s in SOUNDS}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--fp32_pieces", default="results/gate1_v2_scores_pieces.csv")
    ap.add_argument("--int8_pieces", required=True)
    ap.add_argument("--int8_report", default="", help="INT8 JSON from int8_check.py (lists calibration clips)")
    ap.add_argument("--include_calib", action="store_true")
    ap.add_argument("--lead", nargs=3, metavar=("POLICY", "LO", "EXCLUDE"), default=["kmax", "0.1002", "beeper"],
                    help="lead setting; EXCLUDE is a comma list or 'none'")
    ap.add_argument("--onset_rel", type=float, default=0.5)
    a = ap.parse_args()

    ts = Path(a.test_set)
    manifest, fp, gts = load(ts, Path(a.fp32_pieces), a.onset_rel)
    _, q8, _ = load(ts, Path(a.int8_pieces), a.onset_rel)
    clips = sorted(manifest)

    # The 'before' scores must agree: same clips, same SED; only the USS output differs
    worst = max(abs(x[0] - y[0]) for s in SOUNDS for c in clips
                for x, y in zip(fp[(s, "before")][c], q8[(s, "before")][c]))
    print(f"SED scores on the original audio, FP32 file vs INT8 file: max difference {worst:.1e}"
          + ("" if worst < 1e-3 else "  <- the two files do not describe the same SED; stopping"))
    if worst >= 1e-3:
        sys.exit(1)

    calib = set()
    if a.int8_report and not a.include_calib:
        calib = set(json.loads(Path(a.int8_report).read_text()).get("calibration_clips", []))
    scored = [c for c in clips if c not in calib]
    clip_s = json.loads((ts / "settings.json").read_text())["duration_s"]
    hours = len(scored) * clip_s / 3600
    print(f"Scoring {len(scored)} clips ({len(calib)} calibration clips left out), "
          f"{hours * 60:.1f} min per class; frozen thresholds set on the FP32 pieces of all {len(clips)} clips")

    B = {s: fp[(s, "before")] for s in SOUNDS}
    lead_ex = [] if a.lead[2] == "none" else a.lead[2].split(",")
    settings = [("always-on USS", "k1", 0.0, []),
                (f"lead: {a.lead[0]}, lo {float(a.lead[1]):g}, no {'/'.join(lead_ex) or '-'}",
                 a.lead[0], float(a.lead[1]), lead_ex)]
    verdict = True
    for title, policy, lo, ex in settings:
        gate_classes = [s for s in SOUNDS if s not in ex]
        m = masks(B, clips, lo, policy, gate_classes)
        rows, fa_rows = [], []
        for s in SOUNDS:
            anchor = sum(evaluate(B[s], gts[s], USS_THRESHOLD)[1].values())
            d_fp = gated(B[s], fp[(s, "after")], m[s])
            d_q8 = gated(B[s], q8[(s, "after")], m[s])
            thr = match(d_fp, gts[s], anchor)[0]                  # set on FP32, all clips
            sub = lambda d: {c: d[c] for c in scored}            # noqa: E731
            g = {c: v for c, v in gts[s].items() if c in scored}
            h_fp, f_fp = evaluate(sub(d_fp), g, thr)
            h_q8, f_q8 = evaluate(sub(d_q8), g, thr)
            thr_r, h_r, _ = match(sub(d_q8), g, anchor)          # second line: INT8 re-matched
            for c in g:
                rows.append({"s": s, "cond": manifest[c]["condition"], "fp": h_fp[c], "q8": h_q8[c], "rm": h_r[c]})
            fa_rows.append((s, anchor, sum(f_fp.values()), sum(f_q8.values()), thr, thr_r))

        def rec(sel, key):
            return 100 * np.mean([r[key] for r in sel]) if sel else float("nan")

        print(f"\n== {title} ==")
        print(f"   {'':16}{'events':>7}{'FP32 recall':>13}{'INT8 recall':>13}{'loss':>8}{'INT8 re-matched':>17}")
        for cond in CONDITIONS:
            sel = [r for r in rows if r["cond"] == cond]
            print(f"   {cond:16}{len(sel):>7}{rec(sel, 'fp'):>12.1f}%{rec(sel, 'q8'):>12.1f}%"
                  f"{rec(sel, 'fp') - rec(sel, 'q8'):>+7.1f}{rec(sel, 'rm'):>16.1f}%")
        for s in SOUNDS:
            sel = [r for r in rows if r["s"] == s]
            print(f"   {s:16}{len(sel):>7}{rec(sel, 'fp'):>12.1f}%{rec(sel, 'q8'):>12.1f}%"
                  f"{rec(sel, 'fp') - rec(sel, 'q8'):>+7.1f}{rec(sel, 'rm'):>16.1f}%")
        loss = rec(rows, "fp") - rec(rows, "q8")
        changed = sum(r["fp"] != r["q8"] for r in rows)
        print(f"   pooled: recall loss {loss:+.1f} pp over {len(rows)} events "
              f"({changed} events changed: {sum(r['fp'] and not r['q8'] for r in rows)} lost, "
              f"{sum(r['q8'] and not r['fp'] for r in rows)} gained); limit {O1_RECALL_PP:g} pp: "
              f"{'MET' if loss <= O1_RECALL_PP else 'NOT MET'}")
        print("   false alarms at the frozen threshold (allowed = SED-only count at 0.2):")
        fa_ok = True
        for s, allowed, n_fp, n_q8, thr, thr_r in fa_rows:
            ok = n_q8 <= allowed
            fa_ok &= ok
            up = chi2.ppf(0.95, 2 * (n_q8 + 1)) / 2 / hours        # one-sided 95% Poisson bound
            print(f"     {s:11} threshold {thr:.4f}: FP32 {n_fp}, INT8 {n_q8}, allowed {allowed}  "
                  f"({'ok' if ok else 'OVER'}; INT8 rate {n_q8 / hours:.1f}/h, 95% upper bound {up:.1f}/h)"
                  f"   re-matched INT8 threshold {thr_r:.4f}")
        if title.startswith("always-on"):
            verdict &= loss <= O1_RECALL_PP and fa_ok
    print(f"\nO1 recall + false alarms (always-on USS, frozen thresholds): {'MET' if verdict else 'NOT MET'}")
    print(f"The upper bounds show why {hours * 60:.1f} min cannot validate P2 (1 FP/h); the P2 check is the >= 3 h set.")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()