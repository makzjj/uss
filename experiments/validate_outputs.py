"""Output validation: does USS's output sound more like the class we asked for, or like another target?

Gate 1's regressions came from cross-class confusion: on jackhammer clips the "engine" output
often scored as engine, raising the engine threshold. This tests one fixed check, decided before
looking at any result:

  For a USS output queried for class s, score it for every target class c with Cnn14 and divide
  by that class's SED-only matched threshold (so jackhammer's tiny scores are comparable). If any
  other target class scores higher than s on this scale, REJECT the USS evidence for that window:
  the window keeps its original SED score. Otherwise ACCEPT it (max rule as before).

No parameter is tuned on the outcome, so the 7 dev fixes cannot be memorised; the per-class scale
does come from the dev-set SED-only thresholds. Main assumption: those thresholds were set on the
ORIGINAL audio, but the rule is applied to USS outputs, which score higher across the board on
clips without the target. Dividing by the always-on thresholds would be the obvious alternative;
it is not tried here, because choosing between the two after seeing results would be tuning.
Expected side effect: jackhammer's threshold is tiny (about 0.03 vs 0.10-0.18), so its scaled
score is inflated on every output; the rule may therefore reject ENGINE outputs on real engine
windows. Read the real-target rejection column for engine first.

Validation changes accuracy, not compute: USS still has to run before its output can be checked.

Part 1 (GPU, minutes): re-score every saved Gate 1 USS output (uss_outputs/<clip>/query=<sound>)
for all three target classes, using detect_baseline.cnn14_clip_and_frames exactly as run_gate1.py
does. Check: the queried class's score must match Gate 1's saved 'after' score. Cached in
results/<set>_cross_scores.csv (delete it, or --rescore, to redo).

Part 2: configurations, each re-matched at the SED-only false-alarm count (the Gate 1 method for
comparing pipelines): SED only; always-on USS; class-aware (never beeper); the Week 6 lead gate;
each without and with validation. Reports delta recall vs SED only, event fixes / regressions,
which always-on fixes survive and regressions disappear, and how often validation rejects USS on
the windows that actually contain the queried target (the main cost of the check).

In-sample on the dev set, and few events: a lead to confirm on held-out recordings, not a result.
The lead-gate rows are in-sample twice: the lead setting was itself chosen on these clips.

Usage (inside the container, from the uss folder):
  python experiments/validate_outputs.py
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gate_analysis import gated, load, overlapping  # noqa: E402
from score_events import CONDITIONS, evaluate, fuse, match  # noqa: E402
from detect_baseline import USS_THRESHOLD  # noqa: E402
from run_gate1 import SOUNDS, TA, uss_output  # noqa: E402

SR, SEG = 32000, 64000
NAMES = list(SOUNDS)


def rescore(ts, manifest, path):
    """Cnn14 score of every saved USS output piece for all target classes (one pass per piece)."""
    import librosa
    import torch
    from detect_baseline import check_frame_readout, cnn14_clip_and_frames
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from uss.config import panns_paths_dict
    from uss.utils import get_path, load_pretrained_panns

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained_panns(model_type="Cnn14", checkpoint_path=get_path(panns_paths_dict["Cnn14"]),
                                  freeze=True).to(device).eval()
    check_frame_readout(model, SEG, device)
    ids = {c: SOUNDS[c][0] for c in NAMES}
    rows = []
    for i, (name, r) in enumerate(sorted(manifest.items()), 1):
        for q, (_, qdir) in SOUNDS.items():
            f = uss_output(ts, r["clip"], qdir)
            if not f.exists():
                raise SystemExit(f"Missing saved USS output {f}; Gate 1 outputs are needed")
            audio, _ = librosa.load(path=str(f), sr=SR, mono=True)
            for k, p in enumerate(range(0, len(audio), SEG)):
                piece = librosa.util.fix_length(data=audio[p: p + SEG], size=SEG, axis=0)
                with torch.no_grad():
                    clip, _ = cnn14_clip_and_frames(model, torch.Tensor(piece).unsqueeze(0).to(device))
                for c, cid in ids.items():
                    rows.append({"clip": name, "query": q, "piece": k, "class": c,
                                 "score": round(clip[0, cid].item(), 5)})
        if i % 20 == 0:
            print(f"   {i}/{len(manifest)} clips re-scored")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--pieces", default="", help="default: results/<set>_scores_pieces.csv (FP32 Gate 1)")
    ap.add_argument("--cross", default="", help="default: results/<set>_cross_scores.csv")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--lead", nargs=3, metavar=("POLICY", "LO", "EXCLUDE"), default=["kmax", "0.1002", "beeper"])
    ap.add_argument("--onset_rel", type=float, default=0.5)
    a = ap.parse_args()

    ts = Path(a.test_set)
    pieces = Path(a.pieces or f"results/{ts.name}_scores_pieces.csv")
    cross_path = Path(a.cross or f"results/{ts.name}_cross_scores.csv")
    manifest, seq, gts = load(ts, pieces, a.onset_rel)
    clips = sorted(manifest)

    # ---------- part 1 ----------
    if a.rescore or not cross_path.exists():
        print(f"Part 1: re-scoring saved USS outputs for all {len(NAMES)} target classes ...")
        rescore(ts, manifest, cross_path)
    cross = defaultdict(dict)                                   # cross[(clip, query, piece)][class]
    with open(cross_path) as fh:
        for r in csv.DictReader(fh):
            cross[(r["clip"], r["query"], int(r["piece"]))][r["class"]] = float(r["score"])
    B = {s: seq[(s, "before")] for s in NAMES}
    U = {s: seq[(s, "after")] for s in NAMES}
    missing = [(c, s, k) for s in NAMES for c in clips for k in range(len(U[s][c]))
               if set(cross.get((c, s, k), {})) != set(NAMES)]
    if missing:
        sys.exit(f"{cross_path} lacks {len(missing)} (clip, query, piece) entries, e.g. {missing[0]}; "
                 f"it may be from another test set. Run again with --rescore.")
    worst = max(abs(cross[(c, s, k)][s] - U[s][c][k][0]) for s in NAMES for c in clips for k in range(len(U[s][c])))
    print(f"Check: re-scored queried-class score vs Gate 1 saved score, max difference {worst:.1e}"
          + ("" if worst < 1e-3 else "  <- does not match Gate 1; stopping"))
    if worst >= 1e-3:
        sys.exit(1)

    # ---------- the check ----------
    anchor = {s: sum(evaluate(B[s], gts[s], USS_THRESHOLD)[1].values()) for s in NAMES}
    thr_b = {s: match(B[s], gts[s], anchor[s])[0] for s in NAMES}
    accept = {s: {c: [max(cross[(c, s, k)][o] / thr_b[o] for o in NAMES if o != s)
                      < cross[(c, s, k)][s] / thr_b[s] for k in range(len(U[s][c]))] for c in clips} for s in NAMES}
    print("\nValidation rule: reject the USS output for class s when another target class scores higher on it,")
    print("each divided by its SED-only threshold (" + ", ".join(f"{s} {thr_b[s]:.4f}" for s in NAMES) + ").")
    print("Assumption: thresholds set on the original audio are a fair scale for USS outputs (see docstring).")

    def in_event(s, c, k):
        """Window overlaps class s's event in clip c (same rule as gate_analysis.overlapping)."""
        return c in gts[s] and overlapping(k, gts[s][c])

    print(f"\n   {'query':11}{'windows on the queried target':>32}{'rejected':>10}"
          f"{'windows on another target':>28}{'rejected':>10}{'target-free':>13}{'rejected':>10}")
    for s in NAMES:
        tgt = [accept[s][c][k] for c in clips for k in range(len(U[s][c])) if in_event(s, c, k)]
        oth = [accept[s][c][k] for c in clips for k in range(len(U[s][c]))
               if manifest[c]["contains"] not in (s, "none") and in_event(manifest[c]["contains"], c, k)]
        free = [accept[s][c][k] for c in clips for k in range(len(U[s][c])) if manifest[c]["contains"] == "none"]
        say = lambda v: f"{100 * (1 - np.mean(v)):.0f}%" if v else "-"  # noqa: E731
        print(f"   {s:11}{len(tgt):>32}{say(tgt):>10}{len(oth):>28}{say(oth):>10}{len(free):>13}{say(free):>10}")
    print("   (rejecting real-target windows is the cost; rejecting other-target windows is the point)")

    # ---------- part 2: configurations ----------
    def gate(lo, policy, ex):
        g = [s for s in NAMES if s not in ex]
        fired = {s: {c: [s in g and b[0] >= lo for b in B[s][c]] for c in clips} for s in NAMES}
        if policy == "k1":
            return fired
        hop = {c: [any(fired[s][c][k] for s in NAMES) for k in range(len(B[NAMES[0]][c]))] for c in clips}
        return {s: hop if s in g else {c: [False] * len(v) for c, v in hop.items()} for s in NAMES}

    def both(m1, m2):
        return {s: {c: [x and y for x, y in zip(m1[s][c], m2[s][c])] for c in clips} for s in NAMES}

    ex = [] if a.lead[2] == "none" else a.lead[2].split(",")
    never_beeper = gate(0.0, "k1", ["beeper"])
    lead = gate(float(a.lead[1]), a.lead[0], ex)
    allon = gate(0.0, "k1", [])
    configs = [("SED only", None), ("always-on USS", allon), ("always-on + validation", both(allon, accept)),
               ("class-aware (no beeper)", never_beeper), ("class-aware + validation", both(never_beeper, accept)),
               (f"lead gate (lo {float(a.lead[1]):g})", lead), ("lead gate + validation", both(lead, accept))]

    base_hits = {s: match(B[s], gts[s], anchor[s])[1] for s in NAMES}
    allon_hits = {s: match(fuse(B[s], U[s], "max"), gts[s], anchor[s])[1] for s in NAMES}
    fixes0 = {(s, c) for s in NAMES for c in gts[s] if allon_hits[s][c] and not base_hits[s][c]}
    regs0 = {(s, c) for s in NAMES for c in gts[s] if base_hits[s][c] and not allon_hits[s][c]}

    print("\n   d = delta recall vs SED only (pp), re-matched at the same false alarms; f/r = fixes/regressions")
    print(f"   {'configuration':28}" + "".join(f"{'d ' + c:>11}" for c in CONDITIONS) + f"{'f/r':>8}"
          + f"{'always-on fixes kept':>22}{'always-on regressions left':>28}")
    for title, mask in configs:
        hits = {}
        for s in NAMES:
            d = B[s] if mask is None else gated(B[s], U[s], mask[s])
            hits[s] = match(d, gts[s], anchor[s])[1]
        ev = [(s, c) for s in NAMES for c in gts[s]]
        d_cond = {}
        for cond in CONDITIONS:
            sel = [(s, c) for s, c in ev if manifest[c]["condition"] == cond]
            d_cond[cond] = 100 * (np.mean([hits[s][c] for s, c in sel]) - np.mean([base_hits[s][c] for s, c in sel]))
        f = sum(hits[s][c] and not base_hits[s][c] for s, c in ev)
        r = sum(base_hits[s][c] and not hits[s][c] for s, c in ev)
        kept = sum(hits[s][c] for s, c in fixes0)
        left = sum(not hits[s][c] for s, c in regs0)
        print(f"   {title:28}" + "".join(f"{d_cond[c]:>+11.1f}" for c in CONDITIONS) + f"{f:>5}/{r:<3}"
              + f"{kept:>15}/{len(fixes0):<6}{left:>21}/{len(regs0):<6}")

    print("\nLimits: in-sample on the dev set (thresholds and the comparison use the same clips); "
          f"{len(fixes0)} fixes and {len(regs0)} regressions in total,")
    print("so a change of one event is several percent. No parameter was tuned on the outcome, but the")
    print("per-class scale is the dev-set SED-only thresholds. The lead-gate rows are in-sample twice (the")
    print("lead setting was chosen on these clips). Confirming any of this needs held-out target recordings.")


if __name__ == "__main__":
    main()