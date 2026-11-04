"""Week 8: cross-class leakage per query pair (failure analysis), from saved Gate 1 data only.

For every USS query (beeper, engine, jackhammer) and every kind of clip it was run on, by what the clip
actually contains (the queried sound itself, each other target, or no target: background only or with an
interferer), over the 2 s windows that matter (windows overlapping the contained target's event; every
window for target-free clips):
  1. Output level: USS output energy relative to the mixture, in dB (median). For the queried sound itself
     this is ordinary separation; for anything else it is leakage (target-absent output).
  2. Detector fooled: the SED score for the QUERIED class on the original audio vs on the USS output
     (medians), and the share of windows at or above that class's SED-only alert threshold, before vs after.
     A rise is the false-alarm risk that raised the always-on thresholds in Gate 1.
  3. Leaked sound recognised: the SED score for the sound ACTUALLY PRESENT, on the USS output (median).
     Low values with audible leakage are the detector blind spot (e.g. jackhammer heard, scored as engine).

Uses: <test_set>/manifest.csv, the saved USS outputs (uss_outputs/<clip>/query=<sound>.wav),
results/<set>_scores_pieces.csv (SED on the original audio) and results/<set>_cross_scores.csv (SED on
every USS output for all target classes; written by validate_outputs.py; checked complete and equal to the
Gate 1 'after' scores before use). Target clips are reported per condition (ebr0, ebr-5, overlap), since
leakage depends on it; a window counts as containing the target only if at least --min_event_s (0.5 s) of
the event lies inside it. Thresholds are the SED-only ones from the FULL set (as validate_outputs.py) and stay
fixed under --drop_source, which leaves out the clips built from one target recording (e.g. jackhammer_t1)
as a sensitivity check. Cells show windows (n) and clips (c): windows of one clip are correlated, so a cell
built from few clips is weak evidence. Dev data, in-sample.

Usage (inside the container, from the uss folder):
  python experiments/leakage_pairs.py
  python experiments/leakage_pairs.py --drop_source jackhammer_t1
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gate_analysis import load  # noqa: E402
from score_events import CONDITIONS, evaluate, match  # noqa: E402
from detect_baseline import USS_THRESHOLD  # noqa: E402
from run_gate1 import SOUNDS, TA, uss_output  # noqa: E402

NAMES = list(SOUNDS)
SEG, SR, PIECE_S = 64000, 32000, 2.0
CORNER = "query / clip contains"


def main():
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--drop_source", default="", help="leave out clips built from this target recording")
    ap.add_argument("--min_event_s", type=float, default=0.5,
                    help="a window counts as containing the target only with at least this much of the event in it")
    ap.add_argument("--onset_rel", type=float, default=0.5)
    a = ap.parse_args()

    ts = Path(a.test_set)
    manifest, seq, gts = load(ts, Path(f"results/{ts.name}_scores_pieces.csv"), a.onset_rel)
    cross = defaultdict(dict)
    cross_path = Path(f"results/{ts.name}_cross_scores.csv")
    if not cross_path.exists():
        sys.exit(f"Missing {cross_path}: run experiments/validate_outputs.py once to create it")
    with open(cross_path) as fh:
        for r in csv.DictReader(fh):
            cross[(r["clip"], r["query"], int(r["piece"]))][r["class"]] = float(r["score"])
    paths = {Path(r["clip"]).name: r["clip"] for r in csv.DictReader(open(ts / "manifest.csv"))}
    clips = sorted(c for c in manifest if not (a.drop_source and manifest[c].get("target_source", "").startswith(a.drop_source)))
    dropped = len(manifest) - len(clips)
    if a.drop_source and not dropped:
        sys.exit(f"--drop_source {a.drop_source}: no clip has a matching target_source (e.g. jackhammer_t1)")

    B = {s: seq[(s, "before")] for s in NAMES}
    U = {s: seq[(s, "after")] for s in NAMES}
    missing = [(c, q, k) for q in NAMES for c in manifest for k in range(len(U[q][c]))
               if set(cross.get((c, q, k), {})) != set(NAMES)]
    if missing:
        sys.exit(f"{cross_path} lacks {len(missing)} (clip, query, piece) entries, e.g. {missing[0]}: it may be from "
                 f"another test set; re-run validate_outputs.py --rescore")
    worst = max(abs(cross[(c, q, k)][q] - U[q][c][k][0]) for q in NAMES for c in manifest for k in range(len(U[q][c])))
    if worst >= 1e-3:
        sys.exit(f"{cross_path} does not match the Gate 1 'after' scores (max difference {worst:.1e}); re-score it")
    anchor = {s: sum(evaluate(B[s], gts[s], USS_THRESHOLD)[1].values()) for s in NAMES}
    thr = {s: match(B[s], gts[s], anchor[s])[0] for s in NAMES}

    def group(c):
        """(what the clip contains, condition): targets per condition; target-free clips by kind."""
        r = manifest[c]
        if r["contains"] != "none":
            return r["contains"], r["condition"]
        return ("background only" if r["condition"] == "bgonly" else "background + interferer"), "target-free"

    def read(path):
        x, sr = sf.read(str(path), dtype="float32")
        if sr != SR or x.ndim != 1:
            sys.exit(f"{path}: expected {SR} Hz mono, got {sr} Hz with shape {x.shape}")
        return x

    acc = defaultdict(lambda: defaultdict(list))         # acc[(query, contains, condition)][metric] -> values
    clipset = defaultdict(set)
    for c in clips:
        g, cond = group(c)
        mix = read(paths[c])
        n_win = len(B[NAMES[0]][c])
        if g in NAMES:                                   # windows holding at least min_event_s of the event
            on, off = gts[g][c]
            win_ok = [min(off, (k + 1) * PIECE_S) - max(on, k * PIECE_S) >= a.min_event_s for k in range(n_win)]
        else:
            win_ok = [True] * n_win
        for q, (_, qdir) in SOUNDS.items():
            out = read(uss_output(ts, paths[c], qdir))
            for k in range(n_win):
                if not win_ok[k]:
                    continue
                m, o = mix[k * SEG:(k + 1) * SEG], out[k * SEG:(k + 1) * SEG]
                e_mix = float(np.sum(m.astype(np.float64) ** 2)) + 1e-12
                key = (q, g, cond)
                d = acc[key]
                clipset[key].add(c)
                d["level_db"].append(10 * np.log10((float(np.sum(o.astype(np.float64) ** 2)) + 1e-12) / e_mix))
                before = B[q][c][k][0]
                after = cross[(c, q, k)][q]
                d["before"].append(before)
                d["after"].append(after)
                d["over_before"].append(before >= thr[q])
                d["over_after"].append(after >= thr[q])
                if g in NAMES and g != q:
                    d["present_on_output"].append(cross[(c, q, k)][g])
                    d["present_on_mix"].append(B[g][c][k][0])

    print(f"Cross-class leakage per query pair: {ts.name}, {len(clips)} clips"
          + (f" ({dropped} clips from {a.drop_source} left out)" if dropped else "")
          + f"; windows with at least {a.min_event_s:g} s of the contained event (all windows for target-free clips)")
    print("SED-only alert thresholds (full set, fixed under --drop_source; not S4's frozen gated thresholds): "
          + ", ".join(f"{s} {thr[s]:.4f}" for s in NAMES))

    free_cols = ["background only", "background + interferer"]
    blocks = [(cond, NAMES) for cond in CONDITIONS] + [("target-free", free_cols)]

    def cell(key, fn, fmt):
        d = acc.get(key)
        return fmt(fn(d)) + f" c={len(clipset[key])}" if d and d["level_db"] else "-"

    def table(title, fn, fmt):
        print(f"\n{title}")
        for cond, cols in blocks:
            print(f"   [{cond}]")
            print(f"   {CORNER:22}" + "".join(f"{c:>30}" for c in cols))
            for q in NAMES:
                print(f"   {q:22}" + "".join(f"{cell((q, c, cond), fn, fmt):>30}" for c in cols))

    table("1. USS output level relative to the mixture, dB (median; the queried sound itself = ordinary separation)",
          lambda d: np.median(d["level_db"]), lambda v: f"{v:+.1f}")
    table("2a. SED score for the QUERIED class, original -> USS output (medians)",
          lambda d: (np.median(d["before"]), np.median(d["after"])), lambda v: f"{v[0]:.3f} -> {v[1]:.3f}")
    table("2b. Windows at or above the queried class's SED-only threshold, original -> USS output",
          lambda d: (100 * np.mean(d["over_before"]), 100 * np.mean(d["over_after"]), len(d["over_before"])),
          lambda v: f"{v[0]:.0f}% -> {v[1]:.0f}% n={v[2]}")
    print("\n3. SED score for the sound ACTUALLY PRESENT, on the mixture -> on the other query's USS output (medians)")
    for cond in CONDITIONS:
        print(f"   [{cond}]")
        print(f"   {CORNER:22}" + "".join(f"{c:>30}" for c in NAMES))
        for q in NAMES:
            cells = []
            for c in NAMES:
                key = (q, c, cond)
                d = acc.get(key)
                cells.append(f"{np.median(d['present_on_mix']):.3f} -> {np.median(d['present_on_output']):.3f} "
                             f"c={len(clipset[key])}" if d and d.get("present_on_output") else "-")
            print(f"   {q:22}" + "".join(f"{x:>30}" for x in cells))
    print("\nReading: off-diagonal cells in table 1 near the diagonal's level mean the query does not suppress "
          "other targets; a rise in 2b off the diagonal is false-alarm risk; a low value in table 3 while 2b rises "
          "means the leaked sound is mis-scored as the queried class (detector blind spot). Dev data, in-sample.")

    out = Path(f"results/{ts.name}_leakage_pairs" + (f"_no-{a.drop_source}" if a.drop_source else "") + ".csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query", "clip_contains", "condition", "windows", "clips", "level_db_median",
                    "queried_before_median", "queried_after_median", "over_thr_before_pct", "over_thr_after_pct",
                    "present_on_mix_median", "present_on_output_median"])
        for key in sorted(acc):
            d = acc[key]
            if not d["level_db"]:
                continue
            q, c, cond = key
            w.writerow([q, c, cond, len(d["level_db"]), len(clipset[key]), round(float(np.median(d["level_db"])), 2),
                        round(float(np.median(d["before"])), 4), round(float(np.median(d["after"])), 4),
                        round(100 * float(np.mean(d["over_before"])), 1), round(100 * float(np.mean(d["over_after"])), 1),
                        round(float(np.median(d["present_on_mix"])), 4) if d.get("present_on_mix") else "",
                        round(float(np.median(d["present_on_output"])), 4) if d.get("present_on_output") else ""])
    print(f"Saved {out}")


if __name__ == "__main__":
    main()