"""Event-level Gate 1 scorer for gate1_v2 (Requirements v0.3.5 definitions).

Reads
  <test_set>/manifest.csv, settings.json   ground truth: target onset/offset, condition, recording
  results/<set>_scores_pieces.csv          listener score per 2 s piece + frame readout (run_gate1.py)
  <test_set>/uss_outputs/, events/         USS outputs and Scaper's isolated events (separation)

Detection and event time
  A 2 s piece is a detection when its score >= threshold. Its event time is the piece start
  plus the onset frame: the first 0.32 s frame whose readout reaches --onset_rel x the piece's
  highest frame. The frame readout comes from the same Cnn14 (detect_baseline.cnn14_clip_and_frames);
  using the piece start alone could land up to 2 s before the real onset.
Fusion rules (how the USS system turns the original and USS output into one score per piece)
  replace  the USS output's score only
  max      the higher of the original's and the USS output's score (USS can only ADD detections,
           as in recovery mode); the event time comes from whichever of the two gave that score
  Both are scored for the native and the level-matched USS output. The rule is chosen on this
  development set (higher mean delta recall over conditions and output levels) and then frozen:
  pass --fusion <rule> for every later run, including the held-out set.
Matching (per sound)
  hit          a detection with event time in [GT onset - 0.5 s, GT offset]; repeats count once
  false alarm  a detection with no same-sound match (a wrong-sound detection is one too);
               false alarms within 10 s are merged
Matched false alarms
  Anchor = the 'before' pipeline's false alarms at the 0.2 cut-off (USS's own value). For each sound
  and pipeline the threshold is the LOWEST one whose false alarms do not exceed the anchor, so
  recall is compared at (at most) the same false alarms per hour.
Gate 1 verdict (each USS pipeline vs before; ALL SOUNDS POOLED, per-sound results shown alongside)
  delta recall >= +5 pp in at least 2 of the 3 conditions (ebr0, ebr-5, overlap), and event
  fixes > event regressions. False-alarm fixes/regressions are reported too.
  95% CI by bootstrap over recordings (target_source), resampled within each sound.
Separation (native USS output only: SI-SDR is scale-invariant, and leakage must be at native level)
  SI-SDR / SI-SDRi against events/<clip>_events/foreground*_<sound>.wav; for clips without that
  sound, output energy relative to the clip (leakage, dB).
Diagnostics
  What sets each threshold  the highest pieces that can never be a hit, one per clip, with their
                            clip type. Clips are no longer than the 10 s merge window, so each clip
                            holds at most one false alarm and the threshold sits just above the
                            (allowed false alarms + 1)-th of these.
  Regression causes         for each event regression: 'time moved' (still detected, but the event
                            time taken from the USS output left the collar: a scoring artefact),
                            'threshold rose' (a hit at the 'before' threshold), or 'score fell'.
  False-alarm sweep         delta recall with every pipeline re-matched at 0, 1, 2, 5, 10 false alarms.
  Perfect separation        if results/<set>_oracle_pieces.csv exists (run_oracle.py), the listener
                            on the isolated targets (silence elsewhere) is scored as an upper bound.
                            It is never a candidate for the fusion choice or the verdict.

Writes to results/<set>_events/: thresholds.csv, threshold_setters.csv, gt_events.csv (with
regression causes), false_alarms.csv, fa_sweep.csv, separation.csv, report.txt

Usage (inside the container, from the uss folder, after run_gate1.py):
  python experiments/score_events.py                  # development set: chooses and freezes the fusion rule
  python experiments/score_events.py --fusion max     # any later run: use the frozen rule
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from detect_baseline import FRAME_S, USS_THRESHOLD  # noqa: E402
from run_gate1 import SOUNDS, TA, uss_output  # noqa: E402
from score_separation import si_sdr  # noqa: E402

VERSIONS = ["before", "after", "after_lm"]
FUSIONS = ["replace", "max"]
PIPELINES = {  # name: (USS output version, fusion rule, label)
    "replace": ("after", "replace", "native, replace"),
    "max": ("after", "max", "native, max"),
    "replace_lm": ("after_lm", "replace", "level-matched, replace"),
    "max_lm": ("after_lm", "max", "level-matched, max"),
}
ORACLE = {  # perfect separation (run_oracle.py): an upper bound, never a candidate
    "oracle": ("oracle", "replace", "perfect separation, replace"),
    "oracle_max": ("oracle", "max", "perfect separation, max"),
}
CONDITIONS = ["ebr0", "ebr-5", "overlap"]
COLLAR_S = 0.5
FA_MERGE_S = 10.0
MIN_GAIN_PP = 5.0
MIN_CONDITIONS = 2
P2_FA_PER_HOUR = 1.0          # Requirements v0.3.5 P2: <= 1 false alarm per hour
FA_BUDGETS = [0, 1, 2, 5, 10]
N_SETTERS = 3
CAUSES = ["time moved", "threshold rose", "threshold rose + time moved", "score fell"]

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def onset_time(start_s, frames, onset_rel):
    """Piece start + first frame reaching onset_rel x the piece's highest frame."""
    f = np.array([float(v) for v in frames.split()])
    idx = int(np.argmax(f >= onset_rel * f.max())) if f.max() > 0 else 0
    return start_s + idx * FRAME_S


def fuse(before, uss, rule):
    """Per piece: (score, time) of the fused system. Pieces are matched by index."""
    out = {}
    for clip, b in before.items():
        u = uss[clip]
        if len(u) != len(b):
            raise SystemExit(f"{clip}: {len(b)} pieces before USS but {len(u)} after")
        out[clip] = u if rule == "replace" else [x if x[0] >= y[0] else y for x, y in zip(b, u)]
    return out


def in_window(t, win):
    return win[0] - COLLAR_S <= t <= win[1]


def evaluate(dets, gt, thr):
    """dets: {clip: [(score, time)]} for one sound and pipeline; gt: {clip: (onset, offset)}.
    Returns ({gt clip: hit}, {clip: false alarms})."""
    hits, fas = {}, {}
    for clip, ds in dets.items():
        times = sorted(t for s, t in ds if s >= thr)
        if clip in gt:
            inside = [in_window(t, gt[clip]) for t in times]
            hits[clip] = any(inside)
            times = [t for t, m in zip(times, inside) if not m]
        fa, last = 0, None
        for t in times:                          # greedy merge = fewest 10 s windows covering them
            if last is None or t - last > FA_MERGE_S:
                fa += 1
                last = t
        fas[clip] = fa
    return hits, fas


def lowest_threshold(dets, gt, max_fa):
    """Lowest threshold with false alarms <= max_fa (false alarms only fall as the threshold rises)."""
    cands = sorted({s for ds in dets.values() for s, _ in ds} | {1.01})
    lo, hi = 0, len(cands) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(evaluate(dets, gt, cands[mid])[1].values()) <= max_fa:
            hi = mid
        else:
            lo = mid + 1
    return cands[lo]


def match(dets, gt, max_fa):
    """(threshold, hits, false alarms) at the lowest threshold with false alarms <= max_fa."""
    thr = lowest_threshold(dets, gt, max_fa)
    return (thr, *evaluate(dets, gt, thr))


def unmatched_peaks(dets, gt):
    """Per clip: (score, piece) of the highest piece that can never be a hit (event time outside
    the GT window, or a clip without this sound). These are the false alarm candidates."""
    out = {}
    for clip, ds in dets.items():
        cand = [(s, k) for k, (s, t) in enumerate(ds) if clip not in gt or not in_window(t, gt[clip])]
        if cand:
            out[clip] = max(cand)
    return out


def regression_cause(before, fused, win, thr_b, thr_p):
    """Why an event hit 'before' is missed by a pipeline, from the pieces that gave the 'before' hit."""
    found = set()
    for (sb, tb), (sp, tp) in zip(before, fused):
        if sb < thr_b or not in_window(tb, win):
            continue
        if sp >= thr_p:
            found.add("time moved")              # detected, but not inside the collar any more
        elif sp >= thr_b:
            found.add("threshold rose" if in_window(tp, win) else "threshold rose + time moved")
        else:
            found.add("score fell")              # replace only: max never lowers a score
    return next(c for c in CAUSES if c in found)


def delta_table(events, pipelines):
    """Per condition: recall before/after and event fixes/regressions for each pipeline."""
    out = {}
    for c in CONDITIONS:
        ev = [e for e in events if e["condition"] == c]
        if not ev:
            continue
        row = {"n": len(ev), "recall_before": np.mean([e["hit_before"] for e in ev])}
        for p in pipelines:
            row[f"recall_{p}"] = np.mean([e[f"hit_{p}"] for e in ev])
            row[f"delta_{p}"] = 100 * (row[f"recall_{p}"] - row["recall_before"])
            row[f"fixes_{p}"] = sum(e[f"hit_{p}"] and not e["hit_before"] for e in ev)
            row[f"regressions_{p}"] = sum(e["hit_before"] and not e[f"hit_{p}"] for e in ev)
        out[c] = row
    return out


def verdict(table, p):
    n_pass = sum(r[f"delta_{p}"] >= MIN_GAIN_PP for r in table.values())
    fixes = sum(r[f"fixes_{p}"] for r in table.values())
    regs = sum(r[f"regressions_{p}"] for r in table.values())
    return n_pass >= MIN_CONDITIONS and fixes > regs, n_pass, fixes, regs


def load_mono(path):
    y, sr = sf.read(str(path), dtype="float64")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--pieces", default="", help="default: results/<set>_scores_pieces.csv")
    ap.add_argument("--oracle_pieces", default="",
                    help="default: results/<set>_oracle_pieces.csv (run_oracle.py); skipped if missing")
    ap.add_argument("--out_dir", default="", help="default: results/<set>_events")
    ap.add_argument("--fusion", choices=["auto"] + FUSIONS, default="auto",
                    help="auto = choose on this (development) set; afterwards always pass the frozen rule")
    ap.add_argument("--anchor_threshold", type=float, default=USS_THRESHOLD,
                    help="'before' cut-off whose false alarms all pipelines are matched to")
    ap.add_argument("--onset_rel", type=float, default=0.5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_separation", action="store_true")
    a = ap.parse_args()

    ts = Path(a.test_set)
    pieces_path = Path(a.pieces or f"results/{ts.name}_scores_pieces.csv")
    out_dir = Path(a.out_dir or f"results/{ts.name}_events")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ts / "manifest.csv") as fh:
        manifest = {Path(r["clip"]).name: r for r in csv.DictReader(fh)}
    clip_s = json.loads((ts / "settings.json").read_text())["duration_s"]
    hours = len(manifest) * clip_s / 3600     # every clip is scored for every sound

    oracle_path = Path(a.oracle_pieces or f"results/{ts.name}_oracle_pieces.csv")
    pipes = dict(PIPELINES)
    versions, sources = list(VERSIONS), [pieces_path]
    if oracle_path.exists():
        pipes.update(ORACLE)
        versions.append("oracle")
        sources.append(oracle_path)

    # raw[(sound, version)][clip] = [(score, event time)] in piece order
    raw = defaultdict(lambda: defaultdict(dict))
    starts = defaultdict(dict)                # starts[clip][piece] = piece start (s)
    for path in sources:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                t = onset_time(float(r["start_s"]), r["frames"], a.onset_rel)
                raw[(r["sound"], r["version"])][r["clip"]][int(r["piece"])] = (float(r["score"]), t)
                starts[r["clip"]][int(r["piece"])] = float(r["start_s"])
    for s in SOUNDS:
        for v in versions:
            missing = set(manifest) - set(raw[(s, v)])
            if missing:
                raise SystemExit(f"No '{v}' pieces for {s} on {len(missing)} clips (e.g. {sorted(missing)[0]}); "
                                 f"rerun {'run_oracle.py' if v == 'oracle' else 'run_gate1.py'}")
    piece_s = min(st for ps in starts.values() for st in ps.values() if st > 0)
    seq = {key: {c: [ps[k] for k in sorted(ps)] for c, ps in by_clip.items()} for key, by_clip in raw.items()}
    dets = {}
    for s in SOUNDS:
        dets[(s, "before")] = seq[(s, "before")]
        for p, (version, rule, _) in pipes.items():
            dets[(s, p)] = fuse(seq[(s, "before")], seq[(s, version)], rule)
    gts = {s: {c: (float(r["target_onset"]), float(r["target_offset"]))
               for c, r in manifest.items() if r["contains"] == s} for s in SOUNDS}
    names = ["before"] + list(pipes)

    def label(p):
        return pipes[p][2] if p in pipes else p

    def neg_group(s, clip):
        r = manifest[clip]
        if r["contains"] == s:
            return "same sound, outside event"
        return r["condition"] if r["contains"] == "none" else "other target"

    def source(s, p, clip, k):
        """Which audio gave a pipeline's score for this piece."""
        if p == "before":
            return "original"
        version, rule, _ = pipes[p]
        name = "isolated target" if version == "oracle" else "USS output"
        if rule == "replace":
            return name
        return "original" if seq[(s, "before")][clip][k][0] >= seq[(s, version)][clip][k][0] else name

    # ---------- thresholds at matched false alarms ----------
    say(f"Event-level Gate 1 scorer: {ts.name}, {len(manifest)} clips of {clip_s:g} s "
        f"({hours * 60:.1f} min scored per sound)")
    say(f"Hit = event time in [onset - {COLLAR_S} s, offset]; false alarms merged within {FA_MERGE_S:g} s; "
        f"onset frame at {a.onset_rel} x piece max")
    say(f"\nThresholds matched to the 'before' false alarms at cut-off {a.anchor_threshold}:")
    say(f"{'sound':12}{'pipeline':28}{'threshold':>10}{'false alarms':>14}{'per hour':>10}")
    events, thr_rows, fa_rows, setter_rows = [], [], [], []
    anchor_fa, res = {}, {}
    for s in SOUNDS:
        gt = gts[s]
        anchor_fa[s] = sum(evaluate(dets[(s, "before")], gt, a.anchor_threshold)[1].values())
        res[s] = {p: match(dets[(s, p)], gt, anchor_fa[s]) for p in names}
        for p in names:
            thr, _, fas = res[s][p]
            fa = sum(fas.values())
            thr_rows.append({"sound": s, "pipeline": p, "threshold": round(thr, 5), "false_alarms": fa,
                             "fa_per_hour": round(fa / hours, 1), "anchor_false_alarms": anchor_fa[s]})
            say(f"{s:12}{label(p):28}{thr:>10.4f}{fa:>14}{fa / hours:>10.1f}")
        thr_b, hit_b = res[s]["before"][:2]
        for c in gt:
            r = manifest[c]
            causes = {f"cause_{p}": regression_cause(dets[(s, "before")][c], dets[(s, p)][c], gt[c],
                                                     thr_b, res[s][p][0])
                      if hit_b[c] and not res[s][p][1][c] else "" for p in pipes}
            events.append({"sound": s, "clip": c, "condition": r["condition"],
                           "target_source": r["target_source"],
                           **{f"hit_{p}": res[s][p][1][c] for p in names}, **causes})
        for c in manifest:
            fa_rows.append({"sound": s, "clip": c, "group": neg_group(s, c),
                            **{f"fa_{p}": res[s][p][2][c] for p in names}})

    # ---------- what sets each threshold ----------
    say(f"\nWhat sets each threshold: the highest pieces that can never be a hit, one per clip "
        f"(each clip holds at most one false alarm, so the threshold sits just above entry 'allowed + 1'):")
    for s in SOUNDS:
        gt = gts[s]
        for p in names:
            peaks = sorted(((sc, k, c) for c, (sc, k) in unmatched_peaks(dets[(s, p)], gt).items()), reverse=True)
            say(f"  {s} / {label(p)}: threshold {res[s][p][0]:.4f}, {anchor_fa[s]} false alarms allowed")
            for i, (sc, k, c) in enumerate(peaks[:anchor_fa[s] + N_SETTERS]):
                st = starts[c][k]
                drift = c in gt and st < gt[c][1] and st + piece_s > gt[c][0]
                note = "; piece overlaps the event, onset placed outside the collar" if drift else ""
                mark = "  <- threshold just above this" if i == anchor_fa[s] else ""
                say(f"    {sc:.4f}  {c}  ({neg_group(s, c)}, piece {k}, from {source(s, p, c, k)}{note}){mark}")
                setter_rows.append({"sound": s, "pipeline": p, "rank": i + 1, "score": sc, "clip": c,
                                    "group": neg_group(s, c), "piece": k, "source": source(s, p, c, k),
                                    "piece_overlaps_event": drift, "sets_threshold": i == anchor_fa[s]})

    # ---------- fusion rule: choose on this set, or use the frozen one ----------
    table = delta_table(events, pipes)
    mean_delta = {f: np.mean([r[f"delta_{p}"] for r in table.values()
                              for p, (_, rule, _) in PIPELINES.items() if rule == f]) for f in FUSIONS}
    if a.fusion == "auto":
        fusion = max(FUSIONS, key=lambda f: mean_delta[f])   # tie -> replace (listed first)
        how = "chosen on this development set (higher mean delta recall)"
    else:
        fusion = a.fusion
        how = "frozen (--fusion)"
    say(f"\nFusion rule: {fusion}, {how}. Mean delta recall: "
        + ", ".join(f"{f} {mean_delta[f]:+.1f} pp" for f in FUSIONS))
    if a.fusion == "auto":
        say(f"  Freeze it: pass --fusion {fusion} for every later run, including the held-out set.")
    frozen = [p for p, (_, rule, _) in PIPELINES.items() if rule == fusion]

    def tag(p):
        if p in frozen:
            return "   <- frozen rule"
        return "   (upper bound, not a candidate)" if p in ORACLE else ""

    # ---------- delta recall per condition (pooled) ----------
    say("\nRecall per condition (ALL SOUNDS POOLED, each sound at its own matched threshold); "
        "d = delta recall in pp, f/r = event fixes/regressions:")
    if not oracle_path.exists():
        say(f"  (no {oracle_path}: run run_oracle.py to add the perfect-separation upper bound)")
    for p in pipes:
        say(f"\n  {label(p)}{tag(p)}")
        say(f"    {'condition':10}{'n':>4}{'before':>8}{'after':>8}{'d':>8}{'f/r':>8}")
        for c, r in table.items():
            say(f"    {c:10}{r['n']:>4}{r['recall_before']:>8.2f}{r[f'recall_{p}']:>8.2f}"
                f"{r[f'delta_{p}']:>+8.1f}{r[f'fixes_{p}']:>5}/{r[f'regressions_{p}']:<3}")

    say("\nPer sound (delta recall in pp, event fixes/regressions); a sound that drops "
        "can be hidden by the pooled verdict:")
    per_sound = {s: delta_table([e for e in events if e["sound"] == s], pipes) for s in SOUNDS}
    for p in pipes:
        say(f"  {label(p)}{tag(p)}")
        for s, t in per_sound.items():
            fx = sum(r[f"fixes_{p}"] for r in t.values())
            rg = sum(r[f"regressions_{p}"] for r in t.values())
            drop = "  <- drops" if any(r[f"delta_{p}"] < 0 for r in t.values()) else ""
            say(f"    {s:11} " + "  ".join(f"{c} {r[f'delta_{p}']:+5.0f}" for c, r in t.items())
                + f"   fixes {fx}, regressions {rg}{drop}")

    # ---------- causes of event regressions ----------
    say("\nCauses of event regressions (a 'before' hit that the pipeline misses):")
    say("  time moved = still detected, but the event time taken from the USS output left the collar "
        "(a scoring artefact, not a USS failure)")
    say("  threshold rose = would be a hit at the 'before' threshold;  score fell = below even that threshold")
    for p in pipes:
        say(f"  {label(p)}{tag(p)}")
        for s in SOUNDS:
            cs = [e[f"cause_{p}"] for e in events if e["sound"] == s and e[f"cause_{p}"]]
            say(f"    {s:11} {len(cs):>3}" + (": " + ", ".join(f"{c} {cs.count(c)}" for c in CAUSES if c in cs)
                                              if cs else ""))

    # ---------- false-alarm fixes / regressions ----------
    say("\nFalse alarms, frozen rule (fix = a false alarm removed, regression = one added; per clip and sound):")
    groups = sorted({x["group"] for x in fa_rows})
    for p in frozen:
        say(f"  {PIPELINES[p][2]}")
        for g in groups + ["all"]:
            rows = [x for x in fa_rows if g in ("all", x["group"])]
            fx = sum(max(0, x["fa_before"] - x[f"fa_{p}"]) for x in rows)
            rg = sum(max(0, x[f"fa_{p}"] - x["fa_before"]) for x in rows)
            say(f"    {g:27} before {sum(x['fa_before'] for x in rows):>4}  after {sum(x[f'fa_{p}'] for x in rows):>4}"
                f"   fixes {fx:>3}, regressions {rg:>3}")

    # ---------- bootstrap over recordings ----------
    rng = np.random.default_rng(a.seed)
    recs = {s: sorted({e["target_source"] for e in events if e["sound"] == s}) for s in SOUNDS}
    by_rec = defaultdict(list)
    for e in events:
        by_rec[(e["sound"], e["target_source"])].append(e)
    boot = {p: defaultdict(list) for p in PIPELINES}
    boot_pass = {p: 0 for p in PIPELINES}
    for _ in range(a.bootstrap):
        sample = []
        for s, rs in recs.items():
            for rec in rng.choice(rs, size=len(rs), replace=True):
                sample.extend(by_rec[(s, rec)])
        t = delta_table(sample, PIPELINES)
        for p in PIPELINES:
            for c, r in t.items():
                boot[p][c].append(r[f"delta_{p}"])
            boot_pass[p] += verdict(t, p)[0]

    say(f"\nGate 1 verdict, ALL SOUNDS POOLED (delta recall >= +{MIN_GAIN_PP:g} pp in >= {MIN_CONDITIONS} of "
        f"{len(table)} conditions, and event fixes > event regressions):")
    for p in PIPELINES:
        ok, n_pass, fixes, regs = verdict(table, p)
        say(f"\n  {label(p)}{'   <- frozen rule' if p in frozen else '   (not the frozen rule; for reference)'}: "
            f"{'PASS' if ok else 'FAIL'}  ({n_pass} conditions >= +{MIN_GAIN_PP:g} pp; "
            f"fixes {fixes}, regressions {regs})")
        for c, vals in boot[p].items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            say(f"    {c:8} delta {table[c][f'delta_{p}']:+6.1f} pp   95% CI [{lo:+.1f}, {hi:+.1f}]")
        say(f"    verdict holds in {100 * boot_pass[p] / a.bootstrap:.0f}% of {a.bootstrap} recording resamples")
        drops = [f"{s} {c} {r[f'delta_{p}']:+.0f} pp" for s, t in per_sound.items()
                 for c, r in t.items() if r[f"delta_{p}"] < 0]
        if drops:
            say(f"    note: pooled over sounds; these drop: {', '.join(drops)}")

    # ---------- false-alarm sweep ----------
    p2_budget = int(hours * P2_FA_PER_HOUR)
    say(f"\nFalse-alarm sweep: every pipeline (and 'before') re-matched to at most N false alarms per sound; "
        f"pooled delta recall in pp (point estimates, no CI):")
    say(f"  {'':30}" + "".join(f"{c:>9}" for c in table) + "   verdict")
    sweep_rows = []
    for budget in FA_BUDGETS:
        ev = []
        for s in SOUNDS:
            r = {p: match(dets[(s, p)], gts[s], budget) for p in names}
            ev += [{"condition": manifest[c]["condition"], **{f"hit_{p}": r[p][1][c] for p in names}}
                   for c in gts[s]]
        t = delta_table(ev, pipes)
        p2 = f"   <- P2 (<= {P2_FA_PER_HOUR:g} per hour) on {hours * 60:.0f} min" if budget == p2_budget else ""
        say(f"  <= {budget} false alarms ({budget / hours:.1f} per hour){p2}")
        say(f"    {'before recall':28}" + "".join(f"{r['recall_before']:>9.2f}" for r in t.values()))
        for p in pipes:
            ok = verdict(t, p)[0]
            say(f"    {label(p):28}" + "".join(f"{r[f'delta_{p}']:>+9.1f}" for r in t.values())
                + f"   {'PASS' if ok else 'FAIL'}{'' if p not in ORACLE else ' (upper bound)'}")
            for c, r in t.items():
                sweep_rows.append({"max_false_alarms": budget, "fa_per_hour": round(budget / hours, 1),
                                   "pipeline": p, "condition": c, "recall_before": round(r["recall_before"], 4),
                                   "recall": round(r[f"recall_{p}"], 4), "delta_pp": round(r[f"delta_{p}"], 1),
                                   "fixes": r[f"fixes_{p}"], "regressions": r[f"regressions_{p}"],
                                   "verdict": "PASS" if ok else "FAIL"})

    # ---------- separation ----------
    sep_rows = []
    if not a.skip_separation:
        for s, (_, qdir) in SOUNDS.items():
            for c, r in manifest.items():
                est, sr_e = load_mono(uss_output(ts, r["clip"], qdir))
                mix, sr_m = load_mono(r["clip"])
                if sr_e != sr_m:
                    raise SystemExit(f"Sample rates differ for {c}: {sr_e} vs {sr_m}")
                row = {"sound": s, "clip": c, "condition": r["condition"], "contains": r["contains"]}
                if r["contains"] == s:
                    refs = sorted((ts / "events" / f"{Path(c).stem}_events").glob(f"foreground*_{s}.wav"))
                    if not refs:
                        raise SystemExit(f"No isolated {s} event for {c} in events/")
                    ref = sum(load_mono(f)[0] for f in refs)
                    n = min(len(est), len(mix), len(ref))
                    sdr_out, sdr_mix = si_sdr(est[:n], ref[:n]), si_sdr(mix[:n], ref[:n])
                    row.update(group=r["condition"], si_sdr_db=round(float(sdr_out), 2),
                               si_sdri_db=round(float(sdr_out - sdr_mix), 2))
                else:
                    n = min(len(est), len(mix))
                    leak = 10 * np.log10((np.sum(est[:n] ** 2) + 1e-12) / (np.sum(mix[:n] ** 2) + 1e-12))
                    row.update(group=r["condition"] if r["contains"] == "none" else "other target",
                               leakage_db=round(float(leak), 2))
                sep_rows.append(row)

        say("\nSeparation (native USS output), medians:")
        for g in CONDITIONS:
            v = [x for x in sep_rows if x.get("group") == g and "si_sdr_db" in x]
            if v:
                say(f"  {g:13} n={len(v):<4} SI-SDR {np.median([x['si_sdr_db'] for x in v]):+6.1f} dB   "
                    f"SI-SDRi {np.median([x['si_sdri_db'] for x in v]):+6.1f} dB")
        for g in sorted({x["group"] for x in sep_rows if "leakage_db" in x}):
            v = [x["leakage_db"] for x in sep_rows if x["group"] == g and "leakage_db" in x]
            say(f"  {g:13} n={len(v):<4} leakage {np.median(v):+6.1f} dB vs clip (lower = less leaks through)")

    # ---------- limits ----------
    n_neg = sum(r["contains"] == "none" for r in manifest.values())
    say("\nLimits:")
    say(f"  - Target-absent clips: {n_neg} ({n_neg * clip_s / 60:.1f} min). False alarms per hour use all "
        f"{hours * 60:.1f} min scored per sound, so one false alarm = {1 / hours:.1f} per hour: a coarse scale.")
    say(f"    P2 (<= {P2_FA_PER_HOUR:g} per hour) therefore allows {p2_budget} false alarms here; with 0 allowed, "
        f"each threshold is set by a single clip (see 'What sets each threshold').")
    say("  - Recordings per sound: " + ", ".join(f"{s} {len(rs)}" for s, rs in recs.items())
        + ". Few recordings give wide confidence intervals.")
    say("  - Thresholds and the fusion rule are chosen and evaluated on the same development set (in-sample).")
    say(f"  - Event times come from an uncalibrated frame readout ({FRAME_S:.2f} s frames), used only for onsets.")

    # ---------- save ----------
    def write(name, rows):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(out_dir / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)

    write("thresholds.csv", thr_rows)
    write("threshold_setters.csv", setter_rows)
    write("gt_events.csv", events)
    write("false_alarms.csv", fa_rows)
    write("fa_sweep.csv", sweep_rows)
    write("separation.csv", sep_rows)
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved thresholds.csv, threshold_setters.csv, gt_events.csv, false_alarms.csv, fa_sweep.csv, "
          f"separation.csv, report.txt in {out_dir}")


if __name__ == "__main__":
    main()
