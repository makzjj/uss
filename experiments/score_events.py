"""Event-level Gate 1 scorer for gate1_v2 (Requirements v0.3.5 definitions).

Reads
  <test_set>/manifest.csv, settings.json   ground truth: target onset/offset, condition, recording
  results/<set>_scores_pieces.csv          listener score per 2 s piece + frame readout (run_gate1.py)
  <test_set>/uss_outputs/, events/         USS outputs and Scaper's isolated events (separation)

Detection and event time
  A 2 s piece is a detection when its clip score >= threshold. Its event time is the piece start
  plus the onset frame: the first 0.32 s frame whose readout reaches --onset_rel x the piece's
  highest frame. The frame readout comes from the same Cnn14 (detect_baseline.cnn14_clip_and_frames);
  using the piece start alone could land up to 2 s before the real onset.
Matching (per sound)
  hit          a detection with event time in [GT onset - 0.5 s, GT offset]; repeats count once
  false alarm  a detection with no same-sound match; false alarms within 10 s are merged
Matched false alarms
  Anchor = the 'before' pipeline's false alarms at the 0.2 cut-off (USS's own value). For each sound
  and pipeline (before, after, after_lm) the threshold is the LOWEST one whose false alarms do not
  exceed the anchor, so recall is compared at (at most) the same false alarms per hour.
Gate 1 verdict (each USS pipeline vs before)
  delta recall >= +5 pp in at least 2 of the 3 conditions (ebr0, ebr-5, overlap), and fixes > regressions.
  95% CI by bootstrap over recordings (target_source), resampled within each sound.
Separation (native USS output only: SI-SDR is scale-invariant, and leakage must be at native level)
  SI-SDR / SI-SDRi against events/<clip>_events/foreground*_<sound>.wav; for clips without that
  sound, output energy relative to the clip (leakage, dB).

Writes to results/<set>_events/: thresholds.csv, gt_events.csv, separation.csv, report.txt

Usage (inside the container, from the uss folder, after run_gate1.py):
  python experiments/score_events.py
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
PIPELINES = {"after": "USS, native level", "after_lm": "USS, level-matched"}
CONDITIONS = ["ebr0", "ebr-5", "overlap"]
COLLAR_S = 0.5
FA_MERGE_S = 10.0
MIN_GAIN_PP = 5.0
MIN_CONDITIONS = 2

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def onset_time(start_s, frames, onset_rel):
    """Piece start + first frame reaching onset_rel x the piece's highest frame."""
    f = np.array([float(v) for v in frames.split()])
    idx = int(np.argmax(f >= onset_rel * f.max())) if f.max() > 0 else 0
    return start_s + idx * FRAME_S


def evaluate(dets, gt, thr):
    """dets: {clip: [(score, time)]} for one sound and version; gt: {clip: (onset, offset)}.
    Returns ({gt clip: hit}, false alarm count)."""
    hits, fa = {}, 0
    for clip, ds in dets.items():
        times = sorted(t for s, t in ds if s >= thr)
        if clip in gt:
            on, off = gt[clip]
            inside = [on - COLLAR_S <= t <= off for t in times]
            hits[clip] = any(inside)
            times = [t for t, m in zip(times, inside) if not m]
        last = None
        for t in times:                          # greedy merge = fewest 10 s windows covering them
            if last is None or t - last > FA_MERGE_S:
                fa += 1
                last = t
    return hits, fa


def lowest_threshold(dets, gt, max_fa):
    """Lowest threshold with false alarms <= max_fa (false alarms only fall as the threshold rises)."""
    cands = sorted({s for ds in dets.values() for s, _ in ds} | {1.01})
    lo, hi = 0, len(cands) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if evaluate(dets, gt, cands[mid])[1] <= max_fa:
            hi = mid
        else:
            lo = mid + 1
    return cands[lo]


def delta_table(events):
    """Per condition: recall before/after and fixes/regressions for each pipeline."""
    out = {}
    for c in CONDITIONS:
        ev = [e for e in events if e["condition"] == c]
        if not ev:
            continue
        row = {"n": len(ev), "recall_before": np.mean([e["hit_before"] for e in ev])}
        for p in PIPELINES:
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
    ap.add_argument("--out_dir", default="", help="default: results/<set>_events")
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

    # dets[(sound, version)][clip] = [(score, event time)]
    dets = defaultdict(lambda: defaultdict(list))
    with open(pieces_path) as fh:
        for r in csv.DictReader(fh):
            t = onset_time(float(r["start_s"]), r["frames"], a.onset_rel)
            dets[(r["sound"], r["version"])][r["clip"]].append((float(r["score"]), t))
    for s in SOUNDS:
        for v in VERSIONS:
            missing = set(manifest) - set(dets[(s, v)])
            if missing:
                raise SystemExit(f"{pieces_path} has no '{v}' pieces for {s} on {len(missing)} clips "
                                 f"(e.g. {sorted(missing)[0]}); rerun run_gate1.py")

    # ---------- thresholds at matched false alarms ----------
    say(f"Event-level Gate 1 scorer: {ts.name}, {len(manifest)} clips of {clip_s:g} s "
        f"({hours * 60:.1f} min scored per sound)")
    say(f"Hit = event time in [onset - {COLLAR_S} s, offset]; false alarms merged within {FA_MERGE_S:g} s; "
        f"onset frame at {a.onset_rel} x piece max")
    say(f"\nThresholds matched to the 'before' false alarms at cut-off {a.anchor_threshold}:")
    say(f"{'sound':12}{'pipeline':10}{'threshold':>10}{'false alarms':>14}{'per hour':>10}")
    events, thr_rows = [], []
    for s in SOUNDS:
        gt = {c: (float(r["target_onset"]), float(r["target_offset"]))
              for c, r in manifest.items() if r["contains"] == s}
        _, anchor_fa = evaluate(dets[(s, "before")], gt, a.anchor_threshold)
        hit = {}
        for v in VERSIONS:
            thr = lowest_threshold(dets[(s, v)], gt, anchor_fa)
            hit[v], fa = evaluate(dets[(s, v)], gt, thr)
            thr_rows.append({"sound": s, "pipeline": v, "threshold": round(thr, 5), "false_alarms": fa,
                             "fa_per_hour": round(fa / hours, 1), "anchor_false_alarms": anchor_fa})
            say(f"{s:12}{v:10}{thr:>10.4f}{fa:>14}{fa / hours:>10.1f}")
        for c in gt:
            r = manifest[c]
            events.append({"sound": s, "clip": c, "condition": r["condition"],
                           "target_source": r["target_source"],
                           **{f"hit_{v}": hit[v][c] for v in VERSIONS}})

    # ---------- delta recall per condition ----------
    table = delta_table(events)
    say("\nRecall per condition (all sounds pooled, each sound at its own matched threshold):")
    say(f"{'condition':10}{'n':>4}{'before':>8}" + "".join(f"{p:>10}{'d pp':>7}{'fix/reg':>9}" for p in PIPELINES))
    for c, r in table.items():
        say(f"{c:10}{r['n']:>4}{r['recall_before']:>8.2f}" + "".join(
            f"{r[f'recall_{p}']:>10.2f}{r[f'delta_{p}']:>+7.1f}{r[f'fixes_{p}']:>5}/{r[f'regressions_{p}']:<3}"
            for p in PIPELINES))
    say("\nPer sound (delta recall in pp, native / level-matched):")
    for s in SOUNDS:
        t = delta_table([e for e in events if e["sound"] == s])
        say(f"  {s:11} " + "  ".join(f"{c} {r['delta_after']:+.0f} / {r['delta_after_lm']:+.0f}"
                                      for c, r in t.items()))

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
        t = delta_table(sample)
        for p in PIPELINES:
            for c, r in t.items():
                boot[p][c].append(r[f"delta_{p}"])
            boot_pass[p] += verdict(t, p)[0]

    say(f"\nGate 1 verdict (delta recall >= +{MIN_GAIN_PP:g} pp in >= {MIN_CONDITIONS} of "
        f"{len(table)} conditions, and fixes > regressions):")
    for p, label in PIPELINES.items():
        ok, n_pass, fixes, regs = verdict(table, p)
        say(f"\n  {label}: {'PASS' if ok else 'FAIL'}  ({n_pass} conditions >= +{MIN_GAIN_PP:g} pp; "
            f"fixes {fixes}, regressions {regs})")
        for c, vals in boot[p].items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            say(f"    {c:8} delta {table[c][f'delta_{p}']:+6.1f} pp   95% CI [{lo:+.1f}, {hi:+.1f}]")
        say(f"    verdict holds in {100 * boot_pass[p] / a.bootstrap:.0f}% of {a.bootstrap} recording resamples")

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
        f"{hours * 60:.1f} min scored per sound, so one false alarm = {1 / hours:.0f} per hour: a coarse scale.")
    say("  - Recordings per sound: " + ", ".join(f"{s} {len(rs)}" for s, rs in recs.items())
        + ". Few recordings give wide confidence intervals.")
    say("  - Thresholds are chosen and evaluated on the same development set (in-sample).")
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
    write("gt_events.csv", events)
    write("separation.csv", sep_rows)
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved thresholds.csv, gt_events.csv, separation.csv, report.txt in {out_dir}")


if __name__ == "__main__":
    main()
