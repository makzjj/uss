"""Week 6 task C: which classes and SED confidences give fixes vs regressions, and p_min for S4.

Works only from saved Gate 1 scores (no USS or GPU needed):
  <test_set>/manifest.csv, settings.json
  results/<set>_scores_pieces.csv   (run_gate1.py: SED score per 2 s piece, before and after USS)

Why per-piece gating can be simulated exactly: USS separates each 2 s segment on its own, with
no overlap, on the same 2 s grid as the SED pieces (uss/inference.py). So a gated system's output
for a piece equals the saved always-on USS output for that piece; a piece the gate skips keeps
its original score. Native USS output and the frozen max rule are used throughout (v0.3.6).

Definitions follow Requirements v0.3.6 and reuse score_events.py's own functions, so matching,
thresholds and false alarms are identical to the Gate 1 report.

1. Fixes and regressions (always-on USS vs SED only, at matched false alarms), per class, with
   the SED confidence the gate would see: the highest original score for that class among the
   pieces overlapping the event.
2. What sets the always-on thresholds: the original SED score, for the queried class, of the
   pieces whose USS output set each threshold. A confidence-band gate can only help if these
   pieces have LOWER original scores than the pieces that give the fixes.
3. Provisional SED-confidence gate (recovery mode). For class s, a piece is gated when the
   original score for s >= lo (default, --upper none). With --upper sed the gate is a band that
   also stops at the SED-only threshold thr_s. The band looks natural (pieces above thr_s are
   already stage-1 alerts), but under one matched threshold per pipeline it ADDS regressions:
   the threshold still rises, while those already-detected pieces get no USS boost to stay
   above it. With --upper none, lo = 0 reproduces always-on USS exactly. Two query policies:
     k1   class-aware: query only the classes whose band fired (k = classes fired per job)
     kmax recovery as defined in v0.3.6: any class firing launches a job querying all gate classes (3 unless --exclude)
   For every lo: activation rate p (jobs / SED hops; hop = window = 2 s, so no merging),
   mean k, USS load p*k against the one-core budget, share of always-on fixes captured, and the
   gated pipeline re-matched at the same false alarms: delta recall, fixes, regressions.
4. p_min = lowest p capturing >= 80% of always-on fixes (v0.3.6 method), for the real gate and
   for the oracle (gate fires only on the pieces that give the fixes).
5. Random gating at the p_min setting (same p and k, random hops): R3 groundwork.
6. USS load on target-free clips only (background only, background + interferer). The dev set is
   mostly target clips, so p on all clips overstates how often the gate fires on ordinary street
   audio. The target-free load is the closer estimate of everyday cost; bursts during real
   target events come on top of it and are handled by the queue rules.

Caveats printed in the report: everything is in-sample on the dev set; fix counts are tiny;
the dev set is target-rich, so p here is higher than it would be on ordinary street audio.

Writes to results/<set>_gate/: events.csv, setters.csv, sweep.csv, report.txt

Usage (inside the container, from the uss folder):
  python experiments/gate_analysis.py
  python experiments/gate_analysis.py --exclude beeper     # class-aware variant: never gate beeper
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score_events import (COLLAR_S, CONDITIONS, evaluate, fuse, in_window, match,  # noqa: E402
                          onset_time, unmatched_peaks)
from detect_baseline import USS_THRESHOLD  # noqa: E402
from run_gate1 import SOUNDS, TA  # noqa: E402

PIECE_S = 2.0
FIX_SHARE = 0.80           # v0.3.6: p_min captures >= 80% of fixes
T_USS_S = 19.4             # one query on core 3 beside SED (Week 5)
USS_BUSY_MAX = 0.7         # RTF limit applied to the USS core
CONF_BINS = [0.0, 0.01, 0.03, 0.1, 0.2, 1.01]

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def load(ts, pieces_path, onset_rel):
    with open(ts / "manifest.csv") as fh:
        manifest = {Path(r["clip"]).name: r for r in csv.DictReader(fh)}
    raw = defaultdict(lambda: defaultdict(dict))
    with open(pieces_path) as fh:
        for r in csv.DictReader(fh):
            if r["version"] not in ("before", "after"):
                continue
            t = onset_time(float(r["start_s"]), r["frames"], onset_rel)
            raw[(r["sound"], r["version"])][r["clip"]][int(r["piece"])] = (float(r["score"]), t)
    seq = {key: {c: [ps[k] for k in sorted(ps)] for c, ps in by.items()} for key, by in raw.items()}
    for s in SOUNDS:
        for v in ("before", "after"):
            missing = set(manifest) - set(seq.get((s, v), {}))
            if missing:
                raise SystemExit(f"No '{v}' pieces for {s} on {len(missing)} clips; rerun run_gate1.py")
    gts = {s: {c: (float(r["target_onset"]), float(r["target_offset"]))
               for c, r in manifest.items() if r["contains"] == s} for s in SOUNDS}
    return manifest, seq, gts


def gated(before, uss, gate):
    """Max rule applied only on gated pieces; gate[clip][k] is True where USS ran."""
    return {c: [u if g and u[0] > b[0] else b for b, u, g in zip(before[c], uss[c], gate[c])]
            for c in before}


def overlapping(k, win):
    start = k * PIECE_S
    return start < win[1] and start + PIECE_S > win[0] - COLLAR_S


def bin_label(x):
    for lo, hi in zip(CONF_BINS, CONF_BINS[1:]):
        if lo <= x < hi:
            return f"[{lo:g}, {min(hi, 1):g})"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--pieces", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--onset_rel", type=float, default=0.5)
    ap.add_argument("--exclude", nargs="*", default=[], choices=list(SOUNDS),
                    help="classes the gate never fires for (class-aware variant)")
    ap.add_argument("--upper", choices=["none", "sed"], default="none",
                    help="none: gate = original >= lo; sed: band lo <= original < SED-only threshold")
    ap.add_argument("--random_trials", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ts = Path(a.test_set)
    pieces_path = Path(a.pieces or f"results/{ts.name}_scores_pieces.csv")
    out_dir = Path(a.out_dir or f"results/{ts.name}_gate" + ("_band" if a.upper == "sed" else "")
                   + (f"_no-{'-'.join(a.exclude)}" if a.exclude else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, seq, gts = load(ts, pieces_path, a.onset_rel)
    clip_s = json.loads((ts / "settings.json").read_text())["duration_s"]
    hours = len(manifest) * clip_s / 3600
    clips = sorted(manifest)
    n_hops = sum(len(seq[(next(iter(SOUNDS)), "before")][c]) for c in clips)
    gate_classes = [s for s in SOUNDS if s not in a.exclude]
    free = [c for c in clips if manifest[c]["contains"] == "none"]
    n_free_hops = sum(len(seq[(next(iter(SOUNDS)), "before")][c]) for c in free)
    if not n_free_hops:
        raise SystemExit("No target-free clips in the manifest")

    def group(s, c):
        r = manifest[c]
        if r["contains"] == s:
            return "same sound"
        return r["condition"] if r["contains"] == "none" else f"other target ({r['contains']})"

    # ---------- 1. SED-only and always-on, matched at the SED-only false alarms at 0.2 ----------
    B = {s: seq[(s, "before")] for s in SOUNDS}
    U = {s: seq[(s, "after")] for s in SOUNDS}
    A = {s: fuse(B[s], U[s], "max") for s in SOUNDS}
    anchor, thr_b, hit_b, thr_a, hit_a = {}, {}, {}, {}, {}
    for s in SOUNDS:
        anchor[s] = sum(evaluate(B[s], gts[s], USS_THRESHOLD)[1].values())
        thr_b[s], hit_b[s], _ = match(B[s], gts[s], anchor[s])
        thr_a[s], hit_a[s], _ = match(A[s], gts[s], anchor[s])

    say(f"Task C gate analysis: {ts.name}, {len(clips)} clips, {n_hops} SED hops of {PIECE_S:g} s, "
        f"frozen max rule, native USS output")
    say(f"Gate classes: {', '.join(gate_classes)}" + (f" (never gated: {', '.join(a.exclude)})" if a.exclude else ""))
    say(f"\nMatched thresholds (false alarms allowed = SED-only count at {USS_THRESHOLD}):")
    say(f"  {'class':11}{'allowed':>8}{'SED only':>10}{'always-on':>11}")
    for s in SOUNDS:
        say(f"  {s:11}{anchor[s]:>8}{thr_b[s]:>10.4f}{thr_a[s]:>11.4f}")

    events, fix_pieces = [], defaultdict(set)    # fix_pieces[s] = {(clip, k)}
    for s in SOUNDS:
        for c, win in gts[s].items():
            hb, ha = hit_b[s][c], hit_a[s][c]
            kind = "fix" if ha and not hb else "regression" if hb and not ha else "hit both" if hb else "miss both"
            conf = max(B[s][c][k][0] for k in range(len(B[s][c])) if overlapping(k, win))
            if kind == "fix":
                for k, ((sb, _), (sa, ta)) in enumerate(zip(B[s][c], A[s][c])):
                    if sa >= thr_a[s] and in_window(ta, win) and U[s][c][k][0] > sb:
                        fix_pieces[s].add((c, k))
                if not any(cc == c for cc, _ in fix_pieces[s]):
                    raise SystemExit(f"Fix event {s} {c} has no piece whose USS output made the hit; "
                                     f"check matching before trusting this analysis")
            r = manifest[c]
            events.append({"class": s, "clip": c, "condition": r["condition"], "recording": r["target_source"],
                           "outcome": kind, "sed_conf": round(conf, 5), "conf_bin": bin_label(conf),
                           "thr_sed_only": round(thr_b[s], 5), "thr_always_on": round(thr_a[s], 5)})

    say("\n1. Always-on USS vs SED only, per event (class x SED confidence the gate would see):")
    say("   SED confidence = highest original score for the class among pieces overlapping the event")
    say(f"   {'class':11}{'confidence':>14}{'events':>8}{'fixes':>7}{'regress':>9}{'hit both':>10}{'miss both':>11}")
    for s in SOUNDS:
        for lo, hi in zip(CONF_BINS, CONF_BINS[1:]):
            ev = [e for e in events if e["class"] == s and lo <= e["sed_conf"] < hi]
            if ev:
                cnt = {k: sum(e["outcome"] == k for e in ev) for k in ("fix", "regression", "hit both", "miss both")}
                say(f"   {s:11}{bin_label(lo):>14}{len(ev):>8}{cnt['fix']:>7}{cnt['regression']:>9}"
                    f"{cnt['hit both']:>10}{cnt['miss both']:>11}")
    n_fix = sum(e["outcome"] == "fix" for e in events)
    n_reg = sum(e["outcome"] == "regression" for e in events)
    say(f"   total: {n_fix} fixes, {n_reg} regressions")
    for s in SOUNDS:
        fx = [e for e in events if e["class"] == s and e["outcome"] == "fix"]
        if fx:
            say(f"   {s} fixes: SED confidence " + ", ".join(f"{e['sed_conf']:.3f} ({e['condition']})" for e in fx))

    # ---------- 2. threshold setters: original score of the pieces that set each always-on threshold ----------
    say("\n2. What sets each always-on threshold, with the ORIGINAL SED score of that piece for the queried class.")
    say("   A band gate (fire only when the original score >= lo) removes a setter if its original score < lo,")
    say("   while keeping a fix only if the fix piece's original score >= lo:")
    setters = []
    for s in SOUNDS:
        peaks = sorted(((sc, k, c) for c, (sc, k) in unmatched_peaks(A[s], gts[s]).items()), reverse=True)
        fixo = sorted(B[s][c][k][0] for c, k in fix_pieces[s])
        say(f"   {s}: always-on threshold {thr_a[s]:.4f} (SED only {thr_b[s]:.4f}); "
            f"fix pieces' original scores: {', '.join(f'{x:.3f}' for x in fixo) or 'none'}")
        for i, (sc, k, c) in enumerate(peaks[:anchor[s] + 3]):
            orig = B[s][c][k][0]
            src = "USS output" if U[s][c][k][0] > orig else "original"
            mark = "  <- sets threshold" if i == anchor[s] else ""
            say(f"     {sc:.4f} from {src:10} original {orig:.4f}  {c} ({group(s, c)}, piece {k}){mark}")
            setters.append({"class": s, "rank": i + 1, "score": sc, "source": src, "original_score": orig,
                            "clip": c, "group": group(s, c), "piece": k, "sets_threshold": i == anchor[s]})

    # ---------- 3. band-gate sweep ----------
    def gate_masks(lo):
        """Per class: True where original >= lo (and < SED-only threshold with --upper sed)."""
        top = {s: thr_b[s] if a.upper == "sed" else math.inf for s in SOUNDS}
        return {s: {c: [s in gate_classes and lo <= b[0] < top[s] for b in B[s][c]] for c in clips}
                for s in SOUNDS}

    def run_gate(fired, policy):
        """fired[s][c][k]: band fired for class s. Returns p, mean k, captured fixes, per-class gated dets."""
        s0 = next(iter(SOUNDS))
        hop_any = {c: [any(fired[s][c][k] for s in SOUNDS) for k in range(len(B[s0][c]))] for c in clips}
        jobs = sum(sum(v) for v in hop_any.values())
        if policy == "k1":
            ran = fired
            queries = sum(sum(sum(fired[s][c]) for c in clips) for s in SOUNDS)
        else:
            never = {c: [False] * len(v) for c, v in hop_any.items()}
            ran = {s: hop_any if s in gate_classes else never for s in SOUNDS}
            queries = jobs * len(gate_classes)
        captured = sum(any(ran[s][c][k] for cc, k in fix_pieces[s] if cc == c)
                       for s in gate_classes for c in {cc for cc, _ in fix_pieces[s]})
        q_free = sum(sum(sum(ran[s][c]) for c in free) for s in gate_classes)
        dets = {s: gated(B[s], U[s], ran[s]) for s in SOUNDS}
        return jobs / n_hops, (queries / jobs if jobs else 0.0), captured, dets, q_free / n_free_hops

    def score(dets):
        """Re-matched at the same allowed false alarms: per condition delta recall, fixes, regressions."""
        rows = []
        for s in SOUNDS:
            _, hits, _ = match(dets[s], gts[s], anchor[s])
            rows += [{"class": s, "condition": manifest[c]["condition"], "hb": hit_b[s][c], "hg": hits[c]}
                     for c in gts[s]]
        out = {}
        for cond in CONDITIONS:
            ev = [r for r in rows if r["condition"] == cond]
            out[cond] = 100 * (np.mean([r["hg"] for r in ev]) - np.mean([r["hb"] for r in ev]))
        fixes = sum(r["hg"] and not r["hb"] for r in rows)
        regs = sum(r["hb"] and not r["hg"] for r in rows)
        per_class = {s: (sum(r["hg"] and not r["hb"] for r in rows if r["class"] == s),
                         sum(r["hb"] and not r["hg"] for r in rows if r["class"] == s)) for s in SOUNDS}
        return out, fixes, regs, per_class

    fix_events = {s: {c for c, _ in fix_pieces[s]} for s in SOUNDS}
    if sum(len(v) for v in fix_events.values()) != n_fix:
        raise SystemExit("Fix counts disagree between sections 1 and 3; check fix-piece detection")
    n_fix_events = sum(len(fix_events[s]) for s in gate_classes)
    given_up = {s: len(fix_events[s]) for s in a.exclude if fix_events[s]}
    need = math.ceil(FIX_SHARE * n_fix_events) if n_fix_events else 0
    budget = USS_BUSY_MAX * PIECE_S / T_USS_S          # max p*k on the USS core
    los = sorted({0.0} | {round(x, 4) for x in np.geomspace(0.001, 0.2, 24)}
                 | {round(B[s][c][k][0], 5) for s in SOUNDS for c, k in fix_pieces[s]})
    los = [lo for lo in los if lo < max(thr_b.values()) or a.upper == "none"]

    say("\n3. Provisional SED-confidence gate: fire for class s when original score >= lo"
        + (" and < SED-only threshold (band)." if a.upper == "sed" else " (lo = 0 is always-on USS)."))
    say(f"   USS core budget: p * k <= {USS_BUSY_MAX} x {PIECE_S:g} s / {T_USS_S} s = {budget:.3f}. "
        f"Always-on fixes to capture: {n_fix_events} (80% = {need}).")
    if given_up:
        say("   Fixes given up by never gating " + ", ".join(f"{s} ({n})" for s, n in given_up.items())
            + "; they are not in the 80% target.")
    say(f"   d = delta recall (pp) vs SED only, re-matched at the same false alarms; f/r = event fixes/regressions")
    sweep, by_policy = [], defaultdict(list)
    for policy in ("k1", "kmax"):
        say(f"\n   policy {policy} ({'query only classes whose gate fired' if policy == 'k1' else 'query all gate classes'}):")
        say(f"   {'lo':>9}{'p':>8}{'k':>6}{'p*k':>8}{'budget':>8}{'p*k free':>10}{'budget':>8}{'fixes cap':>11}"
            + "".join(f"{'d ' + c:>11}" for c in CONDITIONS) + f"{'f/r':>8}")
        for lo in los:
            p, k, cap, dets, lf = run_gate(gate_masks(lo), policy)
            d, fx, rg, pc = score(dets)
            ok = "ok" if p * k <= budget else "OVER"
            okf = "ok" if lf <= budget else "OVER"
            say(f"   {lo:>9.5f}{p:>8.3f}{k:>6.2f}{p * k:>8.3f}{ok:>8}{lf:>10.3f}{okf:>8}{cap:>6}/{n_fix_events:<4}"
                + "".join(f"{d[c]:>+11.1f}" for c in CONDITIONS) + f"{fx:>5}/{rg:<3}")
            row = {"policy": policy, "lo": lo, "p": round(p, 4), "k": round(k, 3), "load_pk": round(p * k, 4),
                   "within_budget": p * k <= budget, "load_pk_target_free": round(lf, 4),
                   "target_free_within_budget": lf <= budget,
                   "fixes_captured": cap, "fixes_total": n_fix_events,
                   **{f"delta_{c}": round(d[c], 1) for c in CONDITIONS}, "fixes": fx, "regressions": rg,
                   "fixes_gate_classes": sum(pc[s][0] for s in gate_classes),
                   **{f"fix_reg_{s}": f"{pc[s][0]}/{pc[s][1]}" for s in SOUNDS}}
            sweep.append(row)
            by_policy[policy].append((row, dets))

    # ---------- 4. p_min ----------
    say(f"\n4. p_min (v0.3.6 definition) = lowest p capturing >= {need} of {n_fix_events} always-on fixes (80%).")
    say("   'Captured' means the fix piece was gated. The gated pipeline is then re-matched, so its own")
    say("   fix count can differ; the lowest p at which the re-matched gate-class fixes reach the same")
    say("   number is reported on the next line as a check, not as p_min:")
    oracle_hops = {(c, k) for s in gate_classes for c, k in fix_pieces[s]}
    oracle_q = sum(len(fix_pieces[s]) for s in gate_classes)
    say(f"   oracle (fires only on fix pieces): p = {len(oracle_hops) / n_hops:.4f} "
        f"({len(oracle_hops)} of {n_hops} hops), k = {oracle_q / max(len(oracle_hops), 1):.2f} at policy k1")
    pmin_rows = {}
    for policy, rows in by_policy.items():
        okrows = [r for r, _ in rows if r["fixes_captured"] >= need]
        if not okrows or not n_fix_events:
            say(f"   {policy}: no lo captures {need} fixes")
            continue
        best = min(okrows, key=lambda r: (r["p"], -r["lo"]))
        pmin_rows[policy] = best
        say(f"   {policy}: p_min = {best['p']:.4f} at lo = {best['lo']:.4f}, k = {best['k']:.2f}, "
            f"load p*k = {best['load_pk']:.4f} ({'within' if best['within_budget'] else 'OVER'} budget {budget:.3f}); "
            f"gated delta recall " + " / ".join(f"{best[f'delta_{c}']:+.1f}" for c in CONDITIONS)
            + f" pp, fixes/regressions {best['fixes']}/{best['regressions']}")
        kept = [r for r, _ in rows if r["fixes_gate_classes"] >= need]
        if kept:
            chk = min(kept, key=lambda r: (r["p"], -r["lo"]))
            say(f"      check: re-matched gate-class fixes >= {need} first at p = {chk['p']:.4f} "
                f"(lo = {chk['lo']:.4f})" + ("  <- same setting" if chk is best else "  <- differs from p_min"))
        else:
            say(f"      check: re-matched gate-class fixes never reach {need} at any lo")

    # ---------- 5. random gating at the p_min setting ----------
    rng = np.random.default_rng(a.seed)
    hop_list = [(c, k) for c in clips for k in range(len(B[next(iter(SOUNDS))][c]))]
    for policy, best in pmin_rows.items():
        n_jobs = round(best["p"] * n_hops)
        if n_jobs == 0:
            continue
        if n_jobs >= n_hops:
            say(f"\n5. Random gating at {policy} p_min skipped: p = 1 gates every hop, so random gating is identical.")
            continue
        deltas, frs = [], []
        for _ in range(a.random_trials):
            pick = [hop_list[i] for i in rng.choice(len(hop_list), size=n_jobs, replace=False)]
            fired = {s: {c: [False] * len(B[s][c]) for c in clips} for s in SOUNDS}
            base, frac = int(best["k"]), best["k"] - int(best["k"])
            for c, k in pick:                    # k1: random classes per job, mean k matched to the gate
                m = len(gate_classes) if policy == "kmax" else max(1, base + (rng.random() < frac))
                for s in rng.choice(gate_classes, size=min(m, len(gate_classes)), replace=False):
                    fired[s][c][k] = True
            _, _, _, dets, _ = run_gate(fired, policy)
            d, fx, rg, _ = score(dets)
            deltas.append(np.mean([d[c] for c in CONDITIONS]))
            frs.append(fx - rg)
        gate_mean = np.mean([best[f"delta_{c}"] for c in CONDITIONS])
        lo95, hi95 = np.percentile(deltas, [2.5, 97.5])
        say(f"\n5. Random gating at {policy} p_min ({n_jobs} random hops, {a.random_trials} trials): mean delta recall "
            f"{np.mean(deltas):+.1f} pp [95% of trials {lo95:+.1f}, {hi95:+.1f}]; confidence gate {gate_mean:+.1f} pp; "
            f"confidence gate better in {100 * np.mean(np.array(deltas) < gate_mean):.0f}% of random trials, "
            f"tied in {100 * np.mean(np.isclose(deltas, gate_mean)):.0f}%")

    say(f"\n6. USS load on target-free clips only ({len(free)} clips, {n_free_hops} hops, "
        f"{n_free_hops * PIECE_S / 60:.1f} min), budget {budget:.3f}:")
    for policy, rows in by_policy.items():
        r_list = [r for r, _ in rows]
        if policy in pmin_rows:
            b = pmin_rows[policy]
            say(f"   {policy}: at p_min (lo = {b['lo']:.4f}) load on all clips {b['load_pk']:.3f}, "
                f"on target-free clips {b['load_pk_target_free']:.3f} "
                f"({'within' if b['target_free_within_budget'] else 'OVER'} budget, "
                f"{b['load_pk_target_free'] / budget:.1f}x)")
        fits = [r for r in r_list if r["target_free_within_budget"]]
        if fits:
            f0 = min(fits, key=lambda r: r["lo"])
            say(f"   {policy}: lowest lo whose target-free load fits: lo = {f0['lo']:.4f}, load {f0['load_pk_target_free']:.3f}; "
                f"there it captures {f0['fixes_captured']}/{f0['fixes_total']} fixes, delta recall "
                + " / ".join(f"{f0[f'delta_{c}']:+.1f}" for c in CONDITIONS)
                + f" pp, fixes/regressions {f0['fixes']}/{f0['regressions']}")
        else:
            say(f"   {policy}: no lo in the sweep keeps the target-free load within budget")
    say(f"   Target-free audio here is dev backgrounds with interferers; {n_free_hops * PIECE_S / 60:.1f} min is short,")
    say("   so treat these loads as rough. Field audio (Sem 2) gives the real figure.")

    say("\nLimits:")
    say(f"  - In-sample: thresholds, fixes, the band and p_min all come from the same dev set they are scored on.")
    say(f"    Confirm any chosen setting on a set built from the held-out backgrounds before reporting it.")
    say(f"  - {n_fix_events} always-on fixes in total, so the 80% rule moves in steps of one event "
        f"({100 / max(n_fix_events, 1):.0f}% each).")
    say(f"  - The dev set is target-rich (most clips contain a target), so p here overstates the activation")
    say(f"    rate on ordinary street audio; S4 should use p from target-free audio as well as p_min.")
    say(f"  - {hours * 60:.1f} min scored per class; one false alarm = {1 / hours:.1f} per hour.")

    def write(name, rows):
        if rows:
            with open(out_dir / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    write("events.csv", events)
    write("setters.csv", setters)
    write("sweep.csv", sweep)
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved events.csv, setters.csv, sweep.csv, report.txt in {out_dir}")


if __name__ == "__main__":
    main()