"""S4 preparation (server): one stream file and one config for the Pi streaming run.

Writes results/s4/:
  stream.flac     the 220 dev clips end to end, each exactly 10 s at 32 kHz mono, starting at sample 0,
                  so the Pi's 2 s windows are exactly the Gate 1 pieces. Stored losslessly when the clips
                  allow it (16- or 24-bit PCM); otherwise as 24-bit, with the rounding reported.
  s4_order.json   the clip order (seeded shuffle, fixed), each clip's start sample, and the server's
                  original-audio SED scores per window: the Pi's own scores must match these (alignment check).
  s4_config.json  everything the Pi needs, so nothing is copied by hand: class ids, the gate (policy, lo,
                  never-gated classes, skip-alerting), the frozen thresholds (offline gated pipeline, as in
                  schedule_sim.py) and the SED-only thresholds, queue rules (one running, max_wait waiting,
                  TTL, planned time per query) and where separated outputs are scored (core 3 or the SED
                  cores: the Pi run sets this per run).
  SHA256SUMS      hashes for the transfer check.

The frozen thresholds were set on FP32 USS scored by the PyTorch SED; the Pi runs INT8 USS (O1 met in this
configuration) and the ONNX SED (matches PyTorch to ~1e-6). Report the S4 result with that note.

Usage (inside the container, from the uss folder):
  python experiments/s4_prepare.py
  python experiments/schedule_sim.py --order_file results/s4/s4_order.json ...   # prediction for this order
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gate_analysis import gated, load  # noqa: E402
from schedule_sim import NAMES, gate_masks  # noqa: E402
from score_events import evaluate, match  # noqa: E402
from detect_baseline import USS_THRESHOLD  # noqa: E402
from run_gate1 import SOUNDS, TA  # noqa: E402

SR, CLIP_S, HOP = 32000, 10.0, 2.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--out", default="results/s4")
    ap.add_argument("--frozen", default="",
                    help="frozen two-threshold design (two_threshold.py JSON): thresholds, gate, query classes and queue "
                         "settings are taken from it, NEVER computed from this set's clips (required for a held-out "
                         "stream, which also needs run_gate1.py results for its alignment scores)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--policy", choices=["k1", "kmax"], default="kmax")
    ap.add_argument("--lo", type=float, default=0.1002)
    ap.add_argument("--exclude", nargs="*", default=["beeper"], choices=NAMES)
    ap.add_argument("--no_skip_alerting", action="store_true")
    ap.add_argument("--ttl_s", type=float, default=60.0)
    ap.add_argument("--max_wait", type=int, default=1)
    ap.add_argument("--t_plan_sep", type=float, default=9.46, help="planned separation time per query (Week 7 p95)")
    ap.add_argument("--t_plan_score", type=float, default=1.25,
                    help="planned detector pass per separated output on core 3 (Week 7, 1 core + contention)")
    ap.add_argument("--sed_s", type=float, default=0.66,
                    help="SED p95 beside USS (Week 7 ONNX SED, 2 cores); jobs arrive at window end + this")
    ap.add_argument("--onset_rel", type=float, default=0.5)
    a = ap.parse_args()

    ts = Path(a.test_set)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scores_csv = Path(f"results/{ts.name}_scores_pieces.csv")
    manifest, seq, gts = load(ts, scores_csv, a.onset_rel)
    clips = sorted(manifest)
    B = {s: seq[(s, "before")] for s in NAMES}
    U = {s: seq[(s, "after")] for s in NAMES}

    st = json.loads((ts / "settings.json").read_text()) if (ts / "settings.json").exists() else {}
    heldout = st.get("target_set") == "heldout" or "heldout" in ts.name.lower()
    two = None
    if a.frozen:
        fz = json.loads(Path(a.frozen).read_text())
        g = fz["args"]
        if a.no_skip_alerting:
            sys.exit("--no_skip_alerting is not the frozen design (it skips classes already alerting at t1)")
        a.policy, a.lo, a.exclude = g["policy"], g["lo"], g["exclude"]
        a.ttl_s, a.max_wait, a.sed_s, a.t_plan_sep = g["ttl_s"], g["max_wait"], g["sed_s"], g["t_plan"]   # frozen queue
        t2 = {s: fz["t2"][s] for s in NAMES}
        qcls = fz.get("final_query_classes", [s for s in NAMES if t2[s] is not None])
        two = {"t1": fz["t1"], "t2": t2, "query_classes": qcls, "frozen_file": a.frozen,
               "frozen_sha256": sha256(a.frozen)}
        thr = thr_sed = fz["t1"]                       # stage-1 and skip rule at t1
        anchor = fz["allowed_false_alarms"]
        print(f"Thresholds from the frozen design {a.frozen} (not computed from {ts.name}): t1 {fz['t1']}, t2 {t2}, "
              f"queries {qcls}")
    else:
        if heldout:
            sys.exit(f"{ts.name} is held-out data: thresholds must come from the frozen design (--frozen)")
        # ---------- frozen thresholds, exactly as schedule_sim.py (single-threshold design) ----------
        gate_classes = [s for s in NAMES if s not in a.exclude]
        anchor = {s: sum(evaluate(B[s], gts[s], USS_THRESHOLD)[1].values()) for s in NAMES}
        thr_sed = {s: match(B[s], gts[s], anchor[s])[0] for s in NAMES}
        mask = gate_masks(B, clips, a.lo, a.policy, gate_classes)
        off = {s: gated(B[s], U[s], mask[s]) for s in NAMES}
        thr = {s: match(off[s], gts[s], anchor[s])[0] for s in NAMES}

    # ---------- stream ----------
    order = list(np.random.default_rng(a.seed).permutation(clips))
    paths = {Path(r["clip"]).name: r["clip"] for r in csv.DictReader(open(ts / "manifest.csv"))}
    subtypes = {sf.info(paths[c]).subtype for c in order}
    exact = subtypes <= {"PCM_16", "PCM_24"}
    flac_sub = "PCM_16" if subtypes == {"PCM_16"} else "PCM_24"
    n = int(CLIP_S * SR)
    chunks, starts, worst_round = [], [], 0.0
    for i, c in enumerate(order):
        x, sr = sf.read(paths[c], dtype="float32", always_2d=False)
        if sr != SR or x.ndim != 1 or len(x) != n:
            sys.exit(f"{c}: expected {SR} Hz mono, exactly {n} samples; got {sr} Hz, shape {x.shape}")
        starts.append(i * n)
        chunks.append(x)
    stream = np.concatenate(chunks)
    stream_path = out / "stream.flac"
    sf.write(stream_path, stream, SR, format="FLAC", subtype=flac_sub)
    back, _ = sf.read(stream_path, dtype="float32")
    worst_round = float(np.abs(back - stream).max())
    if exact and worst_round != 0.0:
        sys.exit(f"Clips are {subtypes} but the FLAC round trip changed samples (max {worst_round:.2e})")
    print(f"Stream: {len(order)} clips, {len(stream) / SR / 60:.1f} min, clip format {sorted(subtypes)} -> FLAC "
          f"{flac_sub}, {'bit-exact' if exact else f'rounding up to {worst_round:.1e}'}; "
          f"{stream_path.stat().st_size / 1e6:.0f} MB")

    # ---------- order and expected SED scores ----------
    hops = n // int(HOP * SR)
    windows = []
    for i, c in enumerate(order):
        for k in range(hops):
            windows.append({"window": i * hops + k, "clip": c, "piece": k,
                            "server_before": {s: B[s][c][k][0] for s in NAMES}})
    order_json = {"seed": a.seed, "clips": order, "clip_samples": n, "hop_samples": int(HOP * SR),
                  "start_samples": starts, "windows": windows,
                  "expected_alignment_tolerance": 1e-5 if exact else 1e-4}
    (out / "s4_order.json").write_text(json.dumps(order_json))

    # ---------- config for the Pi ----------
    config = {
        "class_ids": {s: SOUNDS[s][0] for s in NAMES},
        "gate": {"policy": a.policy, "lo": a.lo, "never_gated": a.exclude, "skip_alerting": not a.no_skip_alerting},
        "two_threshold": two,
        "thresholds_frozen": thr, "thresholds_sed_only": thr_sed, "allowed_false_alarms": anchor,
        "queue": {"running": 1, "max_wait": a.max_wait, "ttl_s": a.ttl_s,
                  "t_plan_sep_per_query_s": a.t_plan_sep, "t_plan_score_per_output_s": a.t_plan_score},
        "score_outputs_on": "set per Pi run: core3 or sed_cores",
        "note": "Frozen thresholds from FP32 USS + PyTorch SED; the Pi runs INT8 USS + ONNX SED.",
        "sed_p95_s": a.sed_s,
        "provenance": {"test_set": str(ts), "scores_pieces": str(scores_csv), "scores_sha256": sha256(scores_csv),
                       "order_seed": a.seed},
        "stream": {"file": "stream.flac", "sample_rate": SR, "clip_s": CLIP_S, "hop_s": HOP,
                   "flac_subtype": flac_sub, "bit_exact": exact},
    }
    (out / "s4_config.json").write_text(json.dumps(config, indent=2))
    with open(out / "SHA256SUMS", "w") as fh:
        for f in ("stream.flac", "s4_order.json", "s4_config.json"):
            fh.write(f"{sha256(out / f)}  {f}\n")
    print("Frozen thresholds: " + ", ".join(f"{s} {thr[s]:.4f} (SED-only {thr_sed[s]:.4f})" for s in NAMES))
    print(f"Wrote {out}/stream.flac, s4_order.json, s4_config.json, SHA256SUMS")
    common = (f"python experiments/schedule_sim.py --order_file {out}/s4_order.json --policy {a.policy} "
              f"--lo {a.lo:g} --exclude {' '.join(a.exclude) or ''} --ttl_s {a.ttl_s:g} --max_wait {a.max_wait} "
              f"--sed_s {a.sed_s:g}" + ("" if a.no_skip_alerting else " --skip_alerting"))
    sep, score = a.t_plan_sep, a.t_plan_score
    if two:
        print("Two-threshold design: compare the Pi run with heldout_eval.py's or two_threshold.py's scheduled "
              "numbers for this set (schedule_sim.py models the single-threshold design only).")
        return
    print("Predictions for this exact order (planned times; replace with the S4-measured times afterwards):")
    print(f"  core 3 scores outputs:     {common} --t_query {sep + score:.2f} --t_plan {sep + score:.2f}")
    print(f"  SED cores score outputs:   {common} --t_query {sep:.2f} --t_plan {sep:.2f} "
          f"--score_delay_s {2.0 + 2 * a.sed_s:.2f}")   # one hop + main pass + scoring pass, after the last separation


if __name__ == "__main__":
    main()