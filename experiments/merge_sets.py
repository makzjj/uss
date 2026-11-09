"""Merge development test sets into one combined set (Requirements v0.3.9, option A: recalibrate on more data).

Writes <out>/manifest.csv and <out>/settings.json, and results/<out name>_scores_pieces.csv, so the merged set
reads like any other test set (gate_analysis.load needs only those two files). Target clips from every set are
kept; clips present in more than one set (the target-free clips, built from the same backgrounds, interferers and
seeds) are kept ONCE, after checking that their manifest rows and per-piece scores really are identical. Sets
built from held-out targets or held-out backgrounds are refused.

Usage (inside the container, from the uss folder):
  python experiments/merge_sets.py --sets <TA>/testsets/gate1_v2 <TA>/testsets/extradev_v1 --out <TA>/testsets/devplus_v1
"""
import argparse
import csv
import json
import sys
from pathlib import Path

KEY_FIELDS = ("contains", "condition", "target_onset", "target_offset", "background", "interferer_source")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already exists; choose a new --out")
    manifest, pieces, origin, dupes = {}, {}, {}, 0
    fields = None
    for s in map(Path, a.sets):
        st = json.loads((s / "settings.json").read_text()) if (s / "settings.json").exists() else {}
        if st.get("target_set") == "heldout" or st.get("background_set") == "heldout":
            sys.exit(f"{s} was built from held-out data; it can never be merged into development data")
        with open(s / "manifest.csv") as fh:
            rows = list(csv.DictReader(fh))
        fields = fields or list(rows[0].keys())
        for f in rows[0].keys():
            if f not in fields:
                fields.append(f)
        pc = Path(f"results/{s.name}_scores_pieces.csv")
        if not pc.exists():
            sys.exit(f"Missing {pc}: run run_gate1.py --test_set {s} first")
        with open(pc) as fh:
            prow = list(csv.DictReader(fh))
        by_clip = {}
        for r in prow:
            by_clip.setdefault(r["clip"], []).append(r)
        for r in rows:
            name = Path(r["clip"]).name
            if name in manifest:
                old = manifest[name]
                diff = [f for f in KEY_FIELDS if f in r and f in old and r[f] != old[f]]
                if diff:
                    sys.exit(f"{name} appears in {origin[name]} and {s.name} with different {diff}; cannot merge")
                a_sc = sorted((x["sound"], x["version"], x["piece"], float(x["score"])) for x in pieces[name])
                b_sc = sorted((x["sound"], x["version"], x["piece"], float(x["score"])) for x in by_clip.get(name, []))
                if len(a_sc) != len(b_sc) or any(abs(p[3] - q[3]) > 1e-4 or p[:3] != q[:3] for p, q in zip(a_sc, b_sc)):
                    sys.exit(f"{name} appears in {origin[name]} and {s.name} with different scores; cannot merge")
                dupes += 1
                continue
            manifest[name], origin[name] = r, s.name
            pieces[name] = by_clip.get(name, [])
            if not pieces[name]:
                sys.exit(f"{name}: no per-piece scores in {pc}")
    out.mkdir(parents=True)
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["source_set"], extrasaction="ignore")
        w.writeheader()
        for name, r in manifest.items():
            w.writerow({**{f: r.get(f, "") for f in fields}, "source_set": origin[name]})
    pfields = list(next(iter(pieces.values()))[0].keys())
    pout = Path(f"results/{out.name}_scores_pieces.csv")
    with open(pout, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pfields)
        w.writeheader()
        for rs in pieces.values():
            w.writerows(rs)
    first = json.loads((Path(a.sets[0]) / "settings.json").read_text()) if (Path(a.sets[0]) / "settings.json").exists() else {}
    (out / "settings.json").write_text(json.dumps({
        **{k: first[k] for k in ("duration_s", "sample_rate", "hop_s") if k in first},   # read by score_events.py
        "merged_from": [str(Path(x)) for x in a.sets], "target_set": "dev+dev_extra", "background_set": "dev",
        "clips": len(manifest), "shared_clips_kept_once": dupes,
        "note": "Combined development set for recalibration (option A); not independent evidence."}, indent=2))
    n_t = sum(r["contains"] != "none" for r in manifest.values())
    print(f"Merged {len(a.sets)} sets into {out}: {len(manifest)} clips ({n_t} with a target), "
          f"{dupes} shared clips checked identical and kept once")
    print(f"Wrote {out}/manifest.csv and {pout}")


if __name__ == "__main__":
    main()