"""Gate 1 pilot: detection only vs detection after USS, on the test set.

For each sound (beeper, engine, jackhammer):
  1. Listener score on every clip as it is          -> "before USS"
  2. USS by example on every clip (that sound's examples folder)
  3. Listener score on every USS output              -> "after USS"
Clips containing that sound count as "with"; all other clips count as "without".

Listener settings are identical to detect_baseline.py (same model, 2 s pieces, 0.2 cut-off).
Note: USS asked BY EXAMPLE separates every 2 s piece; it does not apply the 0.2 cut-off itself.
The 0.2 cut-off here is applied only by the listener (it is the same value USS uses in its
class-id mode).

USS outputs are saved inside the test set folder (<test_set>/uss_outputs/), so they always
belong to that exact test set. Outputs that already exist are reused, so the script can be
stopped and restarted. The test set builder (build_scaper_set.py for gate1_v2, build_test_set.py
for the pilot set) deletes the whole test set folder, including uss_outputs/, whenever it rebuilds,
so cached outputs can never belong to an older set.

The summary is broken down by condition (ebr0, ebr-5, overlap) for hits, and by negative type
for false alarms: background only (bgonly), background + interferer (int), and clips that contain
a DIFFERENT target sound (other target).

Usage (inside the container, from the uss folder):
  python experiments/run_gate1.py
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from detect_baseline import USS_THRESHOLD, piece_scores  # noqa: E402
from uss.config import panns_paths_dict  # noqa: E402
from uss.utils import get_path, load_pretrained_panns, parse_yaml  # noqa: E402

TA = "/home/zmak002/docker_share/test_audio"
SOUNDS = {  # name: (USS class id, examples folder)
    "beeper": (319, f"{TA}/queries/beeper"),
    "engine": (343, f"{TA}/queries/engine"),
    "jackhammer": (420, f"{TA}/queries/jackhammer"),
}
CONFIG = "scripts/train_configs/ss_model=resunet30,querynet=at_soft,data=full.yaml"


def uss_output_dir(test_set, clip):
    return Path(test_set) / "uss_outputs" / Path(clip).stem


def uss_output(test_set, clip, queries_dir):
    return uss_output_dir(test_set, clip) / f"query={Path(queries_dir).name}.wav"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    p.add_argument("--out", default="",
                   help="default: results/<test set folder name>_scores.csv, so each set keeps its own file")
    a = p.parse_args()

    with open(Path(a.test_set) / "manifest.csv") as fh:
        clips = list(csv.DictReader(fh))

    configs = parse_yaml(CONFIG)
    sr = configs["data"]["sample_rate"]
    piece_samples = int(sr * configs["data"]["segment_seconds"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained_panns(model_type="Cnn14",
                                  checkpoint_path=get_path(panns_paths_dict["Cnn14"]),
                                  freeze=True).to(device)
    model.eval()

    results = []
    for name, (class_id, qdir) in SOUNDS.items():
        print(f"\n=== {name} (class {class_id}) ===")
        for i, row in enumerate(clips, start=1):
            clip = row["clip"]
            sep = uss_output(a.test_set, clip, qdir)
            if not sep.exists():
                print(f"[{i}/{len(clips)}] USS on {Path(clip).name}")
                subprocess.run(["uss", "-i", clip, "--queries_dir", qdir,
                                "--output_dir", str(uss_output_dir(a.test_set, clip))],
                               check=True, stdout=subprocess.DEVNULL)
            before = max(piece_scores(model, clip, class_id, sr, piece_samples, device))
            after = max(piece_scores(model, str(sep), class_id, sr, piece_samples, device))
            results.append({"sound": name, "clip": Path(clip).name,
                            "has_sound": int(row["contains"] == name),
                            "contains": row["contains"],
                            "condition": row.get("condition", ""),
                            "snr_db": row.get("target_ebr_db", row.get("snr_db", "")),
                            "score_before": round(before, 4), "score_after": round(after, 4)})

    out = Path(a.out or f"results/{Path(a.test_set).name}_scores.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"\nResults at the 0.2 cut-off (same value as USS's class-id mode; applied by the listener):")
    print(f"{'sound':12}{'hits before':>13}{'hits after':>12}{'false alarms before':>21}{'false alarms after':>20}")
    for name in SOUNDS:
        r = [x for x in results if x["sound"] == name]
        pos = [x for x in r if x["has_sound"]]
        neg = [x for x in r if not x["has_sound"]]
        hb = sum(x["score_before"] >= USS_THRESHOLD for x in pos)
        ha = sum(x["score_after"] >= USS_THRESHOLD for x in pos)
        fb = sum(x["score_before"] >= USS_THRESHOLD for x in neg)
        fa = sum(x["score_after"] >= USS_THRESHOLD for x in neg)
        print(f"{name:12}{hb:>8}/{len(pos):<4}{ha:>7}/{len(pos):<4}{fb:>16}/{len(neg):<4}{fa:>15}/{len(neg):<4}")
        print(f"{'':12}median score with sound: before {np.median([x['score_before'] for x in pos]):.3f}, "
              f"after {np.median([x['score_after'] for x in pos]):.3f}")
    # Breakdown: hits per condition, false alarms per negative type
    def neg_type(x):
        if x["contains"] == "none":
            return x["condition"] or "none"
        return "other target"

    print(f"\nBreakdown at the 0.2 cut-off (before -> after USS):")
    for name in SOUNDS:
        r = [x for x in results if x["sound"] == name]
        print(f"\n  {name}")
        conds = sorted({x["condition"] for x in r if x["has_sound"]})
        for c in conds:
            g = [x for x in r if x["has_sound"] and x["condition"] == c]
            hb = sum(x["score_before"] >= USS_THRESHOLD for x in g)
            ha = sum(x["score_after"] >= USS_THRESHOLD for x in g)
            print(f"    hits, {c or 'all':12} {hb:>3}/{len(g):<3} -> {ha:>3}/{len(g):<3}")
        for t in sorted({neg_type(x) for x in r if not x["has_sound"]}):
            g = [x for x in r if not x["has_sound"] and neg_type(x) == t]
            fb = sum(x["score_before"] >= USS_THRESHOLD for x in g)
            fa = sum(x["score_after"] >= USS_THRESHOLD for x in g)
            print(f"    false alarms, {t:12} {fb:>3}/{len(g):<3} -> {fa:>3}/{len(g):<3}")
    print(f"\nAll scores saved to {out}  (use gate1_summary.py for AUC per condition)")


if __name__ == "__main__":
    main()