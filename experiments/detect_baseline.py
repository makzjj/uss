"""Detection-only baseline, using exactly the same listener setup as USS.

Matches uss/inference.py:
  - same Cnn14 model and weights (uss.utils.load_pretrained_panns, ~/.cache/panns/Cnn14_mAP=0.431.pth)
  - same audio loading (librosa, 32 kHz, mono)
  - same 2-second pieces, last piece padded with silence (from the USS settings file)
  - same "sound is present" cut-off: 0.2 (USS's at_threshold, used in its class-id and
    levels modes; USS asked by example does not use it)
  - USS decides piece by piece: it only separates a 2 s piece if the listener's
    score for the queried class reaches 0.2, and outputs silence otherwise.
    Here a clip counts as "sound present" if ANY piece reaches the cut-off,
    i.e. exactly when USS (asked for this single class) would not output silence.
  - score ONE class per run, and give USS the same single --class_ids if comparing
    with USS's own gating

Usage (inside the container, from the uss folder):
  python experiments/detect_baseline.py --class_id 319 \
      --pos /path/with_sound1.wav /path/with_sound2.wav \
      --neg /path/without1.flac /path/without2.wav \
      --out results/detect_beeper.csv

Class ids: 319 = Reversing beeps, 349 = Heavy engine, 420 = Jackhammer
"""
import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import torch

from uss.config import panns_paths_dict
from uss.utils import get_path, load_pretrained_panns, parse_yaml

DEFAULT_CONFIG = "scripts/train_configs/ss_model=resunet30,querynet=at_soft,data=full.yaml"
USS_THRESHOLD = 0.2  # at_threshold in uss/inference.py (class-id / levels mode only)
EXTRA_CUTOFFS = [0.05, 0.1, 0.3, 0.5]


def piece_scores(model, path, class_id, sr, piece_samples, device):
    """Confidence for class_id in each 2 s piece, computed the same way as USS."""
    audio, _ = librosa.load(path=path, sr=sr, mono=True)
    if len(audio) == 0:
        raise SystemExit(f"Empty audio file: {path}")
    scores = []
    pointer = 0
    while pointer < len(audio):
        piece = librosa.util.fix_length(
            data=audio[pointer: pointer + piece_samples], size=piece_samples, axis=0)
        x = torch.Tensor(piece).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(input=x)["clipwise_output"][0, class_id].item()
        scores.append(prob)
        pointer += piece_samples
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--class_id", type=int, required=True)
    p.add_argument("--pos", nargs="+", required=True, help="clips that contain the sound")
    p.add_argument("--neg", nargs="+", required=True, help="clips that do not contain the sound")
    p.add_argument("--config_yaml", default=DEFAULT_CONFIG)
    p.add_argument("--out", default="results/detect_baseline.csv")
    a = p.parse_args()

    configs = parse_yaml(a.config_yaml)
    sr = configs["data"]["sample_rate"]
    piece_samples = int(sr * configs["data"]["segment_seconds"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained_panns(
        model_type="Cnn14",
        checkpoint_path=get_path(panns_paths_dict["Cnn14"]),
        freeze=True,
    ).to(device)
    model.eval()

    rows = []
    for label, files in ((1, a.pos), (0, a.neg)):
        for f in files:
            s = piece_scores(model, f, a.class_id, sr, piece_samples, device)
            rows.append({"file": f, "has_sound": label, "clip_score": max(s), "pieces": s})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "has_sound", "clip_score_max", "piece_scores"])
        for r in rows:
            w.writerow([r["file"], r["has_sound"], round(r["clip_score"], 4),
                        " ".join(f"{x:.3f}" for x in r["pieces"])])

    pos = [r["clip_score"] for r in rows if r["has_sound"] == 1]
    neg = [r["clip_score"] for r in rows if r["has_sound"] == 0]

    print(f"\nSettings: {sr} Hz, {piece_samples / sr:.1f} s pieces, Cnn14 (same as USS), device {device}")
    print(f"Class {a.class_id}: {len(pos)} clips with the sound, {len(neg)} without")
    print(f"Clip score, WITH sound:    median {np.median(pos):.3f}  (min {min(pos):.3f}, max {max(pos):.3f})")
    print(f"Clip score, WITHOUT sound: median {np.median(neg):.3f}  (min {min(neg):.3f}, max {max(neg):.3f})")
    print(f"\n{'cut-off':>9} {'hits':>10} {'false alarms':>14}")
    for c in sorted([USS_THRESHOLD] + EXTRA_CUTOFFS):
        hits = sum(s >= c for s in pos)
        fa = sum(s >= c for s in neg)
        tag = "  <- same as USS's class-id mode" if c == USS_THRESHOLD else ""
        print(f"{c:>9} {hits:>5}/{len(pos):<4} {fa:>8}/{len(neg):<4}{tag}")
    print(f"\nPer-clip and per-piece scores saved to {out}")


if __name__ == "__main__":
    main()