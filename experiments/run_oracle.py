"""Perfect-separation upper bound for Gate 1: the listener on Scaper's isolated targets.

A perfect USS output for a sound is that sound's isolated event (events/<clip>_events/
foreground*_<sound>.wav, at its level in the mix) on clips that contain it, and silence on every
other clip. Scoring these with the same listener tells apart two reasons for a weak Gate 1 result:
  - the USS output is not clean enough       -> the oracle does clearly better than USS
  - the listener needs background context,   -> the oracle does not do better either
    or the target is hard to detect even clean

Writes results/<set>_oracle_pieces.csv in the same format as run_gate1.py's pieces file
(version 'oracle'); score_events.py picks it up automatically. No USS is run, so it takes minutes.

Usage (inside the container, from the uss folder):
  python experiments/run_oracle.py
"""
import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from detect_baseline import check_frame_readout, piece_details  # noqa: E402
from run_gate1 import CONFIG, SOUNDS, TA  # noqa: E402
from uss.config import panns_paths_dict  # noqa: E402
from uss.utils import get_path, load_pretrained_panns, parse_yaml  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    p.add_argument("--out", default="", help="default: results/<test set folder name>_oracle_pieces.csv")
    a = p.parse_args()

    ts = Path(a.test_set)
    with open(ts / "manifest.csv") as fh:
        clips = list(csv.DictReader(fh))

    configs = parse_yaml(CONFIG)
    sr = configs["data"]["sample_rate"]
    piece_samples = int(sr * configs["data"]["segment_seconds"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained_panns(model_type="Cnn14",
                                  checkpoint_path=get_path(panns_paths_dict["Cnn14"]),
                                  freeze=True).to(device)
    model.eval()
    check_frame_readout(model, piece_samples, device)

    lengths = {r["clip"]: len(librosa.load(path=r["clip"], sr=sr, mono=True)[0]) for r in clips}
    silence = {}   # (class id, length) -> piece details of an all-silent output
    pieces = []
    for name, (class_id, _) in SOUNDS.items():
        print(f"=== {name} (class {class_id}) ===")
        for row in clips:
            clip = Path(row["clip"])
            n = lengths[row["clip"]]
            if row["contains"] == name:
                refs = sorted((ts / "events" / f"{clip.stem}_events").glob(f"foreground*_{name}.wav"))
                if not refs:
                    raise SystemExit(f"No isolated {name} event for {clip.name} in events/")
                audio = sum(librosa.load(path=str(f), sr=sr, mono=True)[0] for f in refs)
                details = piece_details(model, librosa.util.fix_length(data=audio, size=n, axis=0),
                                        class_id, piece_samples, device)
            else:
                if (class_id, n) not in silence:
                    silence[(class_id, n)] = piece_details(model, np.zeros(n, dtype=np.float32),
                                                           class_id, piece_samples, device)
                details = silence[(class_id, n)]
            for k, (score, frames) in enumerate(details):
                pieces.append({"sound": name, "clip": clip.name, "version": "oracle", "piece": k,
                               "start_s": round(k * piece_samples / sr, 3), "score": round(score, 5),
                               "frames": " ".join(f"{v:.4f}" for v in frames)})

    out = Path(a.out or f"results/{ts.name}_oracle_pieces.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pieces[0].keys()))
        w.writeheader()
        w.writerows(pieces)
    print(f"Oracle per-piece scores saved to {out}; score_events.py reads them automatically")


if __name__ == "__main__":
    main()
