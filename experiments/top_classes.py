"""What does the listener (Cnn14) think is in each file? Top 5 classes per file.

Uses the same listener setup as detect_baseline.py (2 s pieces). For each file it shows
the 5 classes with the highest score in any 2 s piece. It can also show the top classes
of a saved USS query description (the .pkl files in query_conditions/).

Usage (inside the container, from the uss folder):
  python experiments/top_classes.py --files a.wav b.flac ... [--pkls query_conditions/*/*.pkl]
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from detect_baseline import DEFAULT_CONFIG  # noqa: E402
from uss.config import panns_paths_dict  # noqa: E402
from uss.utils import get_path, load_pretrained_panns, parse_yaml  # noqa: E402

import librosa  # noqa: E402

WATCH = {319: "Reversing beeps", 349: "Heavy engine", 420: "Jackhammer"}


def class_names():
    with open("metadata/class_labels_indices.csv") as fh:
        return {int(r["index"]): r["display_name"] for r in csv.DictReader(fh)}


def file_probs(model, path, sr, piece, device):
    audio, _ = librosa.load(path=path, sr=sr, mono=True)
    best = None
    for start in range(0, max(len(audio), 1), piece):
        x = librosa.util.fix_length(data=audio[start:start + piece], size=piece, axis=0)
        with torch.no_grad():
            p = model(input=torch.Tensor(x).unsqueeze(0).to(device))["clipwise_output"][0].cpu().numpy()
        best = p if best is None else np.maximum(best, p)
    return best


def show(label, probs, names):
    top = np.argsort(probs)[::-1][:5]
    print(f"\n{label}")
    for i in top:
        print(f"   {probs[i]:.3f}  {names[i]} ({i})")
    print("   watched: " + ", ".join(f"{n} {probs[i]:.3f}" for i, n in WATCH.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="*", default=[])
    p.add_argument("--pkls", nargs="*", default=[])
    a = p.parse_args()

    names = class_names()
    if a.files:
        cfg = parse_yaml(DEFAULT_CONFIG)
        sr = cfg["data"]["sample_rate"]
        piece = int(sr * cfg["data"]["segment_seconds"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_pretrained_panns(model_type="Cnn14",
                                      checkpoint_path=get_path(panns_paths_dict["Cnn14"]),
                                      freeze=True).to(device)
        model.eval()
        for f in a.files:
            show(f"FILE {f}  (highest score in any 2 s piece)", file_probs(model, f, sr, piece, device), names)

    for k in a.pkls:
        v = np.asarray(pickle.load(open(k, "rb")), dtype=float).ravel()
        if v.size == 527:
            show(f"QUERY DESCRIPTION {k}", v, names)
        else:
            print(f"\nQUERY DESCRIPTION {k}: {v.size} numbers, not class scores (embedding model), skipped")


if __name__ == "__main__":
    main()