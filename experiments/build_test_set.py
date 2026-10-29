"""Build the Gate 1 pilot test set in one go.

Makes:
  <out_dir>/clips/<sound>_bg<k>_snr<s>.wav   mixtures WITH a sound
  <out_dir>/clips/none_bg<k>.wav             backgrounds alone (WITHOUT any sound)
  <out_dir>/refs/<clip name>_target.wav      clean answer keys (for separation scores later)
  <out_dir>/manifest.csv                     what is in every clip

Every clip is set to the same overall loudness (-25 dBFS), so the listener is not
influenced by some clips simply being louder. The same gain is applied to the
answer keys, so mixture = target + background still holds.

Usage (inside the container, from the uss folder):
  python experiments/build_test_set.py
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 32000
TA = "/home/zmak002/docker_share/test_audio"
DEFAULT_TARGETS = {
    "beeper": f"{TA}/beeper.wav",
    "engine": f"{TA}/engine3.wav",
    "jackhammer": f"{TA}/mix_snr0/target.wav",
}
LEVEL_DBFS = -25.0


def load_fixed(path, duration):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=duration)
    n = int(round(duration * SR))
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n].astype(np.float64)


def power(x):
    return float(np.mean(x ** 2))


def level_gain(x):
    """Gain that sets x to LEVEL_DBFS, reduced if that would clip."""
    g = 10 ** (LEVEL_DBFS / 20) / (np.sqrt(power(x)) + 1e-12)
    peak = np.max(np.abs(x * g))
    if peak > 0.99:
        g *= 0.99 / peak
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backgrounds", nargs="+",
                   default=[f"{TA}/bg{k}.flac" for k in range(1, 6)])
    p.add_argument("--snrs", nargs="+", type=float, default=[0.0, -5.0])
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--out_dir", default=f"{TA}/gate1_test_set")
    a = p.parse_args()

    out = Path(a.out_dir)
    # Old USS outputs belong to the old clips; remove them so they are never reused by mistake
    if (out / "uss_outputs").exists():
        shutil.rmtree(out / "uss_outputs")
        print("Removed old USS outputs from a previous test set")
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "refs").mkdir(parents=True, exist_ok=True)

    targets = {name: load_fixed(path, a.duration) for name, path in DEFAULT_TARGETS.items()}
    bgs = [load_fixed(path, a.duration) for path in a.backgrounds]
    for name, t in targets.items():
        if power(t) == 0:
            raise SystemExit(f"Target '{name}' is silent in the first {a.duration} s")
    for path, b in zip(a.backgrounds, bgs):
        if power(b) == 0:
            raise SystemExit(f"Background is silent: {path}")

    rows = []

    # WITH a sound
    for name, tgt in targets.items():
        for k, bg in enumerate(bgs, start=1):
            for snr in a.snrs:
                bg_s = bg * np.sqrt(power(tgt) / (power(bg) * 10 ** (snr / 10)))
                mix = tgt + bg_s
                g = level_gain(mix)
                clip = f"{name}_bg{k}_snr{snr:g}"
                sf.write(out / "clips" / f"{clip}.wav", (mix * g).astype(np.float32), SR)
                sf.write(out / "refs" / f"{clip}_target.wav", (tgt * g).astype(np.float32), SR)
                sf.write(out / "refs" / f"{clip}_background.wav", (bg_s * g).astype(np.float32), SR)
                rows.append({"clip": str(out / "clips" / f"{clip}.wav"), "contains": name,
                             "background": a.backgrounds[k - 1], "snr_db": snr})

    # WITHOUT any sound: backgrounds alone
    for k, bg in enumerate(bgs, start=1):
        clip = f"none_bg{k}"
        sf.write(out / "clips" / f"{clip}.wav", (bg * level_gain(bg)).astype(np.float32), SR)
        rows.append({"clip": str(out / "clips" / f"{clip}.wav"), "contains": "none",
                     "background": a.backgrounds[k - 1], "snr_db": ""})

    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip", "contains", "background", "snr_db"])
        w.writeheader()
        w.writerows(rows)
    (out / "settings.json").write_text(json.dumps({
        "targets": DEFAULT_TARGETS, "backgrounds": a.backgrounds, "snrs_db": a.snrs,
        "duration_s": a.duration, "sample_rate": SR, "level_dbfs": LEVEL_DBFS}, indent=2))

    n_with = sum(r["contains"] != "none" for r in rows)
    print(f"Made {len(rows)} clips: {n_with} with a sound, {len(rows) - n_with} backgrounds alone")
    print(f"Saved in {out}  (list of clips: manifest.csv)")


if __name__ == "__main__":
    main()