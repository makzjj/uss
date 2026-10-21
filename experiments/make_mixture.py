"""Build one controlled mixture (target + background at a known SNR) for USS verification.

Outputs (in --out_dir):
    mixture.wav     what you feed to USS
    target.wav      ground-truth target, with exactly the gain it has inside the mixture
    background.wav  ground-truth background, with exactly the gain it has inside the mixture
    meta.json       inputs, SNR requested vs measured, gains applied

Invariant: mixture == target + background (sample for sample), so target.wav is a
valid reference for SI-SDR against the USS output.
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 32000  # USS operates at 32 kHz


def load_fixed(path, duration, offset):
    """Load mono at SR, cut or zero-pad to exactly `duration` seconds."""
    y, _ = librosa.load(path, sr=SR, mono=True, offset=offset, duration=duration)
    n = int(round(duration * SR))
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n].astype(np.float64)


def power(x):
    return np.mean(x ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--background", required=True)
    p.add_argument("--snr", type=float, default=0.0, help="target-to-background SNR in dB")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--target_offset", type=float, default=0.0, help="start time in target file (s)")
    p.add_argument("--bg_offset", type=float, default=0.0, help="start time in background file (s)")
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()

    tgt = load_fixed(a.target, a.duration, a.target_offset)
    bg = load_fixed(a.background, a.duration, a.bg_offset)

    if power(tgt) == 0 or power(bg) == 0:
        raise SystemExit("Target or background is silent in the chosen window.")

    # Scale the background so that 10*log10(P_tgt / P_bg) == snr
    bg_gain = np.sqrt(power(tgt) / (power(bg) * 10 ** (a.snr / 10)))
    bg = bg * bg_gain

    mix = tgt + bg

    # Avoid clipping when writing to wav: apply ONE gain to all three signals,
    # so the references stay consistent with the mixture.
    peak = np.max(np.abs(mix))
    norm_gain = 0.9 / peak if peak > 0.9 else 1.0
    tgt, bg, mix = tgt * norm_gain, bg * norm_gain, mix * norm_gain

    measured_snr = 10 * np.log10(power(tgt) / power(bg))

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "mixture.wav", mix.astype(np.float32), SR)
    sf.write(out / "target.wav", tgt.astype(np.float32), SR)
    sf.write(out / "background.wav", bg.astype(np.float32), SR)

    meta = {
        "target_file": str(a.target),
        "background_file": str(a.background),
        "target_offset_s": a.target_offset,
        "bg_offset_s": a.bg_offset,
        "duration_s": a.duration,
        "sample_rate": SR,
        "snr_requested_db": a.snr,
        "snr_measured_db": round(float(measured_snr), 3),
        "bg_gain": float(bg_gain),
        "peak_norm_gain": float(norm_gain),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()