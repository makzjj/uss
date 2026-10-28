"""Compare USS output with the target in two extra ways.

1. Spectrogram match: compares WHICH frequencies are present and HOW LOUD, over time,
   ignoring the fine wave details. Closer to what your ear hears.
2. Score per 1-second piece: shows whether the whole clip is bad, or only some parts.

Usage:
  python experiments/compare_sound_content.py --target <target.wav> --estimate <USS output.wav>
"""
import argparse
import json

import librosa
import numpy as np

SR = 32000


def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)


def si_sdr(est, ref, eps=1e-12):
    ref = ref - ref.mean()
    est = est - est.mean()
    proj = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    return 10 * np.log10((np.sum(proj ** 2) + eps) / (np.sum((est - proj) ** 2) + eps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--estimate", required=True)
    a = p.parse_args()

    tgt, est = load(a.target), load(a.estimate)
    n = min(len(tgt), len(est))
    tgt, est = tgt[:n], est[:n]

    # 1. Spectrogram match (magnitudes only, phase ignored)
    T = np.abs(librosa.stft(tgt, n_fft=1024, hop_length=320))
    E = np.abs(librosa.stft(est, n_fft=1024, hop_length=320))
    corr = float(np.corrcoef(T.ravel(), E.ravel())[0, 1])
    E_scaled = E * (np.sum(T * E) / (np.sum(E * E) + 1e-12))
    spec_snr = 10 * np.log10(np.sum(T ** 2) / (np.sum((T - E_scaled) ** 2) + 1e-12))

    # 2. Waveform score per 1-second piece
    per_second = [round(float(si_sdr(est[i:i + SR], tgt[i:i + SR])), 1)
                  for i in range(0, n - SR + 1, SR)]

    print(json.dumps({
        "spectrogram_similarity_0_to_1": round(corr, 3),
        "spectrogram_match_db": round(float(spec_snr), 2),
        "waveform_si_sdr_per_second_db": per_second,
    }, indent=2))


if __name__ == "__main__":
    main()