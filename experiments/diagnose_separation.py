"""Work out why a USS output scored badly.
 
Checks:
  1. Is the output closer to the jackhammer or to the background?
  2. Is the output shifted in time compared with the target?
  3. How loud is the output compared with the target?
 
Usage:
  python experiments/diagnose_separation.py --mix_dir <mix_dir> --estimate <USS output wav>
"""
import argparse
import json
from pathlib import Path
 
import librosa
import numpy as np
from scipy.signal import correlate
 
SR = 32000
 
 
def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)
 
 
def si_sdr(est, ref, eps=1e-12):
    ref = ref - ref.mean()
    est = est - est.mean()
    proj = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    return 10 * np.log10((np.sum(proj ** 2) + eps) / (np.sum((est - proj) ** 2) + eps))
 
 
def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mix_dir", required=True)
    p.add_argument("--estimate", required=True)
    a = p.parse_args()
 
    d = Path(a.mix_dir)
    tgt, bg, est = load(d / "target.wav"), load(d / "background.wav"), load(a.estimate)
    n = min(len(tgt), len(bg), len(est))
    tgt, bg, est = tgt[:n], bg[:n], est[:n]
 
    # Time shift: find the lag where output and target line up best (search +/- 0.5 s)
    max_lag = SR // 2
    corr = correlate(est, tgt, mode="full", method="fft")
    mid = len(tgt) - 1
    window = corr[mid - max_lag: mid + max_lag + 1]
    best_lag = int(np.argmax(np.abs(window)) - max_lag)
 
    if best_lag > 0:
        est_aligned, tgt_aligned = est[best_lag:], tgt[:n - best_lag]
    elif best_lag < 0:
        est_aligned, tgt_aligned = est[:n + best_lag], tgt[-best_lag:]
    else:
        est_aligned, tgt_aligned = est, tgt
 
    result = {
        "1_si_sdr_vs_jackhammer_db": round(float(si_sdr(est, tgt)), 2),
        "1_si_sdr_vs_background_db": round(float(si_sdr(est, bg)), 2),
        "2_best_time_shift_samples": best_lag,
        "2_best_time_shift_ms": round(best_lag / SR * 1000, 1),
        "2_si_sdr_vs_jackhammer_after_shift_db": round(float(si_sdr(est_aligned, tgt_aligned)), 2),
        "3_output_loudness_db": round(float(rms_db(est)), 1),
        "3_target_loudness_db": round(float(rms_db(tgt)), 1),
    }
    print(json.dumps(result, indent=2))
 
 
if __name__ == "__main__":
    main()
 