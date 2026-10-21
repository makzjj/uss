"""Score one USS separation against the clean answer key.
 
Prints:
  SI-SDR of the mixture   (how close the raw mixture already is to the target)
  SI-SDR of the USS output (how close the separated audio is to the target)
  SI-SDRi = output - mixture (how much USS helped; positive = it helped)
 
Usage:
  python experiments/score_separation.py \
    --target   <mix_dir>/target.wav \
    --mixture  <mix_dir>/mixture.wav \
    --estimate <path to USS output wav>
"""
import argparse
import json
 
import librosa
import numpy as np
 
SR = 32000
 
 
def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)
 
 
def si_sdr(estimate, reference, eps=1e-12):
    """Scale-invariant SDR in dB. Higher = closer to the reference."""
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    projection = scale * reference
    noise = estimate - projection
    return 10 * np.log10((np.sum(projection ** 2) + eps) / (np.sum(noise ** 2) + eps))
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--mixture", required=True)
    p.add_argument("--estimate", required=True)
    a = p.parse_args()
 
    tgt, mix, est = load(a.target), load(a.mixture), load(a.estimate)
 
    # Trim all three to the shortest length so they line up sample by sample
    n = min(len(tgt), len(mix), len(est))
    lengths = {"target": len(tgt), "mixture": len(mix), "estimate": len(est)}
    tgt, mix, est = tgt[:n], mix[:n], est[:n]
 
    mix_score = si_sdr(mix, tgt)
    est_score = si_sdr(est, tgt)
 
    result = {
        "lengths_samples": lengths,
        "si_sdr_mixture_db": round(float(mix_score), 2),
        "si_sdr_uss_output_db": round(float(est_score), 2),
        "si_sdr_improvement_db": round(float(est_score - mix_score), 2),
    }
    print(json.dumps(result, indent=2))
 
    if len(set(lengths.values())) > 1:
        print("Note: files had different lengths and were trimmed to match. "
              "A small difference is normal; a big one means the wrong file may be compared.")
 
 
if __name__ == "__main__":
    main()