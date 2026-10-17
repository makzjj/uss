"""Run ONCE on the GPU server (inside the USS container).

Makes the fixed benchmark input: 10 s of a clip, resampled to 32 kHz mono,
so the Pi never has to resample (which would add cost and variation).

Usage: python make_bench_clip.py [source_audio] [offset_seconds]
"""
import sys

import librosa
import numpy as np
import soundfile as sf

SRC = sys.argv[1] if len(sys.argv) > 1 else "harry_potter.flac"
OFFSET = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
OUT = "bench_clip_32k_mono.wav"

y, _ = librosa.load(SRC, sr=32000, mono=True, offset=OFFSET, duration=10.0)
sf.write(OUT, y, 32000)

rms = float(np.sqrt(np.mean(y ** 2)))
print(f"Saved -> {OUT}: {len(y)} samples = {len(y) / 32000:.2f} s, 32 kHz mono, RMS {rms:.4f}")
if rms < 1e-3:
    print("Warning: clip is nearly silent; rerun with a later offset, e.g. 20")