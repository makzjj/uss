"""sed_int8.py step 1 failed, but sed_onnx_diag.py found the export itself exact on original clips.
This repeats step 1's exact check (same clips: mixtures AND saved USS outputs, every 2 s window) and
lists the worst windows with their loudness, to see whether the mismatch comes from near-silent
audio (log of tiny values) or from something else.

Usage (inside the container, from the uss folder):
  python benchmarks/sed_onnx_diag2.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO))
TA = "/home/zmak002/docker_share/test_audio"
SR, SEG = 32000, 64000


def main():
    import librosa
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    from detect_baseline import cnn14_clip_and_frames
    from run_gate1 import SOUNDS, uss_output
    from uss.config import panns_paths_dict
    from uss.utils import get_path, load_pretrained_panns

    sys.path.insert(0, str(REPO / "benchmarks"))
    from sed_int8 import calib_clips, pieces_of

    ts = Path(TA) / "testsets/gate1_v2"
    with open(ts / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    calib_names = {Path(r["clip"]).name for r in calib_clips(rows, SOUNDS)}
    check_rows = [r for r in rows if Path(r["clip"]).name not in calib_names][::20]

    model = load_pretrained_panns(model_type="Cnn14", checkpoint_path=get_path(panns_paths_dict["Cnn14"]),
                                  freeze=True).to("cpu").eval()

    class SED(nn.Module):
        def __init__(s, m):
            super().__init__()
            s.model = m

        def forward(s, x):
            return cnn14_clip_and_frames(s.model, x)

    path = "/tmp/diag2_cnn14.onnx"
    torch.onnx.export(SED(model), (torch.zeros(1, SEG),), path, opset_version=17, input_names=["audio"],
                      output_names=["clipwise", "frames"], do_constant_folding=True)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    results = []
    with torch.no_grad():
        for r in check_rows:
            sources = [("mix", r["clip"])] + [(s, uss_output(ts, r["clip"], q)) for s, (_, q) in SOUNDS.items()]
            for which, f in sources:
                audio, _ = librosa.load(path=str(f), sr=SR, mono=True)
                for k, w in enumerate(pieces_of(audio)):
                    rc, rf = cnn14_clip_and_frames(model, torch.from_numpy(w).unsqueeze(0))
                    c, fr = sess.run(None, {"audio": w[None, :]})
                    rms_db = 20 * np.log10(np.sqrt(np.mean(w.astype(np.float64) ** 2)) + 1e-12)
                    results.append((float(np.abs(c - rc.numpy()).max()), float(np.abs(fr - rf.numpy()).max()),
                                    rms_db, Path(r["clip"]).stem, which, k, float(np.abs(w).max())))
    results.sort(reverse=True)
    print(f"{len(results)} windows ({len(check_rows)} clips x mixture + 3 USS outputs x pieces)")
    print(f"{'clip diff':>10}{'frame diff':>12}{'RMS dBFS':>10}{'peak':>9}  window")
    for cd, fd, rms, clip, which, k, peak in results[:12]:
        print(f"{cd:>10.2e}{fd:>12.2e}{rms:>10.1f}{peak:>9.1e}  {clip} [{which}] piece {k}")
    bad = [x for x in results if x[0] >= 1e-4]
    good = [x for x in results if x[0] < 1e-4]
    print(f"\nwindows over 1e-4: {len(bad)} of {len(results)}")
    if bad:
        print(f"   their loudness: RMS {min(x[2] for x in bad):.1f} to {max(x[2] for x in bad):.1f} dBFS; "
              f"sources: {sorted({x[4] for x in bad})}")
    if good:
        print(f"   windows that match: RMS {min(x[2] for x in good):.1f} to {max(x[2] for x in good):.1f} dBFS")


if __name__ == "__main__":
    main()