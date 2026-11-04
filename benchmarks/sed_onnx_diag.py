"""Find where the Cnn14 ONNX export stops matching PyTorch (sed_int8.py step 1 failed).

Exports the SED in three stages and compares each with PyTorch on real 2 s windows:
  A. log-mel front end only (STFT + mel + log)
  B. A + bn0 + the six conv blocks (features before pooling)
  C. the full model (clip score and frame scores), as sed_int8.py does
Each stage is traced twice: with silence (as sed_int8.py did) and with a real window. If the result
depends on the tracing input, something in the model was frozen as a constant during export.
Also prints the log-mel settings (top_db, amin, ref) that could cause that.

Usage (inside the container, from the uss folder):
  python benchmarks/sed_onnx_diag.py
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
    from uss.config import panns_paths_dict
    from uss.utils import get_path, load_pretrained_panns

    model = load_pretrained_panns(model_type="Cnn14", checkpoint_path=get_path(panns_paths_dict["Cnn14"]),
                                  freeze=True).to("cpu").eval()
    lm = model.logmel_extractor
    print(f"Model class: {type(model).__module__}.{type(model).__name__}; training mode: {model.training}")
    print(f"log-mel settings: top_db={getattr(lm, 'top_db', 'n/a')}, amin={getattr(lm, 'amin', 'n/a')}, "
          f"ref={getattr(lm, 'ref', 'n/a')}, is_log={getattr(lm, 'is_log', 'n/a')}")

    class FrontEnd(nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s, x):
            return s.m.logmel_extractor(s.m.spectrogram_extractor(x))

    class Features(nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s, x):
            m = s.m
            h = m.logmel_extractor(m.spectrogram_extractor(x))
            h = m.bn0(h.transpose(1, 3)).transpose(1, 3)
            for i, b in enumerate([m.conv_block1, m.conv_block2, m.conv_block3,
                                   m.conv_block4, m.conv_block5, m.conv_block6]):
                h = b(h, pool_size=(2, 2) if i < 5 else (1, 1), pool_type="avg")
            return h

    class Full(nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s, x):
            return cnn14_clip_and_frames(s.m, x)

    with open(Path(TA) / "testsets/gate1_v2/manifest.csv") as fh:
        clips = [r["clip"] for r in csv.DictReader(fh)][::44][:5]
    wins = []
    for c in clips:
        audio, _ = librosa.load(path=c, sr=SR, mono=True)
        wins.append(librosa.util.fix_length(data=audio[:SEG], size=SEG, axis=0).astype(np.float32))
    tracers = {"silence": np.zeros(SEG, np.float32), "real window": wins[0]}

    print(f"\nCompared on {len(wins)} real windows; values are max abs difference ONNX vs PyTorch")
    print(f"{'stage':28}{'traced with silence':>22}{'traced with real audio':>26}")
    for name, Mod in [("A. log-mel front end", FrontEnd), ("B. + bn0 + conv blocks", Features),
                      ("C. full model", Full)]:
        cells = []
        for tname, tx in tracers.items():
            path = f"/tmp/diag_{Mod.__name__}.onnx"
            torch.onnx.export(Mod(model).eval(), (torch.from_numpy(tx).unsqueeze(0),), path, opset_version=17,
                              input_names=["audio"], do_constant_folding=True)
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            worst = 0.0
            with torch.no_grad():
                for w in wins:
                    ref = Mod(model)(torch.from_numpy(w).unsqueeze(0))
                    ref = ref if isinstance(ref, tuple) else (ref,)
                    got = sess.run(None, {"audio": w[None, :]})
                    worst = max(worst, max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref)))
            cells.append(worst)
        print(f"{name:28}{cells[0]:>22.2e}{cells[1]:>26.2e}")
    print("\nRead: the first stage with a large value is where the export goes wrong. If 'silence' is large but")
    print("'real audio' is small, a value was frozen at trace time (e.g. a top_db clamp computed with .item()).")


if __name__ == "__main__":
    main()