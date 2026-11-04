"""Export the FP32 SED (Cnn14) to ONNX on the Pi itself, so the 348 MB file never has to be copied.

Same model and weights as benchmark_pi.py --stage sed (panns_inference Cnn14, Cnn14_mAP=0.431.pth).
Outputs both scores the deployment needs, exactly as experiments/detect_baseline.cnn14_clip_and_frames:
the clip score (detection) and the frame readout (event time). On the server, the same export matched
PyTorch to 1.6e-6; this repeats the check here, on this Pi's own PyTorch:
  - windows of the benchmark clip, plus noise at several loudness levels (quiet to loud)
  - the clip score must equal model(x)["clipwise_output"] (the frame readout adds nothing to detection)
  - pass: max abs difference < 1e-4 for clip and frame scores
The model is forced back into eval mode after export (the exporter restores the wrapper's mode,
which would otherwise leave Cnn14 in training mode and break the reference).

Writes cnn14_fp32_pi.onnx and cnn14_fp32_pi.json (sha256, versions, check). Run once, in the venv:
  python export_sed_onnx_pi.py
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

SEG = 64000
PASS = 1e-4


def clip_and_frames(model, x):
    """Copy of experiments/detect_baseline.cnn14_clip_and_frames (the Pi need not have the experiments
    folder). KEEP IN SYNC with that function: the server scores and the Pi timings must use the same readout."""
    h = model.logmel_extractor(model.spectrogram_extractor(x))
    h = model.bn0(h.transpose(1, 3)).transpose(1, 3)
    blocks = [model.conv_block1, model.conv_block2, model.conv_block3,
              model.conv_block4, model.conv_block5, model.conv_block6]
    for i, block in enumerate(blocks):
        h = block(h, pool_size=(2, 2) if i < 5 else (1, 1), pool_type="avg")
    h = torch.mean(h, dim=3)
    pooled = torch.max(h, dim=2)[0] + torch.mean(h, dim=2)
    clipwise = torch.sigmoid(model.fc_audioset(F.relu(model.fc1(pooled))))
    frames = torch.sigmoid(model.fc_audioset(F.relu(model.fc1(2 * h.transpose(1, 2)))))
    return clipwise, frames


class SED(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, x):
        return clip_and_frames(self.model, x)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sed_weights", default=str(Path.home() / ".cache/panns/Cnn14_mAP=0.431.pth"))
    ap.add_argument("--audio", default="bench_clip_32k_mono.wav")
    ap.add_argument("--out", default="cnn14_fp32_pi.onnx")
    a = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from panns_inference.models import Cnn14

    model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                  classes_num=527)
    model.load_state_dict(torch.load(a.sed_weights, map_location="cpu", weights_only=True)["model"])
    model.eval()

    audio, sr = sf.read(a.audio, dtype="float32")
    if sr != 32000 or audio.ndim != 1:
        sys.exit(f"{a.audio}: expected 32 kHz mono")
    wins = [audio[p:p + SEG] for p in range(0, len(audio) - SEG + 1, SEG)][:5]
    rng = np.random.default_rng(0)
    wins += [(rng.standard_normal(SEG) * lvl).astype(np.float32) for lvl in (0.001, 0.01, 0.1, 0.3)]

    torch.onnx.export(SED(model).eval(), (torch.zeros(1, SEG),), a.out, opset_version=17, input_names=["audio"],
                      output_names=["clipwise", "frames"], do_constant_folding=True)
    model.eval()
    assert not model.training, "Cnn14 must be in eval mode for the reference scores"
    onnx.checker.check_model(a.out)
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])

    worst_c = worst_f = worst_ref = 0.0
    with torch.no_grad():
        for w in wins:
            xt = torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0)
            rc, rf = clip_and_frames(model, xt)
            worst_ref = max(worst_ref, float((rc - model(xt)["clipwise_output"]).abs().max()))
            c, f = sess.run(None, {"audio": w[None, :]})
            worst_c = max(worst_c, float(np.abs(c - rc.numpy()).max()))
            worst_f = max(worst_f, float(np.abs(f - rf.numpy()).max()))
    ok = worst_c < PASS and worst_f < PASS and worst_ref < 1e-5
    print(f"{len(wins)} windows (benchmark clip + noise from -60 to -10 dBFS):")
    print(f"   clip readout vs Cnn14's own clipwise_output: {worst_ref:.1e}")
    print(f"   ONNX vs PyTorch: clip {worst_c:.1e}, frames {worst_f:.1e}  -> {'OK' if ok else 'FAIL'}")
    report = {"onnx": a.out, "onnx_sha256": sha256(a.out), "weights": a.sed_weights, "passed": ok,
              "max_abs_diff": {"clip": worst_c, "frames": worst_f, "readout_vs_model": worst_ref},
              "outputs": ["clipwise (1, 527)", "frames (1, F, 527)"],
              "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__}}
    Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"   {a.out} {Path(a.out).stat().st_size / 1e6:.0f} MB, sha256 {report['onnx_sha256'][:16]}...")
    if not ok:
        Path(a.out).unlink()
        sys.exit("Export check failed; the ONNX file was removed")


if __name__ == "__main__":
    main()