"""Week 7: INT8 version of the SED (Cnn14), to close the P1 gap (stage-1 alert ~3.2 s vs 3 s target).

Steps (server, inside the container):
  1. Export Cnn14 to ONNX with BOTH outputs Gate 1 uses: the clip score (detection) and the frame
     readout (event time), via experiments/detect_baseline.cnn14_clip_and_frames. Check ONNX FP32
     against PyTorch on real 2 s windows.
  2. Static INT8 (QDQ, per-channel) with ONNX Runtime. Converted by default: convolutions only.
     Kept FP32: the STFT and log-mel front end (exact transform tables), conv_block1.conv1 (the first
     convolution, fed by bn0's output; kept FP32 as the first layer, like the separator's pre_conv),
     batch norm, and the two fully connected layers (fc1, fc_audioset; tiny compute). Checked afterwards
     that kept layers did stay FP32. Activations are asymmetric INT8 (ActivationSymmetric not set);
     say so if the Pi numbers are compared with a symmetric s8s8 build.
     Calibration: windows of a fixed subset of dev clips (listed in the report), on the mixture AND on
     the saved USS outputs, since the SED scores both in every pipeline.
  3. Score every Gate 1 piece with the INT8 SED, on the original audio and on the saved Gate 1 USS
     outputs (FP32 separator, unchanged, so the comparison isolates the SED), into
     results/<set>_sedint8_scores_pieces.csv in run_gate1.py's format. Then:
       python experiments/o1_frozen.py --sed_int8 --int8_pieces results/gate1_v2_sedint8_scores_pieces.csv \\
           --int8_report cnn14_int8.json
     scores SED-only, always-on and the lead gate at the FP32 thresholds, frozen (same rule as O1).

Pi timing is separate (benchmark_pi.py --stage sed --backend onnx).

Usage (inside the container, from the uss folder):
  python benchmarks/sed_int8.py
  python benchmarks/sed_int8.py --reuse          # skip export/quantize if the .onnx files exist
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO))
TA = "/home/zmak002/docker_share/test_audio"
SR, SEG = 32000, 64000
HEAVY = ["Conv", "ConvTranspose", "MatMul", "Gemm"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pieces_of(audio):
    import librosa
    for p in range(0, len(audio), SEG):
        yield librosa.util.fix_length(data=audio[p: p + SEG], size=SEG, axis=0).astype(np.float32)


def calib_clips(rows, sounds):
    """First clip of every (sound, condition) plus the first 2 target-free clips with an interferer."""
    taken, out = {}, []
    for r in rows:
        if r["contains"] in sounds:
            key, limit = (r["contains"], r["condition"]), 1
        elif r["condition"] == "int":
            key, limit = ("none", "int"), 2
        else:
            continue
        if taken.get(key, 0) < limit:
            taken[key] = taken.get(key, 0) + 1
            out.append(r)
    return out


def keep_names(nodes, keep):
    return [n.name for n in nodes if any(f"/{k}/" in n.name for k in keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--fp32", default="cnn14_fp32.onnx")
    ap.add_argument("--out", default="cnn14_int8.onnx")
    ap.add_argument("--keep_fp32", nargs="*",
                    default=["spectrogram_extractor", "logmel_extractor", "conv_block1/conv1"])
    ap.add_argument("--ops", nargs="+", default=["Conv"], choices=HEAVY)
    ap.add_argument("--calibration", choices=["minmax", "percentile"], default="minmax")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()

    import librosa
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    from detect_baseline import check_frame_readout, cnn14_clip_and_frames
    from run_gate1 import SOUNDS, uss_output
    from uss.config import panns_paths_dict
    from uss.utils import get_path, load_pretrained_panns

    ts = Path(a.test_set)
    ref_csv = Path(f"results/{ts.name}_scores_pieces.csv")
    if not ref_csv.exists():
        sys.exit(f"Missing {ref_csv} (Gate 1 FP32 pieces), needed for the comparison in step 3")
    with open(ts / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    calib = calib_clips(rows, SOUNDS)
    calib_names = {Path(r["clip"]).name for r in calib}
    fp32_path, out = Path(a.fp32), Path(a.out)
    torch.set_num_threads(a.threads)

    model = load_pretrained_panns(model_type="Cnn14", checkpoint_path=get_path(panns_paths_dict["Cnn14"]),
                                  freeze=True).to("cpu").eval()
    check_frame_readout(model, SEG, "cpu")

    class SED(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, x):
            return cnn14_clip_and_frames(self.model, x)

    def load(path):
        audio, _ = librosa.load(path=str(path), sr=SR, mono=True)
        return audio

    def all_audio(r):
        """The mixture plus the three saved Gate 1 USS outputs for one clip."""
        yield "mix", load(r["clip"])
        for s, (_, qdir) in SOUNDS.items():
            yield s, load(uss_output(ts, r["clip"], qdir))

    # ---------- 1. export ----------
    if not (a.reuse and fp32_path.exists()):
        x0 = torch.zeros(1, SEG)
        torch.onnx.export(SED(model).eval(), (x0,), str(fp32_path), opset_version=17, input_names=["audio"],
                          output_names=["clipwise", "frames"], do_constant_folding=True)
        onnx.checker.check_model(str(fp32_path))
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    model.eval()                                   # export restores the wrapper's mode; force inference mode
    assert not model.training, "Cnn14 must be in eval mode for the reference scores"
    sess32 = ort.InferenceSession(str(fp32_path), so, providers=["CPUExecutionProvider"])
    check_rows = [r for r in rows if Path(r["clip"]).name not in calib_names][::20]
    worst_c = worst_f = 0.0
    with torch.no_grad():
        for r in check_rows:
            for _, audio in all_audio(r):
                for w in pieces_of(audio):
                    ref_c, ref_f = cnn14_clip_and_frames(model, torch.from_numpy(w).unsqueeze(0))
                    c, f = sess32.run(None, {"audio": w[None, :]})
                    worst_c = max(worst_c, float(np.abs(c - ref_c.numpy()).max()))
                    worst_f = max(worst_f, float(np.abs(f - ref_f.numpy()).max()))
    ok1 = worst_c < 1e-4 and worst_f < 1e-4
    print(f"1. ONNX FP32 vs PyTorch on {len(check_rows)} clips x 4 audio x 5 windows: max abs difference "
          f"clip {worst_c:.1e}, frames {worst_f:.1e} ({'OK' if ok1 else 'FAIL'}); {fp32_path} "
          f"{fp32_path.stat().st_size / 1e6:.0f} MB")
    if not ok1:
        sys.exit(1)

    # ---------- 2. INT8 ----------
    if not (a.reuse and out.exists()):
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process

        class Reader:
            def __init__(self):
                self.items = [w for r in calib for _, audio in all_audio(r) for w in pieces_of(audio)]
                self.i = 0

            def get_next(self):
                if self.i >= len(self.items):
                    return None
                self.i += 1
                return {"audio": self.items[self.i - 1][None, :]}

        pre = out.with_name(out.stem + "_pre.onnx")
        quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
        nodes = onnx.load(str(pre)).graph.node
        keep = keep_names(nodes, a.keep_fp32)
        for k in a.keep_fp32:
            if not any(f"/{k}/" in n for n in keep):
                sys.exit(f"--keep_fp32 {k}: no node with '/{k}/' in its name")
        reader = Reader()
        print(f"2. Calibrating on {len(calib)} clips x 4 audio = {len(reader.items)} windows ...")
        t0 = time.perf_counter()
        quantize_static(str(pre), str(out), reader, quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, nodes_to_exclude=keep,
                        op_types_to_quantize=a.ops,
                        calibrate_method={"minmax": CalibrationMethod.MinMax,
                                          "percentile": CalibrationMethod.Percentile}[a.calibration],
                        extra_options={"CalibMaxIntermediateOutputs": 2})
        pre.unlink(missing_ok=True)
        print(f"   quantized in {(time.perf_counter() - t0) / 60:.1f} min")
        out.with_suffix(".quant.json").write_text(json.dumps({
            "ops": a.ops, "keep_fp32": a.keep_fp32, "calibration": a.calibration,
            "calibration_clips": sorted(calib_names), "test_set": str(ts),
            "fp32_sha256": sha256(fp32_path)}, indent=2))
        reused = False
    else:
        reused = True
    side = out.with_suffix(".quant.json")
    if not side.exists():
        sys.exit(f"{side} missing: cannot tell how {out} was built; re-run without --reuse")
    built = json.loads(side.read_text())          # what the model was ACTUALLY built with
    if built.get("fp32_sha256") != sha256(fp32_path):
        sys.exit(f"{out} was not quantized from the current {fp32_path} (or {side} predates the check); "
                 f"re-run without --reuse")
    if built["test_set"] != str(ts):
        sys.exit(f"{out} was calibrated on {built['test_set']}, not {ts}; re-run without --reuse")
    calib_names = set(built["calibration_clips"])
    g = onnx.load(str(out)).graph
    made = {o: n.op_type for n in g.node for o in n.output}
    kept = keep_names(g.node, built["keep_fp32"])
    counts, leaked = Counter(), []
    for n in g.node:
        if n.op_type in HEAVY and len(n.input) > 1:
            q = made.get(n.input[1]) == "DequantizeLinear"
            counts[f"{n.op_type} {'int8 weights' if q else 'fp32'}"] += 1
            if q and n.name in kept:
                leaked.append(n.name)
    if leaked:
        sys.exit(f"Layers meant to stay FP32 have INT8 weights: {leaked[:3]}")
    int8_sha = sha256(out)
    print(f"   {out} {out.stat().st_size / 1e6:.0f} MB, sha256 {int8_sha}")
    print("   heavy layers: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + f"; kept FP32 as intended: {', '.join(built['keep_fp32'])}" + (" (reused model)" if reused else ""))

    # ---------- 3. score every Gate 1 piece ----------
    sess8 = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    ids = {s: SOUNDS[s][0] for s in SOUNDS}
    pieces, diffs = [], []
    ref = {}
    with open(ref_csv) as fh:
        for r in csv.DictReader(fh):
            ref[(r["sound"], r["clip"], r["version"], int(r["piece"]))] = float(r["score"])
    t0 = time.perf_counter()
    print(f"3. Scoring {len(rows)} clips (mixture + 3 USS outputs) with the INT8 SED ...")
    for i, r in enumerate(rows, 1):
        name = Path(r["clip"]).name
        for which, audio in all_audio(r):
            scored = [sess8.run(None, {"audio": w[None, :]}) for w in pieces_of(audio)]
            for s, cid in ids.items():
                if which not in ("mix", s):
                    continue
                version = "before" if which == "mix" else "after"
                for k, (c, f) in enumerate(scored):
                    sc = float(c[0, cid])
                    pieces.append({"sound": s, "clip": name, "version": version, "piece": k,
                                   "start_s": round(k * SEG / SR, 3), "score": round(sc, 5),
                                   "frames": " ".join(f"{v:.4f}" for v in f[0, :, cid])})
                    key = (s, name, version, k)
                    if key in ref and name not in calib_names:
                        diffs.append(sc - ref[key])
        if i % 20 == 0:
            print(f"   {i}/{len(rows)} clips")
    path = Path(f"results/{ts.name}_sedint8_scores_pieces.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pieces[0].keys()))
        w.writeheader()
        w.writerows(pieces)
    d = np.abs(np.array(diffs))
    if not len(d):
        sys.exit("No scores matched the Gate 1 reference; check the test set and the reference CSV")
    print(f"   piece scores vs Gate 1 FP32 (clips not used for calibration, {len(d)} scores): median abs "
          f"difference {np.median(d):.4f}, 99th percentile {np.percentile(d, 99):.4f}, max {d.max():.4f}")
    print(f"   server CPU scoring time (NOT a Pi result): {time.perf_counter() - t0:.0f} s")
    report = {"stage": "sed", "model": "Cnn14", "sed": "ONNX Runtime INT8",
              "uss": "FP32 (saved Gate 1 outputs, unchanged)", "test_set": str(ts), "reused": reused,
              "fp32": str(fp32_path), "int8": str(out), "int8_sha256": int8_sha, "fp32_sha256": sha256(fp32_path),
              "ops": built["ops"], "keep_fp32": built["keep_fp32"], "calibration": built["calibration"],
              "calibration_clips": sorted(calib_names), "heavy_layers": dict(counts),
              "export_check_max_abs": {"clip": worst_c, "frames": worst_f},
              "piece_abs_diff": {"median": float(np.median(d)), "p99": float(np.percentile(d, 99)),
                                 "max": float(d.max())},
              "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__}}
    Path(out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"   wrote {path} and {Path(out).with_suffix('.json')}")
    print(f"   next: python experiments/o1_frozen.py --sed_int8 --int8_pieces {path} "
          f"--int8_report {Path(out).with_suffix('.json')}")


if __name__ == "__main__":
    main()