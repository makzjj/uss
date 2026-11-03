"""Week 6: INT8 version of the ONNX separator, checked against Requirements v0.3.6 O1.

O1 (INT8 vs FP32): SI-SDR degradation <= 1 dB on synthetic mixtures; recall loss <= 2 pp;
target-absent false alarms still inside P2. Speedup is measured on the Pi, not here.

What INT8 means here: weights and the numbers flowing between layers are stored as 8-bit
integers instead of 32-bit floats, each with a scale factor. "Static" means the scale factors
for the flowing numbers are fixed in advance by running some real audio through the model
(calibration) and recording each layer's range.

Converted by default: the separator's convolutions (--ops Conv), which hold nearly all the
computation. Kept FP32 by default, each for a reason:
  - FiLM (Gemm): turns the query into steering values for every block; almost no compute, so
    INT8 only adds rounding to the steering
  - ConvTranspose (decoder upsampling): ONNX Runtime's CPU kernels may run it as dequantize ->
    FP32 -> quantize, i.e. rounding with no speed-up (add it with --ops if profiling says otherwise)
  - pre_conv and after_conv (--keep_fp32): the first layer sees the wide-range spectrogram and the
    last produces the mask that multiplies the audio; 8-bit rounding there is the usual cause of
    audible damage
  - element-wise steps (add, multiply, sigmoid, tanh), including mask-times-phase
Run 1 (Week 6) converted Conv, ConvTranspose and Gemm with no first/last exclusion: SI-SDR loss
0.28 dB mean, but INT8 output differed from FP32 by -13.4 dB, hence these defaults.

Kept in FP32 on purpose: the STFT and inverse STFT. They are written as convolutions holding
exact Fourier tables, so the quantizer would otherwise round them to 8 bits and damage the audio
before the separator starts. Only the separator's own layers are quantized.

Steps:
  1. Calibrate on windows from a few dev clips (a fixed subset, listed in the report), each with
     all three cached queries, since the deployed separator sees every query.
  2. Quick check on the OTHER clips (not used for calibration), FP32 ONNX vs INT8 ONNX:
       - how far INT8 output is from FP32 output (dB, as in export_onnx.py)
       - SI-SDR against the isolated Scaper target, whole 10 s clip, matching query (Gate 1 method)
       - leakage on target-absent clips (output energy vs clip, Gate 1 method)
  3. --full: separate every clip with INT8 for every query, score the pieces with Cnn14, and write
     results/<set>_int8_scores_pieces.csv in run_gate1.py's format. Then recall and false alarms
     come from experiments/o1_frozen.py, which scores INT8 at the FP32 thresholds, frozen (as a
     deployed INT8 model would run), so false alarms can actually change and be checked.

Usage (inside the container, from the uss folder):
  python benchmarks/int8_check.py                 # steps 1-2, about 15-25 min on the server CPU
  python benchmarks/int8_check.py --reuse --full  # adds step 3 (all clips; resumable)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

TA = "/home/zmak002/docker_share/test_audio"
SR, SEG = 32000, 64000
SOUNDS = ["beeper", "engine", "jackhammer"]
CLASS_IDS = {"beeper": 319, "engine": 343, "jackhammer": 420}
O1_SISDR_DB = 1.0
HEAVY_OPS = ["Conv", "ConvTranspose", "MatMul", "Gemm"]


def err_db(out, ref):
    return float(10 * np.log10(np.sum((out - ref) ** 2) / max(np.sum(ref ** 2), 1e-20) + 1e-30))


def si_sdr(estimate, reference, eps=1e-12):
    """Same formula as experiments/score_separation.py."""
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    projection = scale * reference
    noise = estimate - projection
    return float(10 * np.log10((np.sum(projection ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def windows(audio):
    """2 s windows, last one zero-padded, as uss.inference.separate_by_query_condition does."""
    n = int(np.ceil(len(audio) / SEG))
    padded = np.zeros(n * SEG, dtype=np.float32)
    padded[: len(audio)] = audio
    return padded.reshape(n, 1, 1, SEG)


def separate(sess, audio, cond):
    out = [sess.run(["waveform"], {"mixture": w, "condition": cond})[0].reshape(-1) for w in windows(audio)]
    return np.concatenate(out)[: len(audio)]


def load_clip(path):
    audio, sr = sf.read(path, dtype="float32")
    if sr != SR or audio.ndim != 1:
        sys.exit(f"{path}: expected {SR} Hz mono")
    return audio


def split_clips(rows, per_cell):
    """Calibration: the first per_cell clips of every (sound, condition) pair plus the first 2
    target-free clips with an interferer. Everything else is for checking, so calibration and
    check audio never overlap."""
    taken, calib = {}, []
    for r in rows:
        if r["contains"] in SOUNDS:
            key, limit = (r["contains"], r["condition"]), per_cell
        elif r["condition"] == "int":
            key, limit = ("none", "int"), 2
        else:
            continue
        if taken.get(key, 0) < limit:
            taken[key] = taken.get(key, 0) + 1
            calib.append(r)
    names = {r["clip"] for r in calib}
    return calib, [r for r in rows if r["clip"] not in names]


class Reader:
    """Feeds calibration windows to the ONNX Runtime quantizer, one at a time."""

    def __init__(self, clips, conds):
        self.items = [(w, c) for r in clips for w in windows(load_clip(r["clip"])) for c in conds.values()]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.items):
            return None
        w, c = self.items[self.i]
        self.i += 1
        return {"mixture": w, "condition": c}

    def rewind(self):
        self.i = 0


def keep_names(nodes, keep):
    """Nodes kept FP32: the STFT / inverse STFT plus any --keep_fp32 module names."""
    return [n.name for n in nodes
            if "stft" in n.name.lower() or any(f"/{k}/" in n.name for k in keep)]


def quantize(fp32, out, reader, method, ops, keep):
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = out.with_name(out.stem + "_pre.onnx")
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    nodes = onnx.load(str(pre)).graph.node
    keep_fp32 = keep_names(nodes, keep)
    if not any("stft" in n.lower() for n in keep_fp32):
        sys.exit("Found no STFT / inverse STFT nodes by name; check the node names before quantizing")
    for k in keep:
        if not any(f"/{k}/" in n for n in keep_fp32):
            sys.exit(f"--keep_fp32 {k}: no node with '/{k}/' in its name")
    t0 = time.perf_counter()
    quantize_static(str(pre), str(out), reader, quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    nodes_to_exclude=keep_fp32, op_types_to_quantize=ops,
                    calibrate_method={"minmax": CalibrationMethod.MinMax,
                                      "percentile": CalibrationMethod.Percentile}[method],
                    extra_options={"CalibMaxIntermediateOutputs": 2})   # bounds memory during calibration
    pre.unlink(missing_ok=True)
    return keep_fp32, time.perf_counter() - t0


def int8_layers(path, keep):
    """Count heavy layers by whether their weights come through a DequantizeLinear (INT8 weights),
    and confirm every node meant to stay FP32 really did. Returns (counts, kept FP32 names)."""
    import onnx
    g = onnx.load(str(path)).graph
    made_by = {o: n.op_type for n in g.node for o in n.output}
    counts, kept, leaked = {}, keep_names(g.node, keep), []
    for n in g.node:
        if n.op_type in HEAVY_OPS and len(n.input) > 1:
            q = made_by.get(n.input[1]) == "DequantizeLinear"
            key = f"{n.op_type} {'int8 weights' if q else 'fp32'}"
            counts[key] = counts.get(key, 0) + 1
            if q and n.name in kept:
                leaked.append(n.name)
    if leaked:
        sys.exit(f"{len(leaked)} layers meant to stay FP32 have INT8 weights, e.g. {leaked[:3]}")
    return counts, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default="resunet30_at_soft_fp32.onnx")
    ap.add_argument("--out", default="resunet30_at_soft_int8_v2.onnx",
                    help="run 1 wrote resunet30_at_soft_int8.onnx; the default here keeps it")
    ap.add_argument("--ops", nargs="+", default=["Conv"], choices=HEAVY_OPS, help="layer types to convert")
    ap.add_argument("--keep_fp32", nargs="*", default=["pre_conv", "after_conv"],
                    help="module names kept FP32 besides the STFT / inverse STFT")
    ap.add_argument("--conditions", default="results/setup/query_conditions")
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--calib_per_cell", type=int, default=1, help="calibration clips per (sound, condition)")
    ap.add_argument("--calibration", choices=["minmax", "percentile"], default="minmax")
    ap.add_argument("--threads", type=int, default=4, help="ONNX Runtime threads on the server")
    ap.add_argument("--reuse", action="store_true", help="skip quantizing if --out already exists")
    ap.add_argument("--full", action="store_true", help="also separate every clip and write a pieces CSV")
    a = ap.parse_args()

    import onnxruntime as ort

    ts = Path(a.test_set)
    with open(ts / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    conds = {s: np.load(Path(a.conditions) / f"{s}.npy").astype(np.float32).reshape(1, 527) for s in SOUNDS}
    calib, check = split_clips(rows, a.calib_per_cell)
    out = Path(a.out)

    # ---------- 1. quantize ----------
    if a.reuse and out.exists():
        t_q = 0.0
        print(f"Reusing {out} (settings below are re-read from the file)")
    else:
        reader = Reader(calib, conds)
        print(f"1. Calibrating on {len(calib)} clips x 3 queries = {len(reader.items)} windows "
              f"({a.calibration}); this is the slow part ...")
        _, t_q = quantize(Path(a.fp32), out, reader, a.calibration, a.ops, a.keep_fp32)
        print(f"   wrote {out} ({out.stat().st_size / 1e6:.1f} MB, from {Path(a.fp32).stat().st_size / 1e6:.1f} MB) "
              f"in {t_q / 60:.1f} min")

    import hashlib
    h = hashlib.sha256()
    with open(out, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    int8_sha = h.hexdigest()
    print(f"   {out} sha256 {int8_sha} (matches onnx_sha256 in the Pi benchmark summaries)")
    layers, keep_fp32 = int8_layers(out, a.keep_fp32)
    print("   heavy layers: " + ", ".join(f"{k} {v}" for k, v in sorted(layers.items())))
    print(f"   checked: all {len(keep_fp32)} nodes meant to stay FP32 did (STFT, inverse STFT"
          + (", " + ", ".join(a.keep_fp32) if a.keep_fp32 else "") + ")")

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    fp = ort.InferenceSession(a.fp32, so, providers=["CPUExecutionProvider"])
    q8 = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    w0 = {"mixture": np.zeros((1, 1, SEG), np.float32), "condition": conds[SOUNDS[0]]}
    fp.run(["waveform"], w0)
    q8.run(["waveform"], w0)                                       # first-call warm-up, not timed

    # ---------- 2. quick check on clips not used for calibration ----------
    print(f"\n2. Checking on {len(check)} clips not used for calibration")
    sep_rows = []
    t_fp = t_q8 = 0.0
    for i, r in enumerate(check, 1):
        mix = load_clip(r["clip"])
        name = Path(r["clip"]).stem
        if r["contains"] in SOUNDS:
            s = r["contains"]
            t0 = time.perf_counter()
            y_fp = separate(fp, mix, conds[s])
            t1 = time.perf_counter()
            y_q8 = separate(q8, mix, conds[s])
            t_fp += t1 - t0
            t_q8 += time.perf_counter() - t1
            refs = sorted((ts / "events" / f"{name}_events").glob(f"foreground*_{s}.wav"))
            if not refs:
                sys.exit(f"No isolated {s} event for {name}")
            ref = sum(load_clip(f) for f in refs)
            n = min(len(ref), len(mix))
            sep_rows.append({"clip": name, "sound": s, "group": r["condition"],
                             "int8_vs_fp32_db": round(err_db(y_q8, y_fp), 1),
                             "si_sdr_fp32": round(si_sdr(y_fp[:n], ref[:n]), 2),
                             "si_sdr_int8": round(si_sdr(y_q8[:n], ref[:n]), 2)})
        elif r["contains"] == "none":
            for s in SOUNDS:
                y_fp, y_q8 = separate(fp, mix, conds[s]), separate(q8, mix, conds[s])
                e_mix = np.sum(mix ** 2) + 1e-12
                sep_rows.append({"clip": name, "sound": s, "group": "target-absent",
                                 "int8_vs_fp32_db": round(err_db(y_q8, y_fp), 1),
                                 "leak_fp32_db": round(float(10 * np.log10((np.sum(y_fp ** 2) + 1e-12) / e_mix)), 2),
                                 "leak_int8_db": round(float(10 * np.log10((np.sum(y_q8 ** 2) + 1e-12) / e_mix)), 2)})
        if i % 20 == 0:
            print(f"   {i}/{len(check)} clips")

    tgt = [x for x in sep_rows if "si_sdr_fp32" in x]
    deg = np.array([x["si_sdr_fp32"] - x["si_sdr_int8"] for x in tgt])
    print(f"\n   INT8 output vs FP32 output (median error, lower = closer): "
          f"{np.median([x['int8_vs_fp32_db'] for x in sep_rows]):.1f} dB")
    print(f"   {'group':14}{'n':>4}{'SI-SDR FP32':>13}{'SI-SDR INT8':>13}{'loss (median)':>15}")
    for g in ["ebr0", "ebr-5", "overlap"]:
        v = [x for x in tgt if x["group"] == g]
        if v:
            d = [x["si_sdr_fp32"] - x["si_sdr_int8"] for x in v]
            print(f"   {g:14}{len(v):>4}{np.median([x['si_sdr_fp32'] for x in v]):>+12.1f} "
                  f"{np.median([x['si_sdr_int8'] for x in v]):>+12.1f} {np.median(d):>+13.2f} dB")
    for s in SOUNDS:
        v = [x for x in tgt if x["sound"] == s]
        if v:
            print(f"   {s:14}{len(v):>4}{'':26}{np.median([x['si_sdr_fp32'] - x['si_sdr_int8'] for x in v]):>+13.2f} dB")
    leak = [x for x in sep_rows if "leak_fp32_db" in x]
    if leak:
        print(f"   target-absent leakage, median: FP32 {np.median([x['leak_fp32_db'] for x in leak]):+.1f} dB, "
              f"INT8 {np.median([x['leak_int8_db'] for x in leak]):+.1f} dB (n = {len(leak)})")
    mean_deg = float(deg.mean()) if len(deg) else float("nan")
    ok_sisdr = mean_deg <= O1_SISDR_DB
    worst = max(tgt, key=lambda x: x["si_sdr_fp32"] - x["si_sdr_int8"]) if tgt else None
    print(f"\n   O1 SI-SDR (mean per-clip loss): {mean_deg:+.2f} dB over {len(deg)} target clips "
          f"(limit {O1_SISDR_DB:g} dB): {'MET' if ok_sisdr else 'NOT MET'}")
    if worst:
        print(f"   alongside: median per-clip loss {np.median(deg):+.2f} dB; worst clip {worst['clip']} "
              f"{worst['si_sdr_fp32'] - worst['si_sdr_int8']:+.2f} dB ({worst['si_sdr_fp32']:+.1f} -> "
              f"{worst['si_sdr_int8']:+.1f})")
    print("   O1 recall and false alarms: need --full, then experiments/o1_frozen.py")
    print(f"   server CPU, {a.threads} threads, total separation time on target clips (NOT a Pi result): "
          f"FP32 {t_fp:.0f} s, INT8 {t_q8:.0f} s")

    report = {"fp32": a.fp32, "int8": str(out), "int8_sha256": int8_sha, "calibration": a.calibration, "ops": a.ops,
              "keep_fp32_modules": a.keep_fp32, "o1_sisdr_median_loss_db": round(float(np.median(deg)), 3),
              "calibration_clips": [Path(r["clip"]).name for r in calib], "fp32_nodes_kept": keep_fp32, "heavy_layers": layers,
              "quantize_minutes": round(t_q / 60, 1), "o1_sisdr_mean_loss_db": round(mean_deg, 3),
              "o1_sisdr_met": ok_sisdr, "server_seconds_not_pi": {"fp32": round(t_fp, 1), "int8": round(t_q8, 1)},
              "rows": sep_rows,
              "versions": {"onnxruntime": ort.__version__}}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"   report: {out.with_suffix('.json')}")

    # ---------- 3. full run for recall / false alarms ----------
    if a.full:
        full_run(a, rows, conds, q8, ts)


def full_run(a, rows, conds, q8, ts):
    """INT8 output for every clip and query, then Cnn14 piece scores in run_gate1.py's format."""
    import librosa
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from detect_baseline import check_frame_readout, piece_details
    from uss.config import panns_paths_dict
    from uss.utils import get_path, load_pretrained_panns

    tag = Path(a.out).stem.replace("resunet30_at_soft_", "")
    out_dir = ts / f"uss_outputs_{tag}"
    print(f"\n3. INT8 separation of all {len(rows)} clips x 3 queries into {out_dir} (resumable)")
    for i, r in enumerate(rows, 1):
        mix = None
        for s in SOUNDS:
            p = out_dir / Path(r["clip"]).stem / f"query={s}.wav"
            if p.exists():
                continue
            mix = load_clip(r["clip"]) if mix is None else mix
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.wav")
            sf.write(tmp, separate(q8, mix, conds[s]), SR)        # 16-bit WAV, like the uss command
            tmp.replace(p)                                        # a killed run never leaves a half file
        if i % 20 == 0:
            print(f"   {i}/{len(rows)} clips")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained_panns(model_type="Cnn14", checkpoint_path=get_path(panns_paths_dict["Cnn14"]),
                                  freeze=True).to(device).eval()
    check_frame_readout(model, SEG, device)
    pieces = []
    for r in rows:
        clip = Path(r["clip"])
        mix, _ = librosa.load(path=str(clip), sr=SR, mono=True)
        for s in SOUNDS:
            sep, _ = librosa.load(path=str(out_dir / clip.stem / f"query={s}.wav"), sr=SR, mono=True)
            for version, audio in (("before", mix), ("after", sep)):
                for k, (score, frames) in enumerate(piece_details(model, audio, CLASS_IDS[s], SEG, device)):
                    pieces.append({"sound": s, "clip": clip.name, "version": version, "piece": k,
                                   "start_s": round(k * SEG / SR, 3), "score": round(score, 5),
                                   "frames": " ".join(f"{v:.4f}" for v in frames)})
    path = Path(f"results/{ts.name}_{tag}_scores_pieces.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pieces[0].keys()))
        w.writeheader()
        w.writerows(pieces)
    print(f"   wrote {path}\n   next: python experiments/o1_frozen.py --int8_pieces {path} "
          f"--int8_report {Path(a.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()