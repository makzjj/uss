"""Week 6: export the ResUNet30 separator to ONNX and check it gives the same output as PyTorch.

Separator only: inputs are one 2 s window (1, 1, 64000) and a cached query condition (1, 527)
from experiments/cache_queries.py; output is the separated window (1, 1, 64000). The query net is
not in the file, because the condition is computed once per class and reused.

One change is needed for export: torchlibrosa's inverse STFT uses two operations older ONNX
cannot express (flip and fold). torchlibrosa ships an ONNX mode that does the same maths with
convolutions; it needs the frame count fixed, which a fixed 2 s window gives. Step 1 checks the
swap changes nothing before exporting.

Checks (all against the unmodified PyTorch model, on real 2 s windows and real conditions):
  1. PyTorch with the ONNX-mode inverse STFT vs original PyTorch
  2. ONNX Runtime vs original PyTorch
  Reported as max abs difference and the error level in dB relative to the output
  (-60 dB = errors a thousand times smaller than the signal). A window passes at <= -60 dB, or
  if its max abs difference is below 1e-4 (a near-silent output can have a large relative error
  from a tiny absolute one, e.g. a wrong-class query on target-free audio).
  3. Graph report: operator counts before and after ONNX Runtime's own optimizations (shows
     whether conv + batch-norm got fused), and server CPU time per window for information only.
     Server x86 timings say nothing about the Pi; time the Pi separately.

Fixed shape: one window per call, (1, 1, 64000), no batch dimension to vary. This matches the
Pi's one-window-at-a-time pipeline; batching several windows needs a re-export with dynamic axes.

Needs: torch, einops, torchlibrosa (as on the Pi), onnx, onnxruntime.
Exits with status 1 if any check fails.
Writes <out>.onnx and <out>.json (checks, hashes, versions).

Usage (inside the container, from the uss folder):
  python benchmarks/export_onnx.py --weights resunet30_at_soft_fp32.pth
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
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from uss.models.resunet import ResUNet30  # noqa: E402

SR, SEG = 32000, 64000          # at_soft config: 32 kHz, 2 s segments
N_FFT, HOP = 2048, 320          # ResUNet30_Base STFT settings
PASS_DB = -60.0
PASS_ABS = 1e-4
TA = "/home/zmak002/docker_share/test_audio"


class Separator(nn.Module):
    """Plain tensor in, tensor out, so the exported graph has named inputs and one output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mixture, condition):
        return self.model({"mixture": mixture, "condition": condition})["waveform"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def err_db(out, ref):
    """Error level relative to the reference output, in dB (more negative = closer)."""
    return float(10 * np.log10(np.sum((out - ref) ** 2) / max(np.sum(ref ** 2), 1e-20) + 1e-30))


def load_model(weights):
    m = ResUNet30(input_channels=1, output_channels=1, condition_size=527)
    m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return m.eval()


def swap_istft_for_onnx(model, frames_num):
    from torchlibrosa.stft import ISTFT
    model.base.istft = ISTFT(n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window="hann", center=True,
                             pad_mode="reflect", freeze_parameters=True, onnx=True,
                             frames_num=frames_num, device="cpu")
    return model


def window_ok(e_db, max_abs):
    return e_db <= PASS_DB or max_abs < PASS_ABS


def test_windows(conditions_dir, test_set, per_cell):
    """Real 2 s windows: per_cell clips for every (sound, condition) pair plus one target-free
    clip, each paired with every cached class condition."""
    import soundfile as sf
    conds = {p.stem: np.load(p).astype(np.float32).reshape(1, 527) for p in sorted(Path(conditions_dir).glob("*.npy"))}
    if not conds:
        sys.exit(f"No cached conditions in {conditions_dir}; run experiments/cache_queries.py first")
    with open(Path(test_set) / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    cells = {}
    for r in rows:
        if r["contains"] != "none":
            cells.setdefault((r["contains"], r["condition"]), []).append(r)
    picks = [r for key in sorted(cells) for r in cells[key][:per_cell]]
    picks += [r for r in rows if r["contains"] == "none"][:1]          # one target-free clip too
    wins = []
    for r in picks:
        audio, sr = sf.read(r["clip"], dtype="float32")
        if sr != SR or audio.ndim != 1:
            sys.exit(f"{r['clip']}: expected {SR} Hz mono")
        for start in (0, SEG * 2):
            x = audio[start:start + SEG]
            if len(x) == SEG:
                for name, c in conds.items():
                    wins.append((f"{Path(r['clip']).stem}@{start // SR}s|{name}", x.reshape(1, 1, SEG), c))
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="resunet30_at_soft_fp32.pth",
                    help="plain state_dict from benchmarks/export_resunet_weights.py (same file the Pi uses)")
    ap.add_argument("--conditions", default="results/setup/query_conditions")
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--per_cell", type=int, default=1, help="clips per (sound, condition) pair")
    ap.add_argument("--out", default="resunet30_at_soft_fp32.onnx")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    torch.set_num_threads(4)

    import onnx
    import onnxruntime as ort

    ref_model = load_model(a.weights)
    with torch.no_grad():
        frames_num = ref_model.base.stft(torch.zeros(1, SEG))[0].shape[2]
    exp_model = swap_istft_for_onnx(load_model(a.weights), frames_num)
    wins = test_windows(a.conditions, a.test_set, a.per_cell)
    print(f"{len(wins)} test windows (real audio x cached conditions); STFT frames per window: {frames_num}")

    # ---------- 1. ONNX-mode inverse STFT changes nothing ----------
    refs, worst1, fails1 = {}, -np.inf, []
    with torch.no_grad():
        for tag, x, c in wins:
            mix, cond = torch.from_numpy(x), torch.from_numpy(c)
            r = ref_model({"mixture": mix, "condition": cond})["waveform"].numpy()
            e = exp_model({"mixture": mix, "condition": cond})["waveform"].numpy()
            refs[tag] = r
            e_db, d = err_db(e, r), float(np.abs(e - r).max())
            worst1 = max(worst1, e_db)
            if not window_ok(e_db, d):
                fails1.append(f"{tag} ({e_db:.1f} dB, max abs {d:.1e})")
    ok1 = not fails1
    print(f"1. ONNX-mode inverse STFT vs original PyTorch: worst error {worst1:.1f} dB "
          f"({'OK' if ok1 else 'FAIL: ' + '; '.join(fails1[:5])})")
    if not ok1:
        sys.exit("The inverse STFT swap changes the output; stopping before export.")

    # ---------- export ----------
    out = Path(a.out)
    x0, c0 = torch.from_numpy(wins[0][1]), torch.from_numpy(wins[0][2])
    torch.onnx.export(Separator(exp_model), (x0, c0), str(out), opset_version=a.opset,
                      input_names=["mixture", "condition"], output_names=["waveform"],
                      do_constant_folding=True)
    onnx.checker.check_model(str(out))
    graph_ops = Counter(n.op_type for n in onnx.load(str(out)).graph.node)
    print(f"Exported {out} ({out.stat().st_size / 1e6:.1f} MB, opset {a.opset}, {sum(graph_ops.values())} nodes)")

    # ---------- 2. ONNX Runtime vs original PyTorch ----------
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt_path = out.with_name(out.stem + "_ort_optimized_server.onnx")
    so.optimized_model_filepath = str(opt_path)
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    per_win, worst2, max_abs, fails2 = [], -np.inf, 0.0, []
    for tag, x, c in wins:
        y = sess.run(["waveform"], {"mixture": x, "condition": c})[0]
        e = err_db(y, refs[tag])
        d = float(np.abs(y - refs[tag]).max())
        worst2, max_abs = max(worst2, e), max(max_abs, d)
        ok = window_ok(e, d)
        if not ok:
            fails2.append(f"{tag} ({e:.1f} dB, max abs {d:.1e})")
        per_win.append({"window": tag, "err_db": round(e, 1), "max_abs_diff": d, "ok": ok,
                        "ref_peak": float(np.abs(refs[tag]).max())})
    ok2 = not fails2
    print(f"2. ONNX Runtime vs original PyTorch: worst error {worst2:.1f} dB, max abs difference {max_abs:.2e} "
          f"({'OK' if ok2 else 'FAIL on ' + str(len(fails2)) + ' windows: ' + '; '.join(fails2[:5])})")

    # ---------- 3. graph report and server timing (information only) ----------
    opt_ops = Counter(n.op_type for n in onnx.load(str(opt_path)).graph.node)
    for op in ("Conv", "BatchNormalization", "Relu", "LeakyRelu", "ConvTranspose"):
        if graph_ops[op] or opt_ops[op]:
            print(f"   {op:20} exported {graph_ops[op]:4}  after ORT optimization {opt_ops[op]:4}")
    fused = sorted(op for op in opt_ops if op.startswith("Fused"))
    if fused:
        print("   fused operators: " + ", ".join(f"{op} {opt_ops[op]}" for op in fused))
    times = {}
    x, c = wins[0][1], wins[0][2]
    for label, run in (("PyTorch, 1 thread", None), ("ONNX Runtime, 1 thread", sess)):
        if run is None:
            torch.set_num_threads(1)
            f = lambda: ref_model({"mixture": torch.from_numpy(x), "condition": torch.from_numpy(c)})  # noqa: E731
        else:
            f = lambda: sess.run(["waveform"], {"mixture": x, "condition": c})  # noqa: E731
        with torch.no_grad():
            for _ in range(2):
                f()
            t = []
            for _ in range(5):
                t0 = time.perf_counter()
                f()
                t.append(time.perf_counter() - t0)
        times[label] = round(float(np.median(t)), 3)
    print("   server CPU, median of 5 (NOT a Pi result): "
          + ", ".join(f"{k} {v:.2f} s" for k, v in times.items()))

    report = {
        "onnx": str(out), "onnx_sha256": sha256(out), "weights": a.weights, "weights_sha256": sha256(a.weights),
        "opset": a.opset, "inputs": {"mixture": [1, 1, SEG], "condition": [1, 527]}, "output": [1, 1, SEG],
        "stft_frames": frames_num, "istft": "torchlibrosa ISTFT onnx=True",
        "check1_istft_swap_worst_db": round(worst1, 1), "check2_ort_worst_db": round(worst2, 1),
        "check2_ort_max_abs_diff": max_abs, "pass_db": PASS_DB, "pass_abs": PASS_ABS, "passed": ok1 and ok2,
        "batching": "none: fixed shape (1, 1, 64000), one window per call",
        "windows": per_win, "ops_exported": dict(graph_ops), "ops_after_ort_optimization": dict(opt_ops),
        "server_cpu_seconds_not_pi": times,
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"\n{'PASSED' if report['passed'] else 'FAILED'}: report in {out.with_suffix('.json')}")
    print("The *_ort_optimized_server.onnx file is for inspection only (may hold x86-only operators);")
    print(f"ship {out} to the Pi. Keep both .onnx files out of git (see .gitignore).")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()