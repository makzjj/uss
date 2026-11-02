"""Week 6: query caching. Compute each target class's query condition ONCE and save it.

What a query condition is: USS asked "by example" runs every example recording through its
query net (Cnn14 tagging, at_soft) and averages the outputs into one 527-number vector. That
vector is the only thing the separator needs to know which sound to extract. It depends only on
the example recordings, never on the audio being separated, so it can be computed once and reused.

The stock `uss --queries_dir` command recomputes it on every call (it also writes a copy to
./query_conditions/). This script computes it with the same function and settings, saves it as
a plain .npy file with a record of where it came from, and checks:
  1. it matches the copy Gate 1 left in ./query_conditions/ (if present), and
  2. separating with the cached vector reproduces Gate 1's saved USS outputs on a few clips.

Writes results/setup/query_conditions/<sound>.npy and <sound>.json (files used with SHA-256,
config, checkpoint, top classes in the vector).

Usage (inside the container, from the uss folder):
  python experiments/cache_queries.py
"""
import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_gate1 import SOUNDS, TA, uss_output  # noqa: E402
from uss.config import IX_TO_LB  # noqa: E402
from uss.inference import calculate_query_emb, load_ss_model, separate_by_query_condition  # noqa: E402
from uss.utils import parse_yaml  # noqa: E402

HOME = Path.home()
CONFIG = HOME / ".cache/uss/scripts/ss_model=resunet30,querynet=at_soft,data=full.yaml"
CKPT = HOME / ".cache/uss/checkpoints/ss_model=resunet30,querynet=at_soft,data=full,devices=8,step=1000000.ckpt"
PCM16_STEP = 1 / 32768          # Gate 1 outputs are 16-bit WAV, so they differ by rounding only


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/setup/query_conditions")
    ap.add_argument("--test_set", default=f"{TA}/testsets/gate1_v2")
    ap.add_argument("--check_clips", type=int, default=3, help="Gate 1 clips per class to re-separate")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = parse_yaml(str(CONFIG))
    sr = configs["data"]["sample_rate"]
    seg = int(sr * configs["data"]["segment_seconds"])
    print("Loading USS (query net + separator) ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"      # same rule as the uss command in Gate 1
    pl_model = load_ss_model(configs=configs, checkpoint_path=str(CKPT)).to(device).eval()
    print(f"Using {device} (Gate 1 used the same rule, so outputs should match within rounding)")

    with open(Path(a.test_set) / "manifest.csv") as fh:
        manifest = list(csv.DictReader(fh))

    ok_all = True
    for name, (class_id, qdir) in SOUNDS.items():
        files = sorted(Path(qdir).glob("*.wav"))      # same file list and order as calculate_query_emb
        t0 = time.perf_counter()
        cond = calculate_query_emb(queries_dir=qdir, pl_model=pl_model, sample_rate=sr,
                                   remove_sil=True, segment_samples=seg).astype(np.float32)
        t_query = time.perf_counter() - t0
        np.save(out / f"{name}.npy", cond)
        top = np.argsort(cond)[::-1][:5]
        record = {
            "sound": name, "target_class_id": class_id, "target_label": IX_TO_LB[class_id],
            "queries_dir": qdir, "files": [{"name": f.name, "sha256": sha256(f)} for f in files],
            "config_yaml": str(CONFIG), "checkpoint": str(CKPT), "shape": list(cond.shape),
            "remove_silence": True, "segment_samples": seg, "sample_rate": sr,
            "query_net_seconds_server": round(t_query, 2),
            "top_classes": [{"id": int(i), "label": IX_TO_LB[int(i)], "value": round(float(cond[i]), 4)}
                            for i in top],
            "target_rank": int((cond > cond[class_id]).sum()) + 1,
        }
        print(f"\n{name}: {len(files)} example files, query net {t_query:.1f} s on {device}; "
              f"target class {class_id} ({IX_TO_LB[class_id]}) = {cond[class_id]:.4f}, "
              f"rank {record['target_rank']} of 527")
        print("   top 5: " + ", ".join(f"{t['label']} {t['value']:.3f}" for t in record["top_classes"]))

        # Check 1: the copy the uss command wrote during Gate 1
        old = Path("query_conditions") / f"config={CONFIG.stem}" / f"{Path(qdir).stem}.pkl"
        if old.exists():
            diff = float(np.abs(np.asarray(pickle.load(open(old, "rb")), np.float32) - cond).max())
            record["max_abs_diff_vs_gate1_pkl"] = diff
            print(f"   vs Gate 1 copy {old}: max abs difference {diff:.2e}  {'OK' if diff < 1e-6 else 'DIFFERS'}")
            ok_all &= diff < 1e-6
        else:
            print(f"   (no Gate 1 copy at {old}; skipped)")

        # Check 2: separating with the cached vector reproduces Gate 1's saved outputs
        clips = [r["clip"] for r in manifest if r["contains"] == name][: a.check_clips]
        record["reproduce_check"] = []
        for clip in clips:
            saved = uss_output(a.test_set, clip, qdir)
            if not saved.exists():
                print(f"   skip {Path(clip).name}: no saved Gate 1 output at {saved}")
                continue
            mix, _ = librosa.load(clip, sr=sr, mono=True)
            sep = separate_by_query_condition(audio=mix, segment_samples=seg, sample_rate=sr,
                                              query_condition=cond, pl_model=pl_model, output_path="")
            ref, _ = sf.read(saved, dtype="float32")
            d = float(np.abs(sep[: len(ref)] - ref).max())
            ok = d <= 2 * PCM16_STEP
            ok_all &= ok
            record["reproduce_check"].append({"clip": Path(clip).name, "max_abs_diff": d, "ok": ok})
            hint = ""
            if not ok and np.abs(ref).max() >= 0.999:
                hint = "; saved output touches full scale, so 16-bit clipping may explain it"
            print(f"   re-separate {Path(clip).name}: max abs difference {d:.2e} "
                  f"({'OK, within 16-bit rounding' if ok else 'DIFFERS' + hint})")
        if not record["reproduce_check"]:
            print(f"   FAIL: no Gate 1 outputs found for {name}, so check 2 tested nothing")
            ok_all = False
        (out / f"{name}.json").write_text(json.dumps(record, indent=2))

    print(f"\nSaved <sound>.npy and <sound>.json in {out}")
    print("All checks passed." if ok_all else "SOME CHECKS FAILED: do not use these files until explained.")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()