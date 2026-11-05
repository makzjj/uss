"""Freesound pool for new target recordings: scan, listen, split by uploader, lock (Requirements v0.3.7).

The Gate 1 recordings (sources/<class>/test) are now development data. Held-out targets come from a
new Freesound pool, chosen by LISTENING (never by Cnn14 rank) and split by uploader so that one
person's recordings never sit on both sides. Three commands:

  scan    Reads pool/<class>/<files> (Freesound's own download names: <id>__<uploader>__<title>.<ext>),
          converts each clip to 32 kHz mono WAV in pool_32k/<class>/<id>.wav (originals untouched), and
          writes pool_manifest.csv with: id, uploader, duration, original rate/channels, clipping, a
          short-clip flag (events are 3-6 s), and a duplicate flag from comparing the AUDIO (not file
          names) with every clip in sources/, queries/ and archive/ (--compare_dirs adds more):
          "same recording" (>= 0.90, a renamed, trimmed or re-levelled copy: reject) or "check by ear"
          (>= 0.75, perhaps the same session or source). Pool clips are also compared with each other,
          so a recording uploaded twice by different people cannot leak across the split. Limit: a copy that
          only PARTLY overlaps a dev clip (neither contains the other) may score lower; listen if in doubt.
          Empty columns for your verdict and notes.
          Optional: --api_token fetches licence, pack and tags from the Freesound API.
  (you)   Listen to each clip and fill the verdict column: keep, ambiguous or reject, with a reason in
          notes (Requirements v0.3.7: target audible; no other target class audible; ambiguous clips form
          a labelled subset). Decide BEFORE splitting, by the written rule, not by any model output.
  split   Assigns whole uploaders to heldout or dev_extra with a fixed seed, balancing each class, and
          writes split.csv plus heldout_lock.json (SHA-256 of every held-out file, original and 32 kHz).
          Rejected clips are left out; ambiguous ones keep their flag on whichever side they land.
  verify  Checks every held-out file still matches its locked hash. Run it before the Week 11 evaluation.

Usage (inside the container, from the uss folder; POOL = folder holding <class>/ subfolders):
  python experiments/freesound_pool.py scan  --pool POOL
  python experiments/freesound_pool.py split --pool POOL
  python experiments/freesound_pool.py verify --pool POOL
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

TA = "/home/zmak002/docker_share/test_audio"
SR = 32000
CLASSES = ["beeper", "engine", "jackhammer"]
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"}
NAME_RE = re.compile(r"^(\d+)__([^_].*?)__(.+)$")        # Freesound: <id>__<uploader>__<title>
MIN_S, FULL_S = 3.0, 6.0                                  # Scaper events are 3-6 s
DUP_FLAG = 0.90          # same recording (renamed, trimmed, resampled or level-changed copies)
SIMILAR_FLAG = 0.75      # possibly the same session or source: check by ear
COLS = ["class", "file", "freesound_id", "uploader", "duration_s", "orig_sr", "orig_channels",
        "clipped_frac", "length_flag", "dup_flag", "possible_duplicate_of", "dup_similarity",
        "licence", "pack", "tags", "verdict", "notes"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(audio, sr):
    """Log-mel envelope at 20 frames/s, mean-removed per band: robust to level, trimming and resampling."""
    import librosa
    m = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=sr // 20, n_mels=32)
    f = np.log(m + 1e-10)
    return f - f.mean(axis=1, keepdims=True)


def similarity(a, b):
    """Best correlation of the shorter fingerprint slid along the longer one, with each band's mean removed
    INSIDE each window (a trimmed copy has different whole-clip band means), computed in chunks of offsets."""
    from numpy.lib.stride_tricks import sliding_window_view
    if a.shape[1] > b.shape[1]:
        a, b = b, a
    n = a.shape[1]
    if n < 40:                                    # under 2 s of audio: too short to compare
        return 0.0
    av = a - a.mean(axis=1, keepdims=True)
    av = av / (av.std() + 1e-9)
    wins = sliding_window_view(b, n, axis=1)      # (bands, offsets, n), a view: no copy
    best = 0.0
    for start in range(0, wins.shape[1], 256):
        w = wins[:, start:start + 256, :]
        w = w - w.mean(axis=2, keepdims=True)     # per band, per window
        sd = w.std(axis=(0, 2)) + 1e-9            # per window
        score = np.einsum("bn,bon->o", av, w) / (av.size * sd)
        best = max(best, float(score.max()))
    return best


def freesound_meta(fid, token):
    url = (f"https://freesound.org/apiv2/sounds/{fid}/?token={token}"
           f"&fields=license,pack,tags,username")
    with urllib.request.urlopen(url, timeout=20) as r:
        j = json.load(r)
    pack = j.get("pack") or ""
    return {"licence": j.get("license", ""), "pack": pack.rstrip("/").split("/")[-1] if pack else "",
            "tags": " ".join(j.get("tags", [])[:8]), "uploader_api": j.get("username", "")}


def scan(a):
    import librosa
    import soundfile as sf
    pool = Path(a.pool).resolve()
    if (pool.parent / "heldout_lock.json").exists() and not a.force:
        sys.exit("The held-out set is locked: scanning would rewrite its 32 kHz copies. Use --force only "
                 "with a written reason (and re-run verify afterwards).")
    out32 = pool.parent / (pool.name + "_32k")
    dev = []
    for name in a.compare_dirs:                   # file names do not matter: the check compares the audio
        d = Path(name) if Path(name).is_absolute() else Path(a.ta) / name
        if not d.exists():
            print(f"   (no {d}; skipped)")
            continue
        dev += [p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT and "testsets" not in p.parts]
    print(f"Fingerprinting {len(dev)} dev sources and queries for the duplicate check ...")
    dev_fp = [(p, fingerprint(librosa.load(str(p), sr=16000, mono=True)[0], 16000)) for p in dev]
    rows, seen, pool_fp, unreadable = [], {}, [], []
    for cls in CLASSES:
        files = sorted(p for p in (pool / cls).glob("*") if p.suffix.lower() in AUDIO_EXT)
        for p in files:
            m = NAME_RE.match(p.stem)
            if not m:
                print(f"   skip {p.name}: not a Freesound download name (<id>__<uploader>__<title>)")
                continue
            fid, uploader = m.group(1), m.group(2)
            if fid in seen:
                sys.exit(f"Freesound id {fid} appears twice ({seen[fid]} and {p}); keep one copy")
            seen[fid] = p
            try:
                try:
                    info = sf.info(str(p))
                except Exception:                                 # mp3, m4a or an odd WAV: read via librosa only
                    info = None
                raw, _ = librosa.load(str(p), sr=None, mono=False)    # original rate and channels: clipping is
                clipped = float(np.mean(np.abs(raw) >= 0.999))        # measured before resampling blurs it
                audio, _ = librosa.load(str(p), sr=SR, mono=True)
                if len(audio) == 0:
                    raise ValueError("no audio samples")
            except Exception as e:
                unreadable.append((p, str(e).splitlines()[0][:120]))
                print(f"   SKIP {p.name}: unreadable ({unreadable[-1][1]})")
                continue
            dur = len(audio) / SR
            dst = out32 / cls / f"{fid}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, audio, SR, subtype="FLOAT")             # float: conversion adds no 16-bit rounding
            fp = fingerprint(librosa.resample(audio, orig_sr=SR, target_sr=16000), 16000)
            best = max(((similarity(fp, f), q) for q, f in dev_fp + pool_fp), default=(0.0, None),
                       key=lambda x: x[0])
            pool_fp.append((p, fp))                                  # later pool clips compare with this one
            row = {"class": cls, "file": p.name, "freesound_id": fid, "uploader": uploader,
                   "duration_s": round(dur, 2), "orig_sr": info.samplerate if info else "",
                   "orig_channels": info.channels if info else "",
                   "clipped_frac": round(clipped, 5),
                   "length_flag": "too short" if dur < MIN_S else ("short: events capped" if dur < FULL_S else ""),
                   "possible_duplicate_of": ((("pool: " + best[1].name) if best[1] in [q for q, _ in pool_fp]
                                              else str(best[1])) if best[0] >= SIMILAR_FLAG else ""),
                   "dup_flag": ("same recording" if best[0] >= DUP_FLAG else
                                "check by ear" if best[0] >= SIMILAR_FLAG else ""),
                   "dup_similarity": round(best[0], 3), "licence": "", "pack": "", "tags": "",
                   "verdict": "", "notes": ""}
            if a.api_token:
                try:
                    meta = freesound_meta(fid, a.api_token)
                    row.update({k: meta[k] for k in ("licence", "pack", "tags")})
                    if meta["uploader_api"] and meta["uploader_api"] != uploader:
                        row["notes"] = f"API uploader {meta['uploader_api']} differs from file name"
                except Exception as e:                            # metadata is optional
                    row["notes"] = f"API lookup failed: {e}"
            rows.append(row)
            print(f"   {cls:10} {fid:>8} {uploader:20} {dur:6.1f} s"
                  + (f"  [{row['length_flag']}]" if row["length_flag"] else "")
                  + (f"  [{row['dup_flag']}: {Path(row['possible_duplicate_of']).name} ({best[0]:.2f})]"
                     if row["dup_flag"] else "")
                  + ("  [clipping]" if row["clipped_frac"] > 0.001 else ""))
    man = pool.parent / "pool_manifest.csv"
    old = {}
    if man.exists():                                              # keep verdicts already written
        with open(man, encoding="utf-8-sig") as fh:          # Excel on Windows may add a BOM
            old = {r["freesound_id"]: r for r in csv.DictReader(fh)}
    for r in rows:
        if r["freesound_id"] in old:
            r["verdict"], r["notes"] = old[r["freesound_id"]]["verdict"], old[r["freesound_id"]]["notes"] or r["notes"]
    with open(man, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    if unreadable:
        print(f"\n{len(unreadable)} file(s) could not be read and are NOT in the manifest:")
        for p, why in unreadable:
            print(f"   {p}  ({why})")
        print("   Check each: an incomplete copy (re-copy it), or not really the format its name says "
              "(re-export it as WAV, e.g. in Audacity). Then run scan again; verdicts already written are kept.")
    per = {c: sum(r["class"] == c for r in rows) for c in CLASSES}
    print(f"\nWrote {man} ({', '.join(f'{c} {n}' for c, n in per.items())}) and 32 kHz copies in {out32}")
    print("Next: listen, fill 'verdict' (keep / ambiguous / reject) and 'notes', then run split.")


def split(a):
    pool = Path(a.pool).resolve()                       # absolute paths in the lock, so verify works anywhere
    man = pool.parent / "pool_manifest.csv"
    with open(man, encoding="utf-8-sig") as fh:              # survives an Excel save
        rows = list(csv.DictReader(fh))
    bad = [r["file"] for r in rows if r["verdict"].strip().lower() not in ("keep", "ambiguous", "reject")]
    if bad:
        sys.exit(f"{len(bad)} clips have no valid verdict yet (keep / ambiguous / reject), e.g. {bad[:3]}")
    lock_path = pool.parent / "heldout_lock.json"
    if lock_path.exists() and not a.force:
        sys.exit(f"{lock_path} already exists: the held-out set is locked. Use --force only with a written reason.")
    use = [r for r in rows if r["verdict"].strip().lower() != "reject"]
    if not use:
        sys.exit("Every clip is rejected: nothing to split")
    up_classes = {}
    for r in use:
        up_classes.setdefault(r["uploader"], []).append(r["class"])
    uploaders = sorted(up_classes)
    random.Random(a.seed).shuffle(uploaders)                       # seeded tie-break among equal sizes
    uploaders.sort(key=lambda u: -len(up_classes[u]))              # biggest uploaders placed first
    target = {c: a.heldout_share * sum(r["class"] == c for r in use) for c in CLASSES}
    side, held = {}, {c: 0 for c in CLASSES}
    for u in uploaders:                         # held-out only if that brings the class totals closer to target
        cls = up_classes[u]
        add = {c: cls.count(c) for c in set(cls)}
        before = sum(abs(target[c] - held[c]) for c in add)
        after = sum(abs(target[c] - held[c] - k) for c, k in add.items())
        side[u] = "heldout" if after < before else "dev_extra"
        if side[u] == "heldout":
            for c, k in add.items():
                held[c] += k
    for c in CLASSES:
        n_c = sum(r["class"] == c for r in use)
        if n_c and held[c] == 0:
            print(f"WARNING: {c} has no held-out clips; add clips from more uploaders before splitting")
        elif n_c and abs(held[c] - target[c]) > max(1.0, 0.25 * n_c):
            print(f"WARNING: {c} held-out {held[c]} vs target {target[c]:.1f}: one uploader dominates; "
                  f"consider more uploaders")
    out32 = pool.parent / (pool.name + "_32k")
    split_rows, lock = [], []
    for r in use:
        s = side[r["uploader"]]
        f32 = out32 / r["class"] / f"{r['freesound_id']}.wav"
        split_rows.append({**r, "split": s, "subset": "ambiguity" if r["verdict"].strip().lower() == "ambiguous"
                           else "main", "file_32k": str(f32)})
        if s == "heldout":
            lock.append({"class": r["class"], "freesound_id": r["freesound_id"], "uploader": r["uploader"],
                         "subset": split_rows[-1]["subset"], "file": r["file"],
                         "sha256_original": sha256(pool / r["class"] / r["file"]),
                         "file_32k": str(f32), "sha256_32k": sha256(f32)})
    with open(pool.parent / "split.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(split_rows[0].keys()))
        w.writeheader()
        w.writerows(split_rows)
    lock_path.write_text(json.dumps({"locked": datetime.now().isoformat(timespec="seconds"), "seed": a.seed,
                                     "heldout_share": a.heldout_share, "rule": "Requirements v0.3.7 selection by listening",
                                     "files": lock}, indent=2))
    print(f"{'class':12}{'keep+ambiguous':>16}{'held-out':>10}{'dev_extra':>11}{'rejected':>10}")
    for c in CLASSES:
        n = [r for r in split_rows if r["class"] == c]
        print(f"{c:12}{len(n):>16}{sum(r['split'] == 'heldout' for r in n):>10}"
              f"{sum(r['split'] == 'dev_extra' for r in n):>11}"
              f"{sum(r['class'] == c and r['verdict'].strip().lower() == 'reject' for r in rows):>10}")
    print(f"Uploaders: {sum(v == 'heldout' for v in side.values())} held-out, "
          f"{sum(v == 'dev_extra' for v in side.values())} dev_extra; none on both sides by construction.")
    print(f"Locked {len(lock)} held-out files in {lock_path}. Do not run any model on them before Week 11.")


def verify(a):
    pool = Path(a.pool).resolve()
    lock = json.loads((pool.parent / "heldout_lock.json").read_text())
    bad = []
    for f in lock["files"]:
        orig = pool / f["class"] / f["file"]
        for path, want in ((Path(f["file_32k"]), f["sha256_32k"]), (orig, f["sha256_original"])):
            if not path.exists() or sha256(path) != want:
                bad.append(str(path))
    print(f"Held-out set locked {lock['locked']}: {len(lock['files'])} files (32 kHz copies and originals), "
          + ("all match their locked hashes." if not bad else f"{len(bad)} CHANGED OR MISSING, e.g. {bad[0]}"))
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["scan", "split", "verify"])
    ap.add_argument("--pool", required=True, help="folder holding beeper/, engine/, jackhammer/ downloads")
    ap.add_argument("--ta", default=TA, help="test_audio root (for the duplicate check)")
    ap.add_argument("--compare_dirs", nargs="+", default=["sources", "queries", "archive"],
                    help="folders of earlier audio to compare against (relative to --ta, or absolute); "
                         "any audio file counts, whatever its name")
    ap.add_argument("--api_token", default="", help="optional Freesound API key for licence, pack and tags")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--heldout_share", type=float, default=0.5)
    ap.add_argument("--force", action="store_true", help="re-scan or re-split over an existing lock (state why in notes)")
    a = ap.parse_args()
    {"scan": scan, "split": split, "verify": verify}[a.command](a)


if __name__ == "__main__":
    main()