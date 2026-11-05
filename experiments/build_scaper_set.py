"""Build the Gate 1 v2 test set with Scaper (Requirements v0.3.5 test conditions).

Conditions (target loudness relative to background, set with Scaper's snr = EBR in LUFS):
  ebr0     : target at 0 dB EBR, no other foreground sound
  ebr-5    : target at -5 dB EBR
  overlap  : target at 0 dB EBR plus an interferer at the same level (input SIR 0 dB)
Target-absent clips (negatives) for every background:
  bgonly   : background alone (one per background; SINGA:PURA clips are exactly 10 s,
             so repeats of the same background would be identical)
  int      : background + interferer at 0 dB EBR (--neg_per_bg per background, each with a
             different interferer)

Pairing: each (sound, recording, background) uses the same random seed in every condition,
so between ebr0, ebr-5 and overlap only the level or the interferer changes, not the event
placement. Seeds come from the clip identity, not the build order, so rebuilding a subset
gives the same audio.

Uses, by default (--targets dev --backgrounds dev), development data only:
  sources/<sound>/test/*      clean target recordings (beeper, engine, jackhammer)
  backgrounds/dev/*.flac      development backgrounds
  interferers/*               non-target foreground sounds
--targets dev_extra / heldout instead takes the 32 kHz copies of the Freesound pool clips on that side of
split.csv (experiments/freesound_pool.py). heldout also needs --confirm_heldout and first verifies every file
against heldout_lock.json; use it once, for the Week 11 evaluation, normally with --backgrounds heldout.
Safety: --out_dir is required for dev_extra / heldout, an existing set folder is never replaced without
--overwrite (USS outputs live inside it), and --backgrounds heldout only works for the held-out build.
Limitation: interferers/ is shared by every set, including held-out (it was used in development).

Writes to testsets/gate1_v2/:
  clips/<name>.wav            10 s, 32 kHz mono soundscapes
  events/<name>_events/       each foreground event and the background on their own (for SI-SDR)
  jams/<name>.jams            Scaper annotation
  manifest.csv                one row per clip: condition, target, onset/offset, sources, seed
Old clips in the output folder are removed first (only if it is inside a "testsets" folder),
so the set always matches this run.

Usage (inside the container, from the uss folder):
  python experiments/build_scaper_set.py                                              # Gate 1 (dev)
  python experiments/build_scaper_set.py --targets dev_extra --out_dir <TA>/testsets/extradev_v1
  python experiments/build_scaper_set.py --targets heldout --backgrounds heldout --confirm_heldout \
      --out_dir <TA>/testsets/heldout_v1                                               # Week 11, once
"""
import argparse
import csv
import hashlib
import json
import shutil
import warnings
import zlib
from pathlib import Path

import numpy as np

if not hasattr(np, "Inf"):   # Scaper 1.6.5 uses np.Inf, removed in NumPy 2
    np.Inf = np.inf

import scaper  # noqa: E402
import soundfile as sf  # noqa: E402

TA = "/home/zmak002/docker_share/test_audio"
SOUNDS = ["beeper", "engine", "jackhammer"]
AUDIO_EXT = (".wav", ".flac")
DUR = 10.0
SR = 32000
REF_DB = -30          # background loudness (LUFS); targets are set relative to this
EVENT_DUR = (3.0, 6.0)
EVENT_START = (0.5, 3.5)  # start + max duration stays inside the 10 s clip


def audio_files(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in AUDIO_EXT)


def link_library(lib, groups):
    """Scaper expects <root>/<label>/<files>; build that with symlinks."""
    for (root, label), files in groups.items():
        d = lib / root / label
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f.name).symlink_to(f.resolve())


def duration(path):
    return sf.info(str(path)).duration


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pool_targets(ta, side, split_csv, lock_json, confirm):
    """Target recordings from the Freesound pool split: the 32 kHz copies on one side."""
    if side == "heldout":
        if not confirm:
            raise SystemExit("--targets heldout needs --confirm_heldout: the held-out set is used once (Week 11)")
        lock = json.loads(Path(lock_json).read_text())
        base = Path(lock_json).resolve().parent
        path = lambda f: Path(f) if Path(f).is_absolute() else (base / f).resolve()  # noqa: E731
        bad = [f["file_32k"] for f in lock["files"]
               if not path(f["file_32k"]).exists() or sha256(path(f["file_32k"])) != f["sha256_32k"]]
        if bad:
            raise SystemExit(f"{len(bad)} held-out files do not match heldout_lock.json, e.g. {bad[0]}; stopping")
        print(f"Held-out lock verified: {len(lock['files'])} files unchanged since {lock['locked']}")
    with open(split_csv, encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == side]
    if not rows:
        raise SystemExit(f"No '{side}' clips in {split_csv}")
    base = Path(split_csv).resolve().parent
    fix = lambda f: Path(f) if Path(f).is_absolute() else (base / f).resolve()  # noqa: E731
    out = {s: sorted(fix(r["file_32k"]) for r in rows if r["class"] == s) for s in SOUNDS}
    subsets = {fix(r["file_32k"]).name: r["subset"] for r in rows}
    for s, fs in out.items():
        for f in fs:
            if not f.exists():
                raise SystemExit(f"{f} (from {split_csv}) not found")
            if duration(f) < EVENT_DUR[1]:
                print(f"Note: {f.name} is {duration(f):.1f} s, shorter than the longest event ({EVENT_DUR[1]} s); "
                      f"its events are capped (short_event column)")
    return out, subsets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_audio", default=TA)
    p.add_argument("--out_dir", default="", help="required unless rebuilding Gate 1 (--targets dev)")
    p.add_argument("--overwrite", action="store_true",
                   help="allow replacing an existing set folder (its USS outputs and results are deleted)")
    p.add_argument("--conditions", nargs="+", default=["ebr0", "ebr-5", "overlap"])
    p.add_argument("--bgs_per_recording", type=int, default=5,
                   help="backgrounds each target recording is mixed with (rotated so all are used)")
    p.add_argument("--neg_per_bg", type=int, default=3,
                   help="background + interferer negatives per background (each a different interferer)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--targets", choices=["dev", "dev_extra", "heldout"], default="dev",
                   help="dev: sources/<sound>/test (Gate 1); dev_extra / heldout: that side of the Freesound split")
    p.add_argument("--backgrounds", choices=["dev", "heldout"], default="dev")
    p.add_argument("--split_csv", default=f"{TA}/split.csv")
    p.add_argument("--lock", default=f"{TA}/heldout_lock.json")
    p.add_argument("--confirm_heldout", action="store_true")
    a = p.parse_args()

    ta = Path(a.test_audio)
    if not a.out_dir:
        if a.targets != "dev":
            raise SystemExit("--targets dev_extra / heldout needs an explicit --out_dir (e.g. testsets/extradev_v1)")
        a.out_dir = f"{TA}/testsets/gate1_v2"
    if a.backgrounds == "heldout" and not (a.targets == "heldout" and a.confirm_heldout):
        raise SystemExit("--backgrounds heldout is only for the held-out build (--targets heldout --confirm_heldout)")
    out = Path(a.out_dir).resolve()
    if "testsets" not in out.parent.parts:
        raise SystemExit(f"Refusing to write to {out}: output must be a folder INSIDE 'testsets' "
                         "(e.g. testsets/gate1_v2), never 'testsets' itself or anywhere else")
    if out.exists() and any(out.iterdir()):
        if not a.overwrite:
            extra = " It holds USS outputs (hours of compute)." if (out / "uss_outputs").exists() else ""
            raise SystemExit(f"{out} already exists.{extra} Choose a new --out_dir, or pass --overwrite to replace it")
        shutil.rmtree(out)
    elif out.exists():
        shutil.rmtree(out)
    for sub in ("clips", "events", "jams"):
        (out / sub).mkdir(parents=True)

    subsets = {}
    if a.targets == "dev":
        targets = {s: audio_files(ta / "sources" / s / "test") for s in SOUNDS}
    else:
        targets, subsets = pool_targets(ta, a.targets, a.split_csv, a.lock, a.confirm_heldout)
    bgs = sorted((ta / "backgrounds" / a.backgrounds).glob("*.flac"))
    ints = audio_files(ta / "interferers")
    for s, fs in targets.items():
        if not fs:
            raise SystemExit(f"No test recordings for {s}")
    if not bgs or not ints:
        raise SystemExit(f"Missing backgrounds/{a.backgrounds} or interferers")
    stems = [i.stem for i in ints]
    if len(stems) != len(set(stems)):
        raise SystemExit("Two interferer files share a name (e.g. x.wav and x.flac); rename one")
    if a.bgs_per_recording > len(bgs):
        print(f"Only {len(bgs)} backgrounds: using {len(bgs)} per recording instead of {a.bgs_per_recording}")
        a.bgs_per_recording = len(bgs)
    if a.neg_per_bg > len(ints):
        print(f"Only {len(ints)} interferers: using {len(ints)} interferer negatives per background")
        a.neg_per_bg = len(ints)

    lib = out / "lib"
    groups = {("foreground", s): fs for s, fs in targets.items()}
    groups[("foreground", "interferer")] = ints
    groups[("background", "street")] = bgs
    link_library(lib, groups)
    fg_path, bg_path = str(lib / "foreground"), str(lib / "background")
    libfile = lambda root, label, f: str(lib / root / label / f.name)  # noqa: E731

    rows = []
    warn_log = []

    def make(name, seed_key, bg, events, condition, contains, target_src="", interferer_src="",
             target_ebr="", interferer_ebr=""):
        seed = (a.seed + zlib.crc32(seed_key.encode())) % (2 ** 31)
        sc = scaper.Scaper(DUR, fg_path, bg_path, random_state=seed)
        sc.sr = SR
        sc.n_channels = 1
        sc.ref_db = REF_DB
        sc.add_background(label=("const", "street"),
                          source_file=("const", libfile("background", "street", bg)),
                          source_time=("const", 0))
        for ev in events:
            src_max = max(0.0, duration(ev["file"]) - ev.get("dur", EVENT_DUR)[1])
            sc.add_event(label=("const", ev["label"]),
                         source_file=("const", libfile("foreground", ev["label"], ev["file"])),
                         source_time=("uniform", 0, src_max) if src_max > 0 else ("const", 0),
                         event_time=("uniform", *ev.get("start", EVENT_START)),
                         event_duration=("uniform", *ev.get("dur", EVENT_DUR)),
                         snr=("const", ev["snr"]), pitch_shift=None, time_stretch=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, ann, _ = sc.generate(
                audio_path=str(out / "clips" / f"{name}.wav"),
                jams_path=str(out / "jams" / f"{name}.jams"),
                allow_repeated_label=True, allow_repeated_source=True,
                reverb=None, fix_clipping=True, peak_normalization=False,
                save_isolated_events=True,
                isolated_events_path=str(out / "events" / f"{name}_events"),
                disable_sox_warnings=True)
        for wmsg in caught:
            if issubclass(wmsg.category, DeprecationWarning):
                continue          # library noise (e.g. jsonschema), not about the audio
            warn_log.append(f"{name}: {wmsg.message}")
        tgt = [x for x in ann if x[2] in SOUNDS]
        intf = [x for x in ann if x[2] == "interferer"]
        if contains != "none" and not tgt:
            raise SystemExit(f"{name}: Scaper produced no target event; the manifest would be unusable")
        t_on, t_off = (round(tgt[0][0], 3), round(tgt[0][1], 3)) if tgt else ("", "")
        i_on, i_off = (round(intf[0][0], 3), round(intf[0][1], 3)) if intf else ("", "")
        overlap_s = round(max(0.0, min(t_off, i_off) - max(t_on, i_on)), 3) if tgt and intf else ""
        short = bool(tgt) and (t_off - t_on) < EVENT_DUR[0] - 0.01
        short_int = bool(intf) and (i_off - i_on) < 6.0 - 0.01      # interferers are drawn at 6-8 s
        rows.append({
            "clip": str(out / "clips" / f"{name}.wav"), "condition": condition, "contains": contains,
            "target_ebr_db": target_ebr, "interferer_ebr_db": interferer_ebr,
            "target_source": target_src, "target_onset": t_on, "target_offset": t_off,
            "interferer_source": interferer_src, "interferer_onset": i_on, "interferer_offset": i_off,
            "overlap_s": overlap_s, "short_event": short, "short_interferer": short_int, "background": bg.name, "seed": seed,
            "target_subset": subsets.get(target_src, "main") if target_src else ""})

    # Clips WITH a target sound
    for s in SOUNDS:
        for ri, rec in enumerate(targets[s]):
            chosen = [bgs[(ri * a.bgs_per_recording + j) % len(bgs)] for j in range(a.bgs_per_recording)]
            for bi, bg in enumerate(chosen):
                for cond in a.conditions:
                    if cond == "overlap":
                        intf = ints[(ri + bi) % len(ints)]
                        evs = [{"label": s, "file": rec, "snr": 0.0},
                               {"label": "interferer", "file": intf, "snr": 0.0,
                                "start": (0.0, 2.0), "dur": (6.0, 8.0)}]
                        make(f"{s}_{rec.stem}_{bg.stem}_overlap", f"{s}|{rec.name}|{bg.name}", bg, evs,
                             "overlap", s, rec.name, intf.name, 0.0, 0.0)
                    else:
                        ebr = float(cond.replace("ebr", ""))
                        evs = [{"label": s, "file": rec, "snr": ebr}]
                        make(f"{s}_{rec.stem}_{bg.stem}_{cond}", f"{s}|{rec.name}|{bg.name}", bg, evs,
                             cond, s, rec.name, "", ebr, "")

    # Target-absent clips
    for bi, bg in enumerate(bgs):
        make(f"none_{bg.stem}_bgonly", f"none|{bg.name}", bg, [], "bgonly", "none")
        for n in range(a.neg_per_bg):
            intf = ints[(bi + n) % len(ints)]
            evs = [{"label": "interferer", "file": intf, "snr": 0.0, "start": (0.0, 2.0), "dur": (6.0, 8.0)}]
            make(f"none_{bg.stem}_int_{intf.stem}", f"none|{bg.name}|{intf.name}", bg, evs,
                 "int", "none", "", intf.name, "", 0.0)

    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "settings.json").write_text(json.dumps({
        "scaper_version": scaper.__version__, "duration_s": DUR, "sample_rate": SR, "ref_db_lufs": REF_DB,
        "conditions": a.conditions, "event_duration_s": EVENT_DUR, "event_start_s": EVENT_START,
        "bgs_per_recording": a.bgs_per_recording, "neg_per_bg": a.neg_per_bg, "base_seed": a.seed,
        "seed_rule": "base_seed + crc32(clip identity); shared across conditions for the same sound|recording|background",
        "targets": {s: [f.name for f in fs] for s, fs in targets.items()},
        "target_set": a.targets, "background_set": a.backgrounds,
        "split_csv_sha256": sha256(a.split_csv) if a.targets != "dev" else None,
        "limitation": "interferers/ is shared by all sets (not held out); development tuning used it",
        "backgrounds": [b.name for b in bgs], "interferers": [i.name for i in ints]}, indent=2))

    (out / "scaper_warnings.txt").write_text("\n".join(warn_log))

    # Pairing check: same recording + background must have the same target timing in every condition
    timing = {}
    unpaired = 0
    for r in rows:
        if r["contains"] == "none":
            continue
        key = (r["contains"], r["target_source"], r["background"])
        t = (r["target_onset"], r["target_offset"])
        if key in timing and timing[key] != t:
            unpaired += 1
        timing.setdefault(key, t)
    n_short = sum(r["short_event"] for r in rows)
    n_pos = sum(r["contains"] != "none" for r in rows)
    print(f"Made {len(rows)} clips: {n_pos} with a target, {len(rows) - n_pos} without")
    for s in SOUNDS:
        print(f"  {s:11} {sum(r['contains'] == s for r in rows)} clips")
    for c in sorted({r["condition"] for r in rows}):
        print(f"  condition {c:8} {sum(r['condition'] == c for r in rows)} clips")
    print(f"Scaper warnings: {len(warn_log)} (see scaper_warnings.txt); target events shorter than "
          f"{EVENT_DUR[0]} s: {n_short}; short interferers: {sum(r['short_interferer'] for r in rows)}")
    print(f"Pairing check (same target timing across conditions): "
          f"{'OK' if unpaired == 0 else f'{unpaired} clips differ - conditions are NOT paired'}")
    print(f"Saved in {out}  (manifest.csv, settings.json)")


if __name__ == "__main__":
    main()