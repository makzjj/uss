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

Uses development data only:
  sources/<sound>/test/*      clean target recordings (beeper, engine, jackhammer)
  backgrounds/dev/*.flac      development backgrounds
  interferers/*               non-target foreground sounds

Writes to testsets/gate1_v2/:
  clips/<name>.wav            10 s, 32 kHz mono soundscapes
  events/<name>_events/       each foreground event and the background on their own (for SI-SDR)
  jams/<name>.jams            Scaper annotation
  manifest.csv                one row per clip: condition, target, onset/offset, sources, seed
Old clips in the output folder are removed first (only if it is inside a "testsets" folder),
so the set always matches this run.

Usage (inside the container, from the uss folder):
  python experiments/build_scaper_set.py
"""
import argparse
import csv
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_audio", default=TA)
    p.add_argument("--out_dir", default=f"{TA}/testsets/gate1_v2")
    p.add_argument("--conditions", nargs="+", default=["ebr0", "ebr-5", "overlap"])
    p.add_argument("--bgs_per_recording", type=int, default=5,
                   help="backgrounds each target recording is mixed with (rotated so all are used)")
    p.add_argument("--neg_per_bg", type=int, default=3,
                   help="background + interferer negatives per background (each a different interferer)")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    ta = Path(a.test_audio)
    out = Path(a.out_dir).resolve()
    if "testsets" not in out.parent.parts:
        raise SystemExit(f"Refusing to write to {out}: output must be a folder INSIDE 'testsets' "
                         "(e.g. testsets/gate1_v2), never 'testsets' itself or anywhere else")
    if out.exists():
        shutil.rmtree(out)
    for sub in ("clips", "events", "jams"):
        (out / sub).mkdir(parents=True)

    targets = {s: audio_files(ta / "sources" / s / "test") for s in SOUNDS}
    bgs = sorted((ta / "backgrounds" / "dev").glob("*.flac"))
    ints = audio_files(ta / "interferers")
    for s, fs in targets.items():
        if not fs:
            raise SystemExit(f"No test recordings for {s}")
    if not bgs or not ints:
        raise SystemExit("Missing backgrounds/dev or interferers")
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
            "overlap_s": overlap_s, "short_event": short, "short_interferer": short_int, "background": bg.name, "seed": seed})

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