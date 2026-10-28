"""Count how often each sound appears in SINGA:PURA.

Prints each label with:
  recordings = how many recordings contain it at least once
  events     = how many times it was labelled in total

Construction-related groups are marked with *.

Usage:
  py count_labels.py --labels_dir labels_public
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

NAMES = {
    "1-0": "Engine (unsure)", "1-1": "Small engine", "1-2": "Medium engine", "1-3": "Large engine",
    "2-0": "Machinery impact (unsure)", "2-1": "Rock drill", "2-2": "Jackhammer", "2-3": "Hoe ram",
    "2-4": "Pile driver",
    "3-0": "Non-machinery impact (unsure)", "3-1": "Glass breaking", "3-2": "Car crash", "3-3": "Explosion",
    "4-0": "Powered saw (unsure)", "4-1": "Chainsaw", "4-2": "Small/medium rotating saw",
    "4-3": "Large rotating saw",
    "5-0": "Alert signal (unsure)", "5-1": "Car horn", "5-2": "Car alarm", "5-3": "Siren",
    "5-4": "Reverse beeper",
    "6-0": "Music (unsure)", "6-1": "Stationary music", "6-2": "Mobile music",
    "7-0": "Human voice (unsure)", "7-1": "Talking", "7-2": "Shouting", "7-3": "Large crowd",
    "7-4": "Amplified speech", "7-5": "Singing",
    "8-0": "Human movement (unsure)", "8-1": "Footsteps", "8-2": "Clapping",
    "9-0": "Animal (unsure)", "9-1": "Dog barking", "9-2": "Bird chirping", "9-3": "Insect chirping",
    "10-0": "Water (unsure)", "10-1": "Hose pump",
    "11-0": "Weather (unsure)", "11-1": "Rain", "11-2": "Thunder", "11-3": "Wind",
    "12-0": "Brake (unsure)", "12-1": "Friction brake", "12-2": "Exhaust brake",
    "13-0": "Train (unsure)", "13-1": "Electric train",
    "0-0": "Nothing from the list heard", "0-1": "Screeching", "0-2": "Plastic crinkling",
    "0-3": "Cleaning", "0-4": "Gear",
}
CONSTRUCTION = ("1-", "2-", "4-")  # engines, machinery impact, powered saw
CONSTRUCTION_EXTRA = {"5-4"}       # reverse beeper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir", required=True)
    a = p.parse_args()

    recordings = defaultdict(set)
    events = defaultdict(int)
    for f in Path(a.labels_dir).rglob("*.csv"):
        with open(f, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                lab = (row.get("event_label") or "").strip()
                name = (row.get("filename") or "").strip()
                if lab:
                    recordings[lab].add(name)
                    events[lab] += 1

    rows = sorted(recordings, key=lambda l: len(recordings[l]), reverse=True)
    with open("label_counts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "name", "construction_related", "recordings", "events"])
        print(f"{'':2}{'label':6}{'name':32}{'recordings':>11}{'events':>8}")
        for lab in rows:
            c = lab.startswith(CONSTRUCTION) or lab in CONSTRUCTION_EXTRA
            nm = NAMES.get(lab, "?")
            print(f"{'*' if c else ' ':2}{lab:6}{nm:32}{len(recordings[lab]):>11}{events[lab]:>8}")
            w.writerow([lab, nm, "yes" if c else "", len(recordings[lab]), events[lab]])
    print("\n* = construction-related. Full table saved to label_counts.csv")


if __name__ == "__main__":
    main()