"""Summarise a run_gate1.py scores file (e.g. results/gate1_v2_scores.csv) without depending on one cut-off.

For each sound, before and after USS:
  AUC   = chance that a random clip WITH the sound scores higher than a random clip
          WITHOUT it (1.0 = perfect ranking, 0.5 = guessing). Needs no cut-off.
  hits / false alarms at 0.2 (same value as USS's class-id mode) and at 0.01.

Usage: python experiments/gate1_summary.py [--csv results/gate1_v2_scores.csv]
       (the pilot set's older results are in results/gate1_pilot.csv)
"""
import argparse
import csv
from collections import defaultdict


def auc(pos, neg):
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/gate1_v2_scores.csv")
    a = ap.parse_args()

    data = defaultdict(list)
    with open(a.csv) as fh:
        for r in csv.DictReader(fh):
            data[r["sound"]].append(r)

    print(f"{'sound':12}{'when':8}{'AUC':>6}{'hits@0.2':>10}{'FA@0.2':>8}{'hits@0.01':>11}{'FA@0.01':>9}   (all conditions)")
    for sound, rows in data.items():
        for when in ("before", "after"):
            col = f"score_{when}"
            pos = [float(r[col]) for r in rows if r["has_sound"] == "1"]
            neg = [float(r[col]) for r in rows if r["has_sound"] == "0"]
            h2 = sum(s >= 0.2 for s in pos); f2 = sum(s >= 0.2 for s in neg)
            h1 = sum(s >= 0.01 for s in pos); f1 = sum(s >= 0.01 for s in neg)
            print(f"{sound:12}{when:8}{auc(pos, neg):6.2f}{h2:>6}/{len(pos):<3}{f2:>4}/{len(neg):<3}"
                  f"{h1:>7}/{len(pos):<3}{f1:>5}/{len(neg):<3}")
    by_condition(data)


def by_condition(data):
    """AUC per condition: that condition's clips WITH the sound vs ALL clips without it."""
    print(f"\nAUC by condition (positives of that condition vs all negatives):")
    print(f"{'sound':12}{'condition':12}{'n_pos':>6}{'n_neg':>6}{'AUC before':>12}{'AUC after':>11}")
    for sound, rows in data.items():
        neg = [r for r in rows if r["has_sound"] == "0"]
        conds = sorted({r.get("condition", "") for r in rows if r["has_sound"] == "1"})
        for c in conds:
            pos = [r for r in rows if r["has_sound"] == "1" and r.get("condition", "") == c]
            if not pos or not neg:
                continue
            ab = auc([float(r["score_before"]) for r in pos], [float(r["score_before"]) for r in neg])
            aa = auc([float(r["score_after"]) for r in pos], [float(r["score_after"]) for r in neg])
            print(f"{sound:12}{c or 'all':12}{len(pos):>6}{len(neg):>6}{ab:>12.2f}{aa:>11.2f}")


if __name__ == "__main__":
    main()