"""Combine results/*_summary.json into one S1-S3 table for Gate 2 (results/pi_summary.csv)."""
import csv
import glob
import json

rows = []
for f in sorted(glob.glob("results/*_summary.json")):
    d = json.load(open(f))
    rows.append({
        "file": f.split("/")[-1].replace("_summary.json", ""),
        "stage": d.get("stage"), "k": d.get("k"), "threads": d.get("threads"),
        "cpus": "".join(map(str, d.get("cpus", []))), "runs": d.get("runs_completed", d.get("repeats")),
        "median_ms": d.get("median_ms"), "p95_ms": d.get("p95_ms"), "rtf_median": d.get("rtf_median"),
        "deadline_p95_s": d.get("deadline_p95_s", ""), "meets_6s": d.get("meets_6s_realtime", ""),
        "hop_overruns": d.get("hop_overruns", ""), "peak_rss_mb": d.get("peak_rss_mb"),
        "mem_avail_min_mb": d.get("mem_available_min_mb", ""), "temp_max_c": d.get("temp_max_c"),
        "valid": d.get("valid_run", ""),
    })

if not rows:
    raise SystemExit("No summaries found in results/")
with open("results/pi_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

cols = ["file", "median_ms", "p95_ms", "rtf_median", "deadline_p95_s", "meets_6s", "hop_overruns", "temp_max_c", "valid"]
print(" | ".join(cols))
for r in rows:
    print(" | ".join(str(r[c]) for c in cols))
print("\nSaved results/pi_summary.csv")