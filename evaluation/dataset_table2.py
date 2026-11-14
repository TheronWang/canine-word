"""Reproduce Table 2 (video hours and extracted vocalization hours per breed).

Uses the released data: videos/<breed>_video_durations.jsonl.gz and
transcriptions_km250.jsonl.gz (one line per vocal segment).

Usage: python dataset_table2.py ../data
"""
import gzip
import json
import os
import sys
from collections import defaultdict

BREEDS = ["shiba_inu", "chihuahua", "husky", "pitbull", "labrador", "german_shepherd"]


def read_jsonl(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(data_dir):
    video_hours = {b: sum(d["duration"] for d in read_jsonl(os.path.join(data_dir, "videos", f"{b}_video_durations.jsonl.gz"))) / 3600
                   for b in BREEDS}
    vocal_hours = defaultdict(float)
    n_segments = 0
    for seg in read_jsonl(os.path.join(data_dir, "transcriptions_km250.jsonl.gz")):
        n_segments += 1
        vocal_hours[seg["breeds"][0]] += seg["duration"] / 3600

    print(f"{'Breed':<16}{'Videos (h)':>12}{'Vocal (h)':>12}")
    for b in BREEDS:
        print(f"{b:<16}{video_hours[b]:>12,.0f}{vocal_hours[b]:>12.0f}")
    print(f"{'Total':<16}{sum(video_hours.values()):>12,.0f}{sum(vocal_hours.values()):>12.0f}")
    print(f"vocal segments: {n_segments:,}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data")
