import json
import logging
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRANSCRIPTION_PATH = "transcription.json"       # path to transcription JSON {clip_id: [int, ...]}
OUTPUT_PATH        = "./phone_duration_statistics.png"
HUBERT_FPS = 50.0  # 20 ms per frame
DELIMITER = -1


def compute_phone_stats(transcription: dict) -> dict:
    phone_durations = defaultdict(list)

    for frames in transcription.values():
        if not frames:
            continue
        # Combine consecutive identical frames into phone runs
        run_start = 0
        for i in range(1, len(frames)):
            if frames[i] != frames[i - 1]:
                phone = frames[run_start]
                if phone != DELIMITER:
                    duration = (i - run_start) / HUBERT_FPS
                    phone_durations[phone].append(duration)
                run_start = i
        # Last run
        phone = frames[run_start]
        if phone != DELIMITER:
            duration = (len(frames) - run_start) / HUBERT_FPS
            phone_durations[phone].append(duration)

    phone_stats = {}
    for phone, durations in sorted(phone_durations.items()):
        arr = np.array(durations)
        phone_stats[phone] = {
            "mean": np.mean(arr),
            "median": np.median(arr),
            "std": np.std(arr),
            "min": np.min(arr),
            "max": np.max(arr),
            "p95": np.percentile(arr, 95),
            "count": len(arr),
        }
        logger.info(
            f"Phone {phone:3d}: count={len(arr):>8,}, mean={np.mean(arr):.3f}s, "
            f"median={np.median(arr):.3f}s, std={np.std(arr):.3f}s, "
            f"95th={np.percentile(arr, 95):.3f}s"
        )

    return phone_stats


def plot_candlestick(phone_stats: dict, output_path: str):
    phones = sorted(phone_stats.keys())
    if not phones:
        return

    means    = [phone_stats[p]["mean"]   for p in phones]
    medians  = [phone_stats[p]["median"] for p in phones]
    stds     = [phone_stats[p]["std"]    for p in phones]
    mins     = [phone_stats[p]["min"]    for p in phones]
    max_95th = [phone_stats[p]["p95"]    for p in phones]

    fig, ax = plt.subplots(figsize=(20, 10))
    width = 0.4

    for i, phone in enumerate(phones):
        mean     = float(means[i])
        median   = float(medians[i])
        std      = float(stds[i])
        min_val  = float(mins[i])
        max_val  = float(max_95th[i])

        body_bottom = float(max(min_val, mean - std))
        body_top    = float(min(max_val, mean + std))
        hw = width / 2   # half-width
        qw = width / 4   # quarter-width

        # Whisker
        ax.plot([i, i], [min_val, max_val], color="k", linewidth=2, zorder=1)

        # Body outline (4 sides)
        ax.plot([i - hw, i + hw], [body_bottom, body_bottom], color="darkblue", linewidth=1.5, zorder=3)
        ax.plot([i - hw, i + hw], [body_top,    body_top],    color="darkblue", linewidth=1.5, zorder=3)
        ax.plot([i - hw, i - hw], [body_bottom, body_top],    color="darkblue", linewidth=1.5, zorder=3)
        ax.plot([i + hw, i + hw], [body_bottom, body_top],    color="darkblue", linewidth=1.5, zorder=3)
        # Body fill via many horizontal lines
        n_fill = max(2, int((body_top - body_bottom) / 0.001))
        for y in np.linspace(body_bottom, body_top, n_fill):
            ax.plot([i - hw, i + hw], [float(y), float(y)],
                    color="steelblue", alpha=0.15, linewidth=1, zorder=2)

        # Mean line
        ax.plot([i - hw, i + hw], [mean, mean],
                color="r", linewidth=3, zorder=4,
                label="Mean" if i == 0 else "")

        # Median diamond
        ax.plot(i, median, "D",
                color="gold", markersize=10,
                markeredgecolor="darkorange", markeredgewidth=2,
                zorder=5, label="Median" if i == 0 else "")

        # Min/max ticks
        ax.plot([i - qw, i + qw], [min_val, min_val], color="k", linewidth=2.5, zorder=3)
        ax.plot([i - qw, i + qw], [max_val, max_val], color="k", linewidth=2.5, zorder=3)

    ax.set_xlabel("Phone ID", fontweight="bold", fontsize=24)
    ax.set_ylabel("Duration (seconds)", fontweight="bold", fontsize=24)
    ax.set_title("Phone Duration Statistics", fontsize=28, fontweight="bold", pad=20)
    ax.set_xticks(np.arange(len(phones)))
    ax.set_xticklabels(phones, rotation=45, ha="right", fontsize=20)
    ax.tick_params(axis="y", labelsize=20)

    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color="r", linewidth=3, label="Mean"),
        plt.Line2D([0], [0], marker="D", color="gold", markersize=10,
                   markeredgecolor="darkorange", markeredgewidth=2,
                   linestyle="", label="Median"),
        Patch(facecolor="steelblue", edgecolor="darkblue", alpha=0.7, label="Mean ± Std Dev"),
        plt.Line2D([0], [0], color="k", linewidth=2, label="Min–95th Percentile"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=22,
              frameon=True, fancybox=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved candlestick figure to {output_path}")


def main():
    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    logger.info(f"Loading transcription from {TRANSCRIPTION_PATH}")
    with open(TRANSCRIPTION_PATH) as f:
        transcription = json.load(f)
    logger.info(f"Loaded {len(transcription):,} segments")

    phone_stats = compute_phone_stats(transcription)
    logger.info(f"Computed stats for {len(phone_stats)} phones")

    plot_candlestick(phone_stats, OUTPUT_PATH)


if __name__ == "__main__":
    main()
