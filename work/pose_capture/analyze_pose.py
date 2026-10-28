import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FONT_PATH = Path("/System/Library/Fonts/STHeiti Light.ttc")


def latest_session():
    sessions = sorted(OUTPUT_DIR.glob("pose_session_*.csv"))
    if not sessions:
        raise FileNotFoundError("No pose_session_*.csv file was found.")
    return sessions[-1]


def load_session(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return rows


def moving_average(values, window=9):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def extract_motion(rows):
    time_s = []
    left_raise = []
    right_raise = []
    detected = 0

    for row in rows:
        if row["pose_detected"] != "1":
            continue
        detected += 1

        required = (
            "left_shoulder_y",
            "left_wrist_y",
            "left_hip_y",
            "right_shoulder_y",
            "right_wrist_y",
            "right_hip_y",
            "left_wrist_visibility",
            "right_wrist_visibility",
        )
        if any(not row[name] for name in required):
            continue
        if min(
            float(row["left_wrist_visibility"]),
            float(row["right_wrist_visibility"]),
        ) < 0.35:
            continue

        left_torso = abs(float(row["left_hip_y"]) - float(row["left_shoulder_y"]))
        right_torso = abs(float(row["right_hip_y"]) - float(row["right_shoulder_y"]))
        if left_torso < 0.04 or right_torso < 0.04:
            continue

        time_s.append(float(row["elapsed_s"]))
        left_raise.append(
            (float(row["left_shoulder_y"]) - float(row["left_wrist_y"]))
            / left_torso
        )
        right_raise.append(
            (float(row["right_shoulder_y"]) - float(row["right_wrist_y"]))
            / right_torso
        )

    return (
        np.asarray(time_s),
        np.asarray(left_raise),
        np.asarray(right_raise),
        detected,
    )


def render_plot(time_s, left_raise, right_raise, source_path):
    left_smooth = moving_average(left_raise)
    right_smooth = moving_average(right_raise)

    font = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=180)
    fig.patch.set_facecolor("#F7F5EF")
    ax.set_facecolor("#F7F5EF")

    ax.plot(
        time_s,
        right_smooth,
        color="#176B87",
        linewidth=2.8,
        label="右手腕",
    )
    ax.plot(
        time_s,
        left_smooth,
        color="#D26A3A",
        linewidth=2.2,
        label="左手腕",
    )
    ax.axhline(
        0,
        color="#4A4A4A",
        linewidth=1.2,
        linestyle=(0, (5, 4)),
    )
    ax.text(
        time_s.max() if len(time_s) else 1,
        0.05,
        "肩部高度",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
        fontproperties=font,
    )

    ax.set_xlabel("时间 / 秒", fontsize=11, fontproperties=font)
    ax.set_ylabel("腕部相对肩部高度", fontsize=11, fontproperties=font)
    ax.set_title(
        "双臂上举测试：腕部运动轨迹",
        fontsize=17,
        fontweight="bold",
        loc="left",
        pad=16,
        fontproperties=font,
    )
    ax.grid(axis="y", color="#DED9CD", linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9C978D")
    ax.tick_params(colors="#555555", labelsize=9)
    ax.legend(
        loc="upper right",
        frameon=False,
        prop=font,
        fontsize=10,
        ncols=2,
    )

    left_amp = float(np.ptp(left_smooth)) if len(left_smooth) else 0
    right_amp = float(np.ptp(right_smooth)) if len(right_smooth) else 0
    balance_gap = (
        float(np.mean(np.abs(left_smooth - right_smooth)))
        if len(left_smooth)
        else 0
    )
    summary = (
        f"左臂幅度 {left_amp:.2f}   "
        f"右臂幅度 {right_amp:.2f}   "
        f"左右平均差 {balance_gap:.2f}"
    )
    fig.text(
        0.125,
        0.015,
        summary,
        fontsize=9,
        color="#555555",
        fontproperties=font,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path = source_path.with_name(
        source_path.stem.replace("pose_session", "pose_trajectory") + ".png"
    )
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path, left_amp, right_amp, balance_gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, nargs="?")
    args = parser.parse_args()

    source_path = args.csv_path or latest_session()
    rows = load_session(source_path)
    time_s, left_raise, right_raise, detected = extract_motion(rows)

    if len(time_s) < 2:
        raise RuntimeError("Not enough valid wrist landmarks to create a plot.")

    output_path, left_amp, right_amp, balance_gap = render_plot(
        time_s,
        left_raise,
        right_raise,
        source_path,
    )

    print(f"Source: {source_path}")
    print(f"Saved: {output_path}")
    print(f"Detected frames: {detected}/{len(rows)}")
    print(
        f"Amplitude: left={left_amp:.3f}, right={right_amp:.3f}, "
        f"mean_gap={balance_gap:.3f}"
    )


if __name__ == "__main__":
    main()
