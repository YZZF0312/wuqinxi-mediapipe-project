import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

BODY_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
)


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open the camera. Check camera permission for VS Code or Terminal."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def read_frame(cap, empty_frames):
    success, frame = cap.read()
    if success:
        return cv2.flip(frame, 1), 0

    empty_frames += 1
    if empty_frames == 1:
        print("Waiting for the camera to provide its first frame...")
    if empty_frames >= 200:
        raise RuntimeError(
            "The camera did not provide a frame. Close other camera apps "
            "and check camera permission for VS Code or Terminal."
        )

    cv2.waitKey(50)
    return None, empty_frames


def draw_pose(image, landmarks):
    height, width = image.shape[:2]

    if landmarks:
        for start, end in BODY_CONNECTIONS:
            start_point = landmarks.landmark[start]
            end_point = landmarks.landmark[end]
            start_xy = (int(start_point.x * width), int(start_point.y * height))
            end_xy = (int(end_point.x * width), int(end_point.y * height))
            cv2.line(image, start_xy, end_xy, (235, 235, 235), 2, cv2.LINE_AA)

    for index in LANDMARKS.values():
        landmark = landmarks.landmark[index] if landmarks else None
        if not landmark:
            continue

        x = int(landmark.x * width)
        y = int(landmark.y * height)
        color = (90, 230, 110) if index <= 16 else (255, 170, 70)
        cv2.circle(image, (x, y), 13, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.circle(image, (x, y), 9, color, -1, cv2.LINE_AA)


def draw_header(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 60), (20, 20, 20), -1)
    cv2.putText(
        image,
        text,
        (18, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )


def landmark_row(elapsed_s, frame_index, landmarks):
    row = {
        "elapsed_s": f"{elapsed_s:.3f}",
        "frame": frame_index,
        "pose_detected": int(landmarks is not None),
    }

    for name, index in LANDMARKS.items():
        landmark = landmarks.landmark[index] if landmarks else None
        row[f"{name}_x"] = f"{landmark.x:.6f}" if landmark else ""
        row[f"{name}_y"] = f"{landmark.y:.6f}" if landmark else ""
        row[f"{name}_z"] = f"{landmark.z:.6f}" if landmark else ""
        row[f"{name}_visibility"] = (
            f"{landmark.visibility:.6f}" if landmark else ""
        )

    return row


def save_rows(rows, action):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"pose_session_{action}_{timestamp}.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--action", default="arm_raise")
    args = parser.parse_args()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = open_camera()

    print("Pose recording is ready.")
    print("Press SPACE to start, or Q to quit.")
    print("During recording, raise both arms overhead and lower them slowly.")

    empty_frames = 0
    recording = False
    countdown_start = 0.0
    record_start = 0.0
    frame_index = 0
    rows = []

    try:
        while cap.isOpened():
            frame, empty_frames = read_frame(cap, empty_frames)
            if frame is None:
                continue

            now = time.monotonic()
            if recording and now - record_start >= args.duration:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            landmarks = results.pose_landmarks

            draw_pose(frame, landmarks)

            if recording:
                elapsed_s = now - record_start
                frame_index += 1
                rows.append(landmark_row(elapsed_s, frame_index, landmarks))
                draw_header(
                    frame,
                    f"REC {elapsed_s:05.1f} / {args.duration:04.1f}s",
                )
            elif countdown_start > 0:
                remaining = 3 - int(now - countdown_start)
                if remaining > 0:
                    draw_header(frame, f"Starting in {remaining}...")
                else:
                    recording = True
                    record_start = now
            else:
                draw_header(frame, "SPACE: start 30s test    Q: quit")

            cv2.imshow("WUQINXI - Pose Data Recording", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" ") and not recording and countdown_start == 0:
                countdown_start = now
                print("Recording starts in 3 seconds.")

        if rows:
            output_path = save_rows(rows, args.action)
            detected = sum(int(row["pose_detected"]) for row in rows)
            detection_rate = detected / len(rows) * 100
            print(f"Saved: {output_path}")
            print(
                f"Frames: {len(rows)}, pose detected: {detected} "
                f"({detection_rate:.1f}%)"
            )
        else:
            print("No recording was saved.")
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
