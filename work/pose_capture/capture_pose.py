from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Shoulders, elbows, wrists, hips, knees, and ankles.
TRACKED_LANDMARKS = {
    11: "L shoulder",
    12: "R shoulder",
    13: "L elbow",
    14: "R elbow",
    15: "L wrist",
    16: "R wrist",
    23: "L hip",
    24: "R hip",
    25: "L knee",
    26: "R knee",
    27: "L ankle",
    28: "R ankle",
}


def draw_interface(image, landmarks):
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width, 54), (20, 20, 20), -1)
    cv2.putText(
        image,
        "S: save pose image    Q: quit",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    if not landmarks:
        cv2.putText(
            image,
            "No pose detected - stand back and face the camera",
            (18, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 210, 255),
            2,
            cv2.LINE_AA,
        )
        return

    for index, name in TRACKED_LANDMARKS.items():
        landmark = landmarks.landmark[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        color = (80, 230, 110) if index <= 16 else (255, 170, 70)
        cv2.circle(image, (x, y), 13, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.circle(image, (x, y), 10, color, -1, cv2.LINE_AA)

        if index in {11, 13, 15, 23, 25, 27}:
            cv2.putText(
                image,
                name,
                (x + 15, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )


def save_pose_image(image):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"pose_screenshot_{timestamp}.png"

    if cv2.imwrite(str(output_path), image):
        print(f"Saved: {output_path}")
    else:
        print(f"Save failed: {output_path}")


def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

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

    print("Pose capture is ready.")
    print("Press S to save a screenshot and Q to quit.")

    empty_frames = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                empty_frames += 1
                if empty_frames == 1:
                    print("Waiting for the camera to provide its first frame...")
                if empty_frames == 100:
                    print("Still waiting for camera permission or another app to release it.")
                if empty_frames >= 200:
                    raise RuntimeError(
                        "The camera did not provide a frame. Close other camera apps "
                        "and check camera permission for VS Code or Terminal."
                    )
                cv2.waitKey(50)
                continue

            empty_frames = 0

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            draw_interface(frame, results.pose_landmarks)
            cv2.imshow("WUQINXI - MediaPipe Pose Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_pose_image(frame)
            elif key == ord("q"):
                break
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
