import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision


def make_detector(model_path: str):
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )
    return vision.HandLandmarker.create_from_options(options)


def main():
    model_path = "models/hand_landmarker.task"
    detector = make_detector(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    last_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        size = max(h, w)
        if h != w:
            pad_y = (size - h) // 2
            pad_x = (size - w) // 2
            square = np.zeros((size, size, 3), dtype=rgb.dtype)
            square[pad_y:pad_y + h, pad_x:pad_x + w] = rgb
            rgb_for_mp = square
        else:
            rgb_for_mp = rgb
        try:
            mp_image = mp.Image.create_from_numpy_array(rgb_for_mp)
        except Exception:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_for_mp)

        result = detector.detect(mp_image)
        n = len(result.hand_landmarks) if result and result.hand_landmarks else 0

        now = time.time()
        if now - last_print > 1.0:
            print(f"hands_detected={n}")
            last_print = now

        cv2.putText(frame, f"hands_detected={n}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Tasks Detect Probe", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
