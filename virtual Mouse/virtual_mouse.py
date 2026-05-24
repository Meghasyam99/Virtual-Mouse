import cv2
import numpy as np
import time
import HandTracking as ht
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# ===============================
# VARIABLES
# ===============================
width, height = 640, 480
frameR = 50
smoothening = 8

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

last_click_time = 0
gesture_delay = 0.4

prev_scroll_y = None
prev_vol_y = None

movement_threshold = 5  # anti-jitter

# Camera (more reliable backend fallback on Windows)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

detector = ht.handDetector(maxHands=1, detectionCon=0.2, trackCon=0.2)

screen_width, screen_height = pyautogui.size()

pTime = 0

# ===============================
# MAIN LOOP
# ===============================
while True:
    success, img = cap.read()
    if not success:
        continue

    h, w = img.shape[:2]

    img = detector.findHands(img)
    # Draw landmarks + bbox again (but no debug text overlays).
    lmlist, bbox = detector.findPosition(img, draw=True, drawBbox=True)

    # Draw working area (inner screen)
    cv2.rectangle(img, (frameR, frameR),
                  (w - frameR, h - frameR),
                  (255, 0, 255), 2)

    if len(lmlist) != 0:

        x1, y1 = lmlist[8][1:]   # Index
        x2, y2 = lmlist[12][1:]  # Middle

        fingers = detector.fingersUp()
        if not fingers or len(fingers) != 5:
            fingers = [0, 0, 0, 0, 0]

        # Use bbox (already computed) for adaptive thresholds.
        xmin, ymin, xmax, ymax = bbox if bbox and len(bbox) == 4 else (None, None, None, None)
        if xmin is not None:
            hand_w = max(1, xmax - xmin)
            hand_h = max(1, ymax - ymin)
            hand_size = max(hand_w, hand_h)
        else:
            hand_size = None

        click_thresh = 35 if hand_size is None else int(max(25, min(70, 0.15 * hand_size)))
        pinch_thresh = 50 if hand_size is None else int(max(35, min(120, 0.25 * hand_size)))

        # ===============================
        # 1. CURSOR MOVE (ANTI-JITTER)
        # ===============================
        if fingers[1] == 1 and fingers[2] == 0:

            x3 = np.interp(x1, (frameR, w-frameR), (0, screen_width))
            y3 = np.interp(y1, (frameR, h-frameR), (0, screen_height))

            dx = x3 - prev_x
            dy = y3 - prev_y

            if abs(dx) < movement_threshold:
                x3 = prev_x
            if abs(dy) < movement_threshold:
                y3 = prev_y

            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            pyautogui.moveTo(screen_width - curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y
            prev_scroll_y = None
            prev_vol_y = None

        # ===============================
        # 2. LEFT CLICK
        # ===============================
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:

            length, _, _ = detector.findDistance(8, 12, img, draw=False)

            if length < click_thresh:
                if time.time() - last_click_time > gesture_delay:
                    pyautogui.click()
                    last_click_time = time.time()

            prev_scroll_y = None
            prev_vol_y = None

        # ===============================
        # 3. RIGHT CLICK
        # ===============================
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:

            if time.time() - last_click_time > gesture_delay:
                pyautogui.rightClick()
                last_click_time = time.time()

            prev_scroll_y = None
            prev_vol_y = None

        # ===============================
        # 4. CONTINUOUS SCROLL
        # ===============================
        elif (fingers[1] == 1 and fingers[2] == 1 and
              fingers[3] == 1 and fingers[4] == 1):

            if prev_scroll_y is not None:
                diff = y2 - prev_scroll_y

                if abs(diff) > 10:
                    scroll_speed = int(diff * 1.5)
                    pyautogui.scroll(-scroll_speed)

            prev_scroll_y = y2
            prev_vol_y = None

        # ===============================
        # 5. VOLUME CONTROL (RELIABLE)
        # ===============================
        elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0:

            # pinch detection
            length, _, _ = detector.findDistance(4, 8, img, draw=False)

            if length < pinch_thresh:

                if prev_vol_y is not None:
                    diff = y1 - prev_vol_y

                    if diff < -8:
                        pyautogui.scroll(100)   # volume up

                    elif diff > 8:
                        pyautogui.scroll(-100)  # volume down

                prev_vol_y = y1

            prev_scroll_y = None

        else:
            prev_scroll_y = None
            prev_vol_y = None

        pass

    # ===============================
    # FPS DISPLAY
    # ===============================
    cTime = time.time()
    fps = int(1 / (cTime - pTime)) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {fps}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()