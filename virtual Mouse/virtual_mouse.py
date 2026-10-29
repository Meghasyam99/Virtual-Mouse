import cv2
import numpy as np
import time
import math
import threading
import ctypes
import HandTracking as ht

# ===============================
# DIRECT WIN32 HIGH-SPEED ENGINE
# ===============================
user32 = ctypes.windll.user32

# Get primary screen dimensions directly from Windows
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

# Win32 Mouse & Keyboard Event Constants
MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004
MOUSEEVENTF_RIGHTDOWN  = 0x0008
MOUSEEVENTF_RIGHTUP    = 0x0010
MOUSEEVENTF_WHEEL      = 0x0800
VK_VOLUME_UP           = 0xAF
VK_VOLUME_DOWN         = 0xAE
KEYEVENTF_KEYUP        = 0x0002

def fast_move(x, y):
    """Instant 0.005ms cursor move directly in Windows kernel"""
    user32.SetCursorPos(int(x), int(y))

def fast_left_click():
    """Instant left click with zero framework delay"""
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def fast_right_click():
    """Instant right click with zero framework delay"""
    user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

def fast_scroll(delta):
    """Direct wheel scroll scaled for Windows applications"""
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(delta), 0)

def fast_volume_up():
    """Hardware media volume up key event"""
    user32.keybd_event(VK_VOLUME_UP, 0, 0, 0)
    user32.keybd_event(VK_VOLUME_UP, 0, KEYEVENTF_KEYUP, 0)

def fast_volume_down():
    """Hardware media volume down key event"""
    user32.keybd_event(VK_VOLUME_DOWN, 0, 0, 0)
    user32.keybd_event(VK_VOLUME_DOWN, 0, KEYEVENTF_KEYUP, 0)

def fast_minimize():
    """Instant 0ms minimize for active application window, or Win+D if on camera/desktop"""
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    buff = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buff, length + 1)
    title = buff.value

    # If foreground window is not our camera window, minimize it directly
    if hwnd and "Virtual Mouse" not in title:
        user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE = 6
    else:
        # If camera window or desktop is active, send Win + D to show/toggle desktop
        user32.keybd_event(0x5B, 0, 0, 0)       # Win down
        user32.keybd_event(0x44, 0, 0, 0)       # D down
        user32.keybd_event(0x44, 0, 0x0002, 0)  # D up
        user32.keybd_event(0x5B, 0, 0x0002, 0)  # Win up


# ===============================
# ZERO-LAG THREADED CAMERA
# ===============================
class FastCamera:
    """Decouples camera I/O from inference for maximum FPS and zero buffer lag"""
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.002)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        time.sleep(0.05)
        if self.cap.isOpened():
            self.cap.release()


# ===============================
# CONFIGURATION & TUNING
# ===============================
width, height = 640, 480
frameR = 70             # Working area margin (smaller margin = higher sensitivity)
speed_multiplier = 1.15 # Cursor speed boost factor

prev_x, prev_y = None, None
curr_x, curr_y = 0, 0

last_click_time = 0
click_debounce = 0.18   # Snappy 180ms debounce for quick clicks

prev_scroll_y = None
prev_vol_y = None
right_clicked = False
fist_closed = False
last_minimize_time = 0
minimize_debounce = 0.6   # 600ms debounce for fist minimize

cam = FastCamera(src=0, width=width, height=height)
detector = ht.handDetector(maxHands=1, detectionCon=0.35, trackCon=0.35)

pTime = time.perf_counter()

# ===============================
# MAIN REAL-TIME LOOP
# ===============================
while True:
    success, img = cam.read()
    if not success or img is None:
        time.sleep(0.002)
        continue

    # Instant mirror flip
    img = cv2.flip(img, 1)
    h, w = img.shape[:2]

    # Hand detection & landmarks
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img, draw=True, drawBbox=True)

    # Draw active working area rectangle
    cv2.rectangle(img, (frameR, frameR),
                  (w - frameR, h - frameR),
                  (255, 0, 255), 2)

    mode = "Idle"

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]   # Index fingertip
        x2, y2 = lmlist[12][1:]  # Middle fingertip

        fingers = detector.fingersUp()
        if not fingers or len(fingers) != 5:
            fingers = [0, 0, 0, 0, 0]

        # Adaptive pinch & click thresholds based on hand distance from camera
        xmin, ymin, xmax, ymax = bbox if bbox and len(bbox) == 4 else (None, None, None, None)
        if xmin is not None:
            hand_size = max(max(1, xmax - xmin), max(1, ymax - ymin))
        else:
            hand_size = None

        click_thresh = 35 if hand_size is None else int(max(24, min(60, 0.15 * hand_size)))
        pinch_thresh = 45 if hand_size is None else int(max(30, min(85, 0.22 * hand_size)))

        # ==============================================================
        # 1. VOLUME CONTROL (Thumb + Index pinch, other fingers down)
        # ==============================================================
        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            mode = "Volume Pinch"
            length, img, line_info = detector.findDistance(4, 8, img, draw=True)

            if length < pinch_thresh:
                cv2.circle(img, (line_info[4], line_info[5]), 12, (0, 255, 255), cv2.FILLED)
                if prev_vol_y is not None:
                    diff = y1 - prev_vol_y
                    if diff < -6:
                        fast_volume_up()
                        mode = "Volume UP ▲"
                        prev_vol_y = y1
                    elif diff > 6:
                        fast_volume_down()
                        mode = "Volume DOWN ▼"
                        prev_vol_y = y1
                else:
                    prev_vol_y = y1
            else:
                prev_vol_y = None

            prev_scroll_y = None
            right_clicked = False
            fist_closed = False

        # ==============================================================
        # 2. FAST CONTINUOUS SCROLL (4 or 5 fingers up)
        # ==============================================================
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            mode = "Scroll Ready"
            if prev_scroll_y is not None:
                diff = y2 - prev_scroll_y
                if abs(diff) > 4:
                    # High-speed responsive scroll multiplier
                    scroll_power = int(abs(diff) * 12)
                    steps = int(-np.sign(diff) * max(120, scroll_power))
                    fast_scroll(steps)
                    mode = "Scroll UP ▲" if steps > 0 else "Scroll DOWN ▼"

            prev_scroll_y = y2
            prev_vol_y = None
            right_clicked = False
            fist_closed = False

        # ==============================================================
        # 3. INSTANT RIGHT CLICK (3 fingers up: Index, Middle, Ring)
        # ==============================================================
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
            mode = "Right Click Ready"
            now = time.time()
            if not right_clicked and (now - last_click_time > click_debounce):
                fast_right_click()
                right_clicked = True
                last_click_time = now
                mode = "Right Click FIRED!"

            prev_scroll_y = None
            prev_vol_y = None
            fist_closed = False

        # ==============================================================
        # 4. INSTANT LEFT CLICK (2 fingers up: Index & Middle)
        # ==============================================================
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            mode = "Aiming (Left Click)"
            length, img, line_info = detector.findDistance(8, 12, img, draw=True)

            # Keep cursor moving with low latency while aiming
            x3 = np.interp(x1, (frameR, w - frameR), (0, screen_width)) * speed_multiplier
            y3 = np.interp(y1, (frameR, h - frameR), (0, screen_height)) * speed_multiplier
            x3 = max(0, min(screen_width - 1, x3))
            y3 = max(0, min(screen_height - 1, y3))

            if prev_x is None:
                prev_x, prev_y = x3, y3

            curr_x = prev_x + (x3 - prev_x) / 1.8
            curr_y = prev_y + (y3 - prev_y) / 1.8
            fast_move(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Contact trigger
            if length < click_thresh:
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                now = time.time()
                if now - last_click_time > click_debounce:
                    fast_left_click()
                    last_click_time = now
                    mode = "Left Click FIRED!"

            prev_scroll_y = None
            prev_vol_y = None
            right_clicked = False
            fist_closed = False

        # ==============================================================
        # 5. HIGH-SPEED DYNAMIC CURSOR MOVE (Only Index finger up)
        # ==============================================================
        elif fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            mode = "Cursor Move"

            # Screen coordinate interpolation with speed multiplier
            x3 = np.interp(x1, (frameR, w - frameR), (0, screen_width)) * speed_multiplier
            y3 = np.interp(y1, (frameR, h - frameR), (0, screen_height)) * speed_multiplier
            x3 = max(0, min(screen_width - 1, x3))
            y3 = max(0, min(screen_height - 1, y3))

            if prev_x is None:
                prev_x, prev_y = x3, y3

            # Calculate pixel jump distance on screen
            dist = math.hypot(x3 - prev_x, y3 - prev_y)

            # DYNAMIC VELOCITY-ADAPTIVE SMOOTHING:
            # - High speed flick: smooth = 1.1 (instant zero-lag tracking)
            # - Normal motion: smooth = 1.6 (fast and direct)
            # - Fine hovering: smooth = 3.2 (rock solid, no jitter)
            if dist > 35:
                adaptive_smooth = 1.1
            elif dist > 12:
                adaptive_smooth = 1.6
            elif dist > 3:
                adaptive_smooth = 2.8
            else:
                adaptive_smooth = 4.0
                x3, y3 = prev_x, prev_y  # Anti-tremor deadband

            curr_x = prev_x + (x3 - prev_x) / adaptive_smooth
            curr_y = prev_y + (y3 - prev_y) / adaptive_smooth

            fast_move(curr_x, curr_y)
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)

            prev_x, prev_y = curr_x, curr_y
            prev_scroll_y = None
            prev_vol_y = None
            right_clicked = False
            fist_closed = False

        # ==============================================================
        # 6. MINIMIZE OPTION (Five fingers close at once / Fist)
        # ==============================================================
        elif fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            mode = "Fist (Minimize)"
            now = time.time()
            if not fist_closed and (now - last_minimize_time > minimize_debounce):
                fast_minimize()
                fist_closed = True
                last_minimize_time = now
                mode = "MINIMIZED! [Win]"

            # Draw visual feedback indicator on fist (knuckle center)
            if len(lmlist) > 9:
                cx, cy = lmlist[9][1], lmlist[9][2]
                cv2.circle(img, (cx, cy), 18, (0, 165, 255), cv2.FILLED)
                cv2.putText(img, "MINIMIZE", (cx - 45, cy - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            prev_x, prev_y = None, None
            prev_scroll_y = None
            prev_vol_y = None
            right_clicked = False

        else:
            prev_scroll_y = None
            prev_vol_y = None
            right_clicked = False
            fist_closed = False

    else:
        # Hand tracking lost; reset coordinate cache to avoid jumping upon return
        prev_x, prev_y = None, None
        prev_scroll_y = None
        prev_vol_y = None
        right_clicked = False
        fist_closed = False
        mode = "No Hand Detected"

    # ===============================
    # TOP STATUS HUD & REAL-TIME FPS
    # ===============================
    cTime = time.perf_counter()
    fps = int(1.0 / (cTime - pTime)) if (cTime - pTime) > 0 else 0
    pTime = cTime

    # Dark banner bar at top
    cv2.rectangle(img, (0, 0), (w, 42), (25, 25, 25), cv2.FILLED)
    cv2.putText(img, f'FPS: {fps}', (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(img, f'MODE: {mode}', (145, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.imshow("Virtual Mouse (High Speed)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
