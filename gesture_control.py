"""
Gesture Controlled PC
=====================
Control your PC using hand gestures, tracked live from your webcam.

Pipeline: OpenCV (camera + display) -> MediaPipe Hands (landmark detection)
-> a small rule-based classifier (finger up/down states) -> PyAutoGUI (OS actions)

Run:
    python gesture_control.py

Quit any time with 'q' in the video window.
"""

import os
import time
import math
import platform
import subprocess
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyautogui
import screen_brightness_control as sbc

# Safety: moving the mouse to a screen corner aborts pyautogui actions.
pyautogui.FAILSAFE = True

# ---------------------------------------------------------------------------
# MODEL DOWNLOAD — automatically fetches the hand landmarker model if missing.
# ---------------------------------------------------------------------------

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def ensure_model():
    """Download the hand landmarker model if it doesn't exist locally."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"[setup] Downloading hand landmarker model to {MODEL_PATH} ...")
    print(f"        (this only happens once)")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[setup] Download complete ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CONFIG — remap gestures to actions here without touching the logic below.
# Each entry is a function that runs when the gesture is confidently detected.
# ---------------------------------------------------------------------------

def action_play_pause():
    pyautogui.press("space")

def action_volume_up():
    pyautogui.press("volumeup")

def action_volume_down():
    pyautogui.press("volumedown")

def action_mute():
    pyautogui.press("volumemute")

def action_next_slide():
    pyautogui.press("right")

def action_prev_slide():
    pyautogui.press("left")

def action_next_track():
    pyautogui.press("nexttrack")

def action_prev_track():
    pyautogui.press("prevtrack")

def action_screenshot():
    fname = f"screenshot_{int(time.time())}.png"
    pyautogui.screenshot(fname)
    print(f"[screenshot saved] {fname}")

def action_alt_tab():
    pyautogui.hotkey("alt", "tab")

def action_brightness_up():
    try:
        brightness = sbc.get_brightness()
        current = brightness[0] if brightness else 50
        new = min(current + 10, 100)
        sbc.set_brightness(new)
        print(f"[brightness] {current} -> {new}")
    except Exception as e:
        print(f"[brightness] could not adjust automatically ({e}). "
              f"On macOS you may need the 'brightness' CLI (brew install brightness).")

def action_brightness_down():
    try:
        brightness = sbc.get_brightness()
        current = brightness[0] if brightness else 50
        new = max(current - 10, 0)
        sbc.set_brightness(new)
        print(f"[brightness] {current} -> {new}")
    except Exception as e:
        print(f"[brightness] could not adjust automatically ({e}). "
              f"On macOS you may need the 'brightness' CLI (brew install brightness).")

def _launch(candidates, app_name):
    """Try each candidate command list until one launches without error."""
    for cmd in candidates:
        try:
            subprocess.Popen(cmd, start_new_session=True)
            print(f"[launch] opened {app_name}")
            return
        except (FileNotFoundError, OSError):
            continue
    print(f"[launch] could not find {app_name} on PATH — "
          f"edit the candidate commands in this action to point at your install.")

def action_open_chrome():
    system = platform.system()
    if system == "Windows":
        candidates = [["cmd", "/c", "start", "chrome"]]
    elif system == "Darwin":
        candidates = [["open", "-a", "Google Chrome"]]
    else:
        candidates = [["google-chrome"], ["google-chrome-stable"], ["chromium-browser"], ["chromium"]]
    _launch(candidates, "Chrome")

def action_open_vscode():
    system = platform.system()
    if system == "Windows":
        candidates = [["cmd", "/c", "code"]]
    else:
        candidates = [["code"]]
    _launch(candidates, "VS Code")


# Each gesture below has a UNIQUE combination of extended fingers (a distinct
# "bit pattern"), so no two gestures can be confused for each other — see
# classify_gesture() for the exact pattern each name corresponds to.
GESTURE_ACTIONS = {
    "fist":        ("Screenshot",        action_screenshot),
    "open_palm":   ("Play / Pause",      action_play_pause),
    "point_up":    ("Alt+Tab",           action_alt_tab),
    "peace":       ("Next Slide",        action_next_slide),
    "three":       ("Previous Slide",    action_prev_slide),
    "four":        ("Open Chrome",       action_open_chrome),
    "thumbs_up":   ("Volume Up",         action_volume_up),
    "thumbs_down": ("Volume Down",       action_volume_down),
    "pinky":       ("Previous Track",    action_prev_track),
    "rock_on":     ("Next Track",        action_next_track),
    "shaka":       ("Brightness Up",     action_brightness_up),
    "gun":         ("Brightness Down",   action_brightness_down),
    "ok_sign":     ("Open VS Code",      action_open_vscode),
}

# How long (seconds) a gesture must be held steadily before it fires once.
HOLD_TIME = 0.5
# Minimum time between two firings of the SAME gesture (prevents spam).
COOLDOWN = 1.0

# ---------------------------------------------------------------------------
# MOUSE MODE CONFIG
# ---------------------------------------------------------------------------
# EMA smoothing factor: 0.0 = frozen, 1.0 = raw/jittery. 0.3 is a good balance.
MOUSE_SMOOTHING = 0.3
# Minimum normalized hand movement to register as cursor motion.
MOUSE_DEADZONE = 0.005
# Pixels to scroll per frame when in scroll mode.
SCROLL_SPEED = 15
# Fraction of frame edges to ignore so you don't need to reach absolute corners.
MOUSE_FRAME_MARGIN = 0.1
# Number of consecutive frames a click gesture must be held before it fires.
# At ~30 fps, 3 frames ≈ 100 ms — fast enough to feel instant, but filters
# out single-frame glitches.
CLICK_DEBOUNCE_FRAMES = 3

# Ordered list for the on-screen gesture guide panel.
# Each entry: (gesture_key, emoji, short_label)
GESTURE_GUIDE = [
    ("fist",        "Fist",           "Screenshot"),
    ("open_palm",   "Open Palm",      "Play/Pause"),
    ("point_up",    "Point Up",       "Alt+Tab"),
    ("peace",       "Peace",          "Next Slide"),
    ("three",       "Three",          "Prev Slide"),
    ("four",        "Four",           "Open Chrome"),
    ("thumbs_up",   "Thumb Up",       "Vol Up"),
    ("thumbs_down", "Thumb Down",     "Vol Down"),
    ("pinky",       "Pinky",          "Prev Track"),
    ("rock_on",     "Rock On",        "Next Track"),
    ("shaka",       "Shaka",          "Bright Up"),
    ("gun",         "Gun",            "Bright Down"),
    ("ok_sign",     "OK Sign",        "Open VS Code"),
]

# Guide entries shown when Mouse Mode is active.
MOUSE_GUIDE = [
    ("move",         "Index Only",     "Move Cursor"),
    ("left_click",   "Index + Thumb",  "Left Click/Drag"),
    ("right_click",  "Index + Pinky",  "Right Click"),
    ("scroll",       "Index + Middle", "Scroll Up/Down"),
]

# ---------------------------------------------------------------------------
# HAND GEOMETRY HELPERS
# ---------------------------------------------------------------------------

# Landmark indices (MediaPipe Hands, 21 points)
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

# MediaPipe hand skeleton connections for drawing (same as the old
# mp.solutions.hands.HAND_CONNECTIONS).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle  (via 0→9 shortcut)
    (0, 13), (13, 14), (14, 15), (15, 16), # ring    (via 0→13 shortcut)
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky   (via 0→17 shortcut)
    (5, 9), (9, 13), (13, 17),             # palm cross-connections
]


def draw_hand_landmarks(frame, landmarks):
    """Draw hand skeleton on the frame, replicating the old mp_draw behavior."""
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Draw connections (lines)
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

    # Draw landmarks (circles)
    for px, py in points:
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)



def create_guide_panel(height, current_gesture=None, mouse_mode=False,
                       mouse_action_label=""):
    """Create a standalone guide panel image to display beside the feed.

    When *mouse_mode* is True the panel shows mouse controls instead of the
    gesture table, using a yellow/cyan colour scheme to visually distinguish
    the two modes.
    """
    panel_w = 280
    panel = np.zeros((height, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # dark background

    if mouse_mode:
        # ---------- Mouse mode panel ----------
        header_color = (0, 255, 255)  # yellow/cyan
        cv2.putText(panel, "MOUSE MODE", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2)
        cv2.line(panel, (10, 42), (panel_w - 10, 42), header_color, 1)

        y = 70
        line_h = 38
        for gesture_key, gesture_name, action_label in MOUSE_GUIDE:
            is_active = (mouse_action_label and
                         action_label.lower().startswith(
                             mouse_action_label.lower()[:4]))

            if is_active:
                cv2.rectangle(panel, (4, y - 18), (panel_w - 4, y + 10),
                              (0, 100, 100), -1)

            name_color = (0, 255, 255) if is_active else (200, 200, 200)
            act_color = (0, 255, 180) if is_active else (130, 130, 130)

            cv2.putText(panel, gesture_name, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, name_color, 1)
            cv2.putText(panel, action_label, (145, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, act_color, 1)
            y += line_h

        # Footer
        cv2.putText(panel, "[M] Gesture Mode  [H] Hide  [Q] Quit",
                    (12, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 100, 100), 1)

        cv2.line(panel, (0, 0), (0, height), header_color, 2)
    else:
        # ---------- Gesture mode panel (original) ----------
        cv2.putText(panel, "GESTURE GUIDE", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.line(panel, (10, 42), (panel_w - 10, 42), (0, 220, 255), 1)

        y = 70
        line_h = (height - 120) // len(GESTURE_GUIDE)
        line_h = min(line_h, 34)

        for gesture_key, gesture_name, action_label in GESTURE_GUIDE:
            is_active = (gesture_key == current_gesture)

            if is_active:
                cv2.rectangle(panel, (4, y - 18), (panel_w - 4, y + 10),
                              (0, 120, 0), -1)

            name_color = (0, 255, 0) if is_active else (200, 200, 200)
            action_color = (0, 255, 180) if is_active else (130, 130, 130)

            cv2.putText(panel, gesture_name, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, name_color, 1)
            cv2.putText(panel, action_label, (145, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, action_color, 1)
            y += line_h

        cv2.putText(panel, "[M] Mouse Mode  [H] Hide  [Q] Quit",
                    (12, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 100, 100), 1)

        cv2.line(panel, (0, 0), (0, height), (0, 220, 255), 2)

    return panel


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def get_finger_states(landmarks, handedness_label):
    """Return dict of which fingers are extended: index, middle, ring, pinky, thumb.

    Uses simple, resolution-independent comparisons of normalized landmark
    coordinates. Works for either hand and works whether the hand is upright
    or slightly rotated, though it's tuned for a roughly upright, front-facing
    hand (typical webcam framing).
    """
    lm = landmarks

    fingers = {}
    # Non-thumb fingers: extended if the tip is higher (smaller y) than the pip joint.
    fingers["index"] = lm[INDEX_TIP].y < lm[INDEX_PIP].y
    fingers["middle"] = lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y
    fingers["ring"] = lm[RING_TIP].y < lm[RING_PIP].y
    fingers["pinky"] = lm[PINKY_TIP].y < lm[PINKY_PIP].y

    # Thumb: extended if tip is farther from the palm (MCP of index) than the
    # thumb's own IP joint is, i.e. thumb is stuck out sideways/up rather than
    # curled across the palm. This works regardless of left/right hand.
    thumb_extended = _dist(lm[THUMB_TIP], lm[INDEX_MCP]) > _dist(lm[THUMB_IP], lm[INDEX_MCP]) * 1.15
    fingers["thumb"] = thumb_extended

    return fingers


def classify_gesture(landmarks, handedness_label):
    """Map finger states + thumb orientation to a named gesture, or None.

    Every gesture here corresponds to a unique combination of which fingers
    are extended (index/middle/ring/pinky/thumb), so there's no overlap
    between gestures — each one is checked against a distinct pattern.
    Reference (I=index, M=middle, R=ring, P=pinky, T=thumb; 1=extended):

        fist          00000
        thumbs_up     0000T (thumb up)
        thumbs_down   0000T (thumb down)
        point_up      I0000
        peace         IM000
        three         IMR00
        four          IMRP0
        open_palm     IMRPT
        pinky         000P0
        rock_on       I00P0
        gun           I000T
        shaka         000PT
        ok_sign       special case: thumb+index tips touching, M+R+P extended
    """
    f = get_finger_states(landmarks, handedness_label)
    lm = landmarks

    I, M, R, P, T = f["index"], f["middle"], f["ring"], f["pinky"], f["thumb"]

    # --- OK sign checked first: thumb & index tips touching overrides their
    #     individual up/down state, so this must not fall through to the
    #     generic index/thumb checks below. ---
    if _dist(lm[THUMB_TIP], lm[INDEX_TIP]) < 0.055 and M and R and P:
        return "ok_sign"

    up_count = sum([I, M, R, P])

    # --- Fist: nothing extended at all ---
    if up_count == 0 and not T:
        return "fist"

    # --- Thumb only: direction relative to wrist decides up vs down ---
    if up_count == 0 and T:
        if lm[THUMB_TIP].y < lm[WRIST].y - 0.05:
            return "thumbs_up"
        elif lm[THUMB_TIP].y > lm[WRIST].y + 0.05:
            return "thumbs_down"
        return None

    # --- Open palm: all five extended ---
    if up_count == 4 and T:
        return "open_palm"

    # --- Four: all four fingers, thumb tucked ---
    if up_count == 4 and not T:
        return "four"

    # --- Point up: index only ---
    if I and not M and not R and not P and not T:
        return "point_up"

    # --- Peace: index + middle only ---
    if I and M and not R and not P and not T:
        return "peace"

    # --- Three: index + middle + ring, no pinky, no thumb ---
    if I and M and R and not P and not T:
        return "three"

    # --- Pinky only ---
    if P and not I and not M and not R and not T:
        return "pinky"

    # --- Rock on: index + pinky, middle + ring curled, no thumb ---
    if I and P and not M and not R and not T:
        return "rock_on"

    # --- Gun / L-shape: thumb + index only ---
    if T and I and not M and not R and not P:
        return "gun"

    # --- Shaka / hang loose: thumb + pinky only ---
    if T and P and not I and not M and not R:
        return "shaka"

    return None


# ---------------------------------------------------------------------------
# MOUSE MODE — air mouse, pinch click, and scroll
# ---------------------------------------------------------------------------

# Screen resolution for mapping hand coordinates to pixel positions.
_SCREEN_W, _SCREEN_H = pyautogui.size()


def get_mouse_action(landmarks, handedness_label):
    """Classify the hand pose into a mouse-mode action using finger states.

    Returns one of:
        ("move", norm_x, norm_y)        — index finger only
        ("left_click", norm_x, norm_y)  — index + thumb extended
        ("right_click", norm_x, norm_y) — index + pinky extended
        ("scroll", norm_x, norm_y)      — index + middle extended
        None                            — unrecognized / no hand
    """
    lm = landmarks
    f = get_finger_states(lm, handedness_label)
    I, M, R, P, T = f["index"], f["middle"], f["ring"], f["pinky"], f["thumb"]

    # Use index fingertip as the pointer position.
    norm_x, norm_y = lm[INDEX_TIP].x, lm[INDEX_TIP].y

    # Index must be up for any mouse action.
    if not I:
        return None

    # --- Scroll: index + middle, others down ---
    if M and not R and not P and not T:
        return ("scroll", norm_x, norm_y)

    # --- Left click / drag: index + thumb, others down ---
    if T and not M and not R and not P:
        return ("left_click", norm_x, norm_y)

    # --- Right click: index + pinky, others down ---
    if P and not M and not R and not T:
        return ("right_click", norm_x, norm_y)

    # --- Move cursor: index only, everything else down ---
    if not M and not R and not P and not T:
        return ("move", norm_x, norm_y)

    return None


def _map_to_screen(norm_x, norm_y):
    """Map normalized hand coordinates (0-1) to screen pixel coordinates,
    with margins so you don't need to reach the absolute frame edges."""
    margin = MOUSE_FRAME_MARGIN
    # Clamp and remap from [margin, 1-margin] → [0, 1]
    x = (norm_x - margin) / (1.0 - 2 * margin)
    y = (norm_y - margin) / (1.0 - 2 * margin)
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return x * _SCREEN_W, y * _SCREEN_H


def apply_mouse_action(action, mouse_state):
    """Execute a mouse action with debounce-based click logic.

    mouse_state dict keys:
        smooth_x, smooth_y   — EMA-smoothed screen coordinates
        left_holding         — True while left mouse button is held down
        left_frames          — consecutive frames seeing "left_click" action
        right_frames         — consecutive frames seeing "right_click" action
        right_fired          — True if right-click already fired this hold
        prev_scroll_y        — last frame's norm_y for scroll delta
    """
    if action is None:
        # Hand lost or unrecognized pose — release any held buttons.
        if mouse_state["left_holding"]:
            try:
                pyautogui.mouseUp()
            except pyautogui.FailSafeException:
                pass
            mouse_state["left_holding"] = False
        mouse_state["left_frames"] = 0
        mouse_state["right_frames"] = 0
        mouse_state["right_fired"] = False
        return "No mouse action"

    kind, norm_x, norm_y = action
    target_x, target_y = _map_to_screen(norm_x, norm_y)

    # ---- EMA smoothing ----
    if mouse_state["smooth_x"] is None:
        mouse_state["smooth_x"] = target_x
        mouse_state["smooth_y"] = target_y
    else:
        dx = target_x - mouse_state["smooth_x"]
        dy = target_y - mouse_state["smooth_y"]
        dist = math.hypot(dx, dy)
        if dist > MOUSE_DEADZONE * _SCREEN_W:
            mouse_state["smooth_x"] += dx * MOUSE_SMOOTHING
            mouse_state["smooth_y"] += dy * MOUSE_SMOOTHING

    sx = int(mouse_state["smooth_x"])
    sy = int(mouse_state["smooth_y"])

    try:
        # ---- Scroll mode ----
        if kind == "scroll":
            # Release any held click first
            if mouse_state["left_holding"]:
                pyautogui.mouseUp()
                mouse_state["left_holding"] = False
            mouse_state["left_frames"] = 0
            mouse_state["right_frames"] = 0

            prev_y = mouse_state.get("prev_scroll_y")
            if prev_y is not None:
                delta = prev_y - norm_y
                scroll_amount = int(delta * SCROLL_SPEED * 100)
                if abs(scroll_amount) > 0:
                    pyautogui.scroll(scroll_amount, _pause=False)
            mouse_state["prev_scroll_y"] = norm_y
            return "Scrolling"

        # ---- Left click / drag: index + thumb ----
        if kind == "left_click":
            mouse_state["left_frames"] += 1
            mouse_state["right_frames"] = 0  # reset right counter

            if mouse_state["left_holding"]:
                # Already holding — continue drag
                pyautogui.moveTo(sx, sy, _pause=False)
                return "Left drag"
            elif mouse_state["left_frames"] >= CLICK_DEBOUNCE_FRAMES:
                # Debounce passed — fire mouse down
                pyautogui.moveTo(sx, sy, _pause=False)
                pyautogui.mouseDown()
                mouse_state["left_holding"] = True
                return "Left click"
            else:
                # Still debouncing — just move
                pyautogui.moveTo(sx, sy, _pause=False)
                return "Moving cursor"

        # ---- Right click: index + pinky ----
        if kind == "right_click":
            mouse_state["right_frames"] += 1
            # Release left if held
            if mouse_state["left_holding"]:
                pyautogui.mouseUp()
                mouse_state["left_holding"] = False
            mouse_state["left_frames"] = 0

            if (mouse_state["right_frames"] >= CLICK_DEBOUNCE_FRAMES
                    and not mouse_state["right_fired"]):
                pyautogui.moveTo(sx, sy, _pause=False)
                pyautogui.rightClick(_pause=False)
                mouse_state["right_fired"] = True
                return "Right click"
            else:
                pyautogui.moveTo(sx, sy, _pause=False)
                return "Moving cursor"

        # ---- Plain move: index only ----
        if kind == "move":
            # Release left if held
            if mouse_state["left_holding"]:
                pyautogui.mouseUp()
                mouse_state["left_holding"] = False
            mouse_state["left_frames"] = 0
            mouse_state["right_frames"] = 0
            mouse_state["right_fired"] = False
            pyautogui.moveTo(sx, sy, _pause=False)
            return "Moving cursor"

    except pyautogui.FailSafeException:
        return "Blocked (mouse in corner)"

    return ""


def draw_mouse_overlay(frame, landmarks, action, mouse_state):
    """Draw mouse-mode HUD elements on the video frame."""
    h, w, _ = frame.shape

    if landmarks is not None:
        # Draw crosshair at index fingertip
        idx_tip = landmarks[INDEX_TIP]
        cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)

        # Outer ring
        cv2.circle(frame, (cx, cy), 18, (255, 255, 0), 2)
        # Crosshair lines
        cv2.line(frame, (cx - 24, cy), (cx - 10, cy), (255, 255, 0), 2)
        cv2.line(frame, (cx + 10, cy), (cx + 24, cy), (255, 255, 0), 2)
        cv2.line(frame, (cx, cy - 24), (cx, cy - 10), (255, 255, 0), 2)
        cv2.line(frame, (cx, cy + 10), (cx, cy + 24), (255, 255, 0), 2)
        # Center dot
        cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)

        # Click feedback
        if mouse_state["left_holding"]:
            cv2.circle(frame, (cx, cy), 26, (0, 255, 0), 3)  # green = left click/drag
        elif mouse_state["right_fired"]:
            cv2.circle(frame, (cx, cy), 26, (255, 100, 0), 3)  # blue = right click
        elif action is not None and action[0] == "scroll":
            cv2.arrowedLine(frame, (cx, cy - 30), (cx, cy - 50), (0, 200, 255), 2)
            cv2.arrowedLine(frame, (cx, cy + 30), (cx, cy + 50), (0, 200, 255), 2)

    # Mode label top-right
    label = "MOUSE MODE"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, label, (w - label_size[0] - 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam. Check camera permissions / index.")
        return

    # Set up HandLandmarker with the new Tasks API (VIDEO mode for frame-by-
    # frame processing with tracking between frames).
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # Track how long the current gesture has been held, and last fire time.
    current_gesture = None
    gesture_start_time = 0.0
    last_fired = {name: 0.0 for name in GESTURE_ACTIONS}
    fired_this_hold = False

    prev_time = time.time()
    frame_timestamp_ms = 0
    show_guide = True   # toggle with 'h' key
    mouse_mode = False  # toggle with 'm' key

    # Mouse mode state — reset each time mouse mode is entered.
    def _new_mouse_state():
        return {
            "smooth_x": None,
            "smooth_y": None,
            "left_holding": False,
            "left_frames": 0,
            "right_frames": 0,
            "right_fired": False,
            "prev_scroll_y": None,
        }
    mouse_state = _new_mouse_state()

    print("Gesture Controlled PC running. Press 'q' in the video window to quit.")
    print("Press 'h' in the video window to toggle the gesture guide panel.")
    print("Press 'm' in the video window to toggle Mouse Mode (air mouse).")
    print("Recognized gestures:")
    for name, (label, _) in GESTURE_ACTIONS.items():
        print(f"  {name:12s} -> {label}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to MediaPipe Image and run detection.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms += 33  # ~30 fps; must be monotonically increasing
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            now = time.time()
            landmarks = None
            handedness_label = "Right"

            if results.hand_landmarks:
                landmarks = results.hand_landmarks[0]  # first hand
                if results.handedness:
                    handedness_label = results.handedness[0][0].category_name
                draw_hand_landmarks(frame, landmarks)

            status_text = "No hand detected"
            mouse_action_label = ""

            if mouse_mode:
                # =================== MOUSE MODE ===================
                mouse_action = None
                if landmarks is not None:
                    mouse_action = get_mouse_action(landmarks, handedness_label)

                mouse_action_label = apply_mouse_action(mouse_action, mouse_state)
                status_text = f"Mouse: {mouse_action_label}"

                # Draw mouse-specific overlay (crosshair, click feedback)
                draw_mouse_overlay(frame, landmarks, mouse_action, mouse_state)

                # Reset scroll reference when leaving scroll mode
                if mouse_action is None or mouse_action[0] != "scroll":
                    mouse_state["prev_scroll_y"] = None

            else:
                # =================== GESTURE MODE ===================
                detected = None
                if landmarks is not None:
                    detected = classify_gesture(landmarks, handedness_label)

                if detected != current_gesture:
                    current_gesture = detected
                    gesture_start_time = now
                    fired_this_hold = False

                status_text = "No gesture"
                if current_gesture:
                    label, action_fn = GESTURE_ACTIONS.get(
                        current_gesture, (current_gesture, None))
                    held_for = now - gesture_start_time
                    status_text = f"{current_gesture} ({label}) held {held_for:.1f}s"

                    if (
                        action_fn is not None
                        and held_for >= HOLD_TIME
                        and not fired_this_hold
                        and (now - last_fired[current_gesture]) >= COOLDOWN
                    ):
                        try:
                            action_fn()
                            last_fired[current_gesture] = now
                            fired_this_hold = True
                            status_text += "  [FIRED]"
                        except pyautogui.FailSafeException:
                            status_text += "  [BLOCKED - mouse in corner]"

            # FPS
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # HUD text
            status_color = (255, 255, 0) if mouse_mode else (0, 255, 0)
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            mode_hint = "[M] Gesture Mode" if mouse_mode else "[M] Mouse Mode"
            cv2.putText(frame, f"Press 'q' to quit  {mode_hint}",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Build display: camera feed + optional guide panel side by side
            if show_guide:
                panel = create_guide_panel(
                    frame.shape[0],
                    current_gesture=current_gesture,
                    mouse_mode=mouse_mode,
                    mouse_action_label=mouse_action_label,
                )
                display = np.hstack((frame, panel))
            else:
                display = frame

            cv2.imshow("Gesture Controlled PC", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                show_guide = not show_guide
            elif key == ord("m"):
                mouse_mode = not mouse_mode
                if mouse_mode:
                    # Entering mouse mode — reset state
                    mouse_state = _new_mouse_state()
                    print("[mode] Switched to MOUSE MODE (air mouse + finger clicks)")
                else:
                    # Leaving mouse mode — release any held buttons
                    if mouse_state["left_holding"]:
                        try:
                            pyautogui.mouseUp()
                        except pyautogui.FailSafeException:
                            pass
                    mouse_state = _new_mouse_state()
                    current_gesture = None
                    fired_this_hold = False
                    print("[mode] Switched to GESTURE MODE")
    finally:
        # Clean up: release any held mouse button before exiting
        if mouse_mode and mouse_state["left_holding"]:
            try:
                pyautogui.mouseUp()
            except pyautogui.FailSafeException:
                pass
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
