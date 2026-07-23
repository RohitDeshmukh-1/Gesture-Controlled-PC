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

import time
import math
import platform
import subprocess


import cv2
import mediapipe as mp
import pyautogui
import screen_brightness_control as sbc

# Safety: moving the mouse to a screen corner aborts pyautogui actions.
pyautogui.FAILSAFE = True

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
# HAND GEOMETRY HELPERS
# ---------------------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Landmark indices (MediaPipe Hands, 21 points)
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18


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
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam. Check camera permissions / index.")
        return

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    # Track how long the current gesture has been held, and last fire time.
    current_gesture = None
    gesture_start_time = 0.0
    last_fired = {name: 0.0 for name in GESTURE_ACTIONS}
    fired_this_hold = False

    prev_time = time.time()

    print("Gesture Controlled PC running. Press 'q' in the video window to quit.")
    print("Recognized gestures:")
    for name, (label, _) in GESTURE_ACTIONS.items():
        print(f"  {name:12s} -> {label}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = "Right"
            if results.multi_handedness:
                handedness_label = results.multi_handedness[0].classification[0].label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected = classify_gesture(hand_landmarks.landmark, handedness_label)

        now = time.time()

        if detected != current_gesture:
            current_gesture = detected
            gesture_start_time = now
            fired_this_hold = False

        status_text = "No gesture"
        if current_gesture:
            label, action_fn = GESTURE_ACTIONS.get(current_gesture, (current_gesture, None))
            held_for = now - gesture_start_time
            status_text = f"{current_gesture} ({label}) held {held_for:.1f}s"

            if (
                action_fn is not None
                and held_for >= HOLD_TIME
                and not fired_this_hold
                and (now - last_fired[current_gesture]) >= COOLDOWN
            ):
                action_fn()
                last_fired[current_gesture] = now
                fired_this_hold = True
                status_text += "  [FIRED]"

        # FPS
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Gesture Controlled PC", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
