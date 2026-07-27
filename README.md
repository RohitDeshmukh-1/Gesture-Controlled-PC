# Gesture Controlled PC

Control your computer using hand gestures tracked live from your webcam — no special hardware needed, just a standard camera and Python.

---

## How It Works

The system runs a real-time pipeline with four stages:

```
Webcam → OpenCV → MediaPipe HandLandmarker → Gesture Classifier → OS Action
```

1. **OpenCV** captures video frames from your webcam and displays the live feed with overlays.
2. **MediaPipe HandLandmarker** (Tasks API) detects your hand in each frame and returns 21 3D landmark points (fingertips, knuckles, wrist, etc.). It uses a pre-trained `.task` model file that is **automatically downloaded** on first run.
3. A **rule-based gesture classifier** checks which fingers are extended vs. curled by comparing landmark positions — each gesture maps to a unique finger pattern, so no two gestures can be confused.
4. When a gesture is held steadily for 0.5 seconds, **PyAutoGUI** (and supporting libraries) execute the mapped OS action — pressing a key, adjusting brightness, taking a screenshot, or launching an app.

The entire pipeline runs in a single Python script ([`gesture_control.py`](gesture_control.py)). The MediaPipe model (`hand_landmarker.task`, ~12 MB) is downloaded automatically the first time you run the script — no manual setup needed.

---

## Tech Stack

| Library | Version | Role |
|---|---|---|
| [OpenCV](https://opencv.org/) (`opencv-python`) | ≥ 4.10.0 | Webcam capture, frame processing, and live video display |
| [MediaPipe](https://mediapipe.dev/) (`mediapipe`) | ≥ 0.10.30 | Real-time hand landmark detection via the modern Tasks API (21 keypoints per hand) |
| [PyAutoGUI](https://pyautogui.readthedocs.io/) (`pyautogui`) | ≥ 0.9.54 | Simulating keyboard presses, hotkeys, and taking screenshots |
| [screen-brightness-control](https://github.com/Crozzers/screen_brightness_control) (`screen-brightness-control`) | ≥ 0.24.1 | Cross-platform screen brightness adjustment via OS APIs |

**Built-in Python modules also used:** `subprocess` (launching apps), `platform` (OS detection), `time`, `math`, `urllib.request` (auto model download).

---

## Supported Gestures

Every gesture uses a **unique combination of extended fingers**, making them unambiguous. The classifier checks each finger independently: **I** = index, **M** = middle, **R** = ring, **P** = pinky, **T** = thumb. `1` = extended, `0` = curled.

| Gesture | Finger Pattern | Action |
|---|---|---|
| 👊 Fist | `0 0 0 0 0` | Take a screenshot (saved to working directory) |
| ✋ Open palm | `1 1 1 1 1` | Play / Pause (spacebar) |
| ☝️ Point up | `I 0 0 0 0` | Alt + Tab (switch windows) |
| ✌️ Peace | `I M 0 0 0` | Next Slide (right arrow) |
| 🤟 Three fingers | `I M R 0 0` | Previous Slide (left arrow) |
| 🖖 Four fingers (no thumb) | `I M R P 0` | Open Google Chrome |
| 👍 Thumbs up | `0 0 0 0 T↑` | Volume Up |
| 👎 Thumbs down | `0 0 0 0 T↓` | Volume Down |
| 🤙 Pinky only | `0 0 0 P 0` | Previous Track |
| 🤘 Rock on | `I 0 0 P 0` | Next Track |
| 🤙👍 Shaka | `0 0 0 P T` | Brightness Up (+10%) |
| 🔫 Gun / L-shape | `I 0 0 0 T` | Brightness Down (-10%) |
| 👌 OK sign | Thumb + index tips touching, M R P extended | Open VS Code |

---

## Mouse Mode (Air Mouse)

Press **`m`** in the video window to toggle **Mouse Mode**. In this mode, your hand becomes a touchless mouse — point, click, drag, and scroll without touching your desk.

While Mouse Mode is active, all standard gesture actions (play/pause, screenshot, etc.) are **disabled** to prevent accidental triggers. Press `m` again to switch back to Gesture Mode.

### Mouse Controls

| Hand Pose | Action |
|---|---|
| ☝️ **Index finger only** | **Move cursor** — index fingertip maps to screen position with EMA smoothing |
| ☝️👍 **Index + Thumb extended** | **Left click** (quick pose) / **Drag** (hold pose and move hand) |
| ☝️🤙 **Index + Pinky extended** | **Right click** (single fire) |
| ✌️ **Index + Middle extended** | **Scroll** — move hand up/down to scroll pages |

### How the Smoothing & Debounce Work

- **Exponential Moving Average (EMA)** (`MOUSE_SMOOTHING = 0.3`) filters out frame-to-frame hand jitter.
- **Deadzone** (`MOUSE_DEADZONE = 0.005`) ignores tiny micro-tremors when holding your finger still.
- **Frame debounce** (`CLICK_DEBOUNCE_FRAMES = 3`): Click gestures must be held for 3 consecutive frames (~100 ms) before triggering, eliminating single-frame mis-clicks.

### Mouse Mode Config

All tuning values are at the top of [`gesture_control.py`](gesture_control.py):

| Parameter | Default | Description |
|---|---|---|
| `MOUSE_SMOOTHING` | `0.3` | EMA factor for cursor smoothing (0.0–1.0) |
| `MOUSE_DEADZONE` | `0.005` | Minimum normalized movement to register |
| `CLICK_DEBOUNCE_FRAMES` | `3` | Consecutive frames to confirm a click (~100 ms) |
| `SCROLL_SPEED` | `15` | Scroll sensitivity multiplier |
| `MOUSE_FRAME_MARGIN` | `0.1` | Frame edge margin (fraction, 0.0–0.5) |

---

## Setup

### Prerequisites

- **Python 3.9+** (tested on 3.10, 3.11, and 3.13)
- A working **webcam**
- Internet connection on first run (to auto-download the ~12 MB hand landmarker model)

### Installation

```bash
# Clone or download this project, then:
cd Gesture

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python gesture_control.py
```

On first run, the script will download the `hand_landmarker.task` model (~12 MB) automatically. Then a webcam window opens showing the live feed with hand landmarks drawn on screen. Hold any gesture steady for ~0.5 seconds to trigger its action. The current gesture, hold time, and FPS are shown as overlay text.

**Press `q` in the video window to quit.**

---

## Configuration

All tuning values are at the top of [`gesture_control.py`](gesture_control.py):

| Parameter | Default | Description |
|---|---|---|
| `HOLD_TIME` | `0.5s` | How long a gesture must be held before it fires. Lower = snappier, higher = fewer accidental triggers. |
| `COOLDOWN` | `1.0s` | Minimum time between repeat firings of the *same* gesture. Prevents spamming (e.g., holding thumbs-up won't rapidly repeat volume-up). |
| `min_detection_confidence` | `0.7` | MediaPipe's confidence threshold for detecting a hand in the frame. |
| `min_tracking_confidence` | `0.6` | MediaPipe's confidence threshold for tracking landmarks frame-to-frame. |

### Remapping Gestures

Every gesture maps to a simple Python function in the `GESTURE_ACTIONS` dictionary. To change what a gesture does, just edit the function it points to:

```python
# Example: make the "fist" gesture mute instead of screenshot
GESTURE_ACTIONS = {
    "fist": ("Mute", action_mute),
    ...
}
```

### Adding New Gestures

1. Pick a finger combination not already in the table above.
2. Write an `action_...()` function for what it should do.
3. Add a matching `if` branch in `classify_gesture()` that returns a name.
4. Add that name to `GESTURE_ACTIONS`.

```python
# Example: thumb + middle finger only (others curled)
if T and not I and M and not R and not P:
    return "thumb_middle"
```

> **Tip:** Gestures using thumb, index, or pinky are the most reliable — middle and ring fingers share tendons and are harder to raise independently.

---

## Safety

- **PyAutoGUI failsafe is enabled** — moving your mouse to any screen corner immediately aborts all automated actions.
- **No OS bypassing** — all actions go through standard OS APIs. On macOS, you must explicitly grant Camera and Accessibility permissions. On Windows, no elevated privileges are needed.
- **No shell injection** — app launching uses `subprocess.Popen` with list arguments (never `shell=True`), so no arbitrary commands can be injected.
- **Brightness is clamped** to 0–100%, preventing out-of-range values.

---

## Platform Notes

### Windows
Works out of the box. Media keys (`volumeup`, `volumedown`, `nexttrack`, `prevtrack`) are natively supported by PyAutoGUI.

### macOS
- Grant **Camera** permission to your terminal/IDE on first run (System Settings → Privacy & Security → Camera).
- Grant **Accessibility** permission for PyAutoGUI to simulate key presses (System Settings → Privacy & Security → Accessibility).
- Brightness control is limited — if `screen-brightness-control` doesn't work, install the `brightness` CLI (`brew install brightness`) and modify the `action_brightness_*` functions to call it via `subprocess`.

### Linux
- Install `python3-tk` and `scrot` if `pyautogui.screenshot()` fails:
  ```bash
  sudo apt install python3-tk scrot
  ```
- Media keys depend on your desktop environment. If `volumeup`/`nexttrack` don't work, swap those action functions for `amixer`/`playerctl` calls.

---

## Known Limitations

- **Single hand only** — tracks one hand at a time (`max_num_hands=1`) for stability. Can be changed in the `mp_hands.Hands(...)` call.
- **Lighting dependent** — MediaPipe works best with your hand clearly lit and reasonably close to the camera.
- **Rule-based classifier** — unusual hand angles may occasionally be misread. The finger-state thresholds in `get_finger_states()` and `classify_gesture()` can be tuned for your specific setup.

---

## Project Structure

```
Gesture/
├── gesture_control.py     # Main script — all logic in one file
├── requirements.txt       # Python dependencies
├── hand_landmarker.task   # MediaPipe model (auto-downloaded on first run)
└── README.md              # This file
```

## License

This project is provided as-is for personal and educational use.
