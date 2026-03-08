# main.py — EXPLAINED

## Overview

This is the entry point of the **Tiresias** project. In Phase 1, its only job is to open the webcam, read frames in real-time, display them in a window, and allow the user to quit cleanly. Think of this as giving Tiresias its "eyes."

---

## Full Source Code

```python
"""
Tiresias - main.py
====================
Phase 1: The Eyes (Basic Webcam Feed)

Opens the default webcam, captures frames in real-time,
and displays them in a window. Press 'q' to quit.
"""

import cv2
import sys


def initialize_camera(camera_index: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """Initialize and configure the webcam."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Is a webcam connected?")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"[Tiresias] Camera initialized at {width}x{height}")
    return cap


def run():
    """Main loop: capture and display webcam frames."""
    cap = initialize_camera()

    print("[Tiresias] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to grab frame. Exiting.")
            break

        cv2.imshow("Tiresias - Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[Tiresias] Camera released. Goodbye.")


if __name__ == "__main__":
    run()
```

---

## Line-by-Line Explanation

### Imports

```python
import cv2
```
- **`cv2`** is OpenCV (Open Source Computer Vision Library). It provides functions to capture video from cameras, manipulate images, and display them in GUI windows. We use it for everything camera-related.

```python
import sys
```
- **`sys`** is a built-in Python module that provides access to system-specific parameters. We use `sys.exit(1)` to terminate the program with a non-zero exit code if the camera fails to open, signaling an error to the OS.

---

### `initialize_camera()` Function

```python
def initialize_camera(camera_index: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
```
- We define a function with **type hints** for clarity. `camera_index=0` means the default/first webcam. `width` and `height` set the desired resolution. The function returns a `cv2.VideoCapture` object.

```python
    cap = cv2.VideoCapture(camera_index)
```
- **Why `cv2.VideoCapture(0)`?** This creates a capture object that connects to the camera at index `0`. On most systems, `0` is the built-in/default webcam. If you have multiple cameras, `1`, `2`, etc. select the others.

```python
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Is a webcam connected?")
        sys.exit(1)
```
- **Why check `isOpened()`?** `VideoCapture()` doesn't throw an error if the camera isn't available — it silently fails. We must explicitly check. If the camera isn't found (e.g., no webcam plugged in, or another app is using it), we print an error and exit immediately rather than crashing later with a confusing error.

```python
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
```
- **Why set width and height?** By default, OpenCV may capture at the camera's native resolution (often 1920x1080), which is more pixels than we need and slows down processing. We request 640x480 — a good balance of quality and performance. Note: `cap.set()` is a *request*; the camera may not support that exact resolution and will use the closest one it supports.

```python
    print(f"[Tiresias] Camera initialized at {width}x{height}")
    return cap
```
- Confirmation message and return the configured capture object for use in the main loop.

---

### `run()` Function

```python
def run():
    cap = initialize_camera()
```
- Call our setup function to get the configured camera object.

```python
    print("[Tiresias] Press 'q' to quit.")
```
- Inform the user how to exit, since there's no GUI close button that stops the loop.

```python
    while True:
```
- **Why `while True`?** Video is just a rapid sequence of still images (frames). To create a "live feed," we need an infinite loop that grabs a new frame as fast as possible. The loop only breaks when the user presses 'q' or the camera fails.

```python
        ret, frame = cap.read()
```
- **`cap.read()`** grabs the next frame from the camera. It returns two values:
  - `ret` (bool): `True` if a frame was successfully captured, `False` if something went wrong (e.g., camera disconnected).
  - `frame` (numpy.ndarray): The actual image data as a NumPy array with shape `(height, width, 3)` — 3 channels for Blue, Green, Red (BGR format, not RGB — an OpenCV convention).

```python
        if not ret:
            print("[ERROR] Failed to grab frame. Exiting.")
            break
```
- **Why check `ret`?** If the camera disconnects mid-stream or a hardware error occurs, `ret` becomes `False`. Without this check, `frame` would be `None` and `cv2.imshow()` would crash.

```python
        cv2.imshow("Tiresias - Live Feed", frame)
```
- **`cv2.imshow()`** creates a named window and displays the frame in it. The first argument is the window title. If the window doesn't exist yet, OpenCV creates it automatically.

```python
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
```
- **Why `cv2.waitKey(1)`?** This is critical and does two things:
  1. **Waits 1 millisecond** for a keypress. Without this delay, `imshow()` cannot render the frame (OpenCV's GUI needs this call to process window events).
  2. **Returns the key code** of whatever key was pressed, or `-1` if no key was pressed.
- **Why `& 0xFF`?** On some systems (especially 64-bit Linux), `waitKey()` returns a 32-bit integer. The `& 0xFF` bitmask extracts only the last 8 bits, giving us the standard ASCII value.
- **`ord("q")`** converts the character `'q'` to its ASCII code (113). If the user pressed 'q', we break out of the loop.

---

### Cleanup

```python
    cap.release()
```
- **Why `cap.release()`?** This frees the camera hardware so other applications can use it. If you skip this, the camera may stay "locked" by your process even after the script ends.

```python
    cv2.destroyAllWindows()
```
- **Why `destroyAllWindows()`?** Closes all OpenCV GUI windows. Without this, windows may linger as zombie processes on some operating systems.

```python
    print("[Tiresias] Camera released. Goodbye.")
```
- Confirmation that cleanup succeeded.

---

### Entry Point Guard

```python
if __name__ == "__main__":
    run()
```
- **Why this guard?** This ensures `run()` only executes when you run the file directly (`python main.py`). If another file imports this module (e.g., `from main import initialize_camera`), it won't automatically start the webcam. This is a Python best practice for making code both runnable and importable.

---

## Viva Questions

### Q1: Why does the video feed lag if we don't call `cap.release()` — and what happens to system resources?

**Answer:** `cap.release()` frees the underlying camera device handle (a shared OS resource). If you don't call it:
- The camera remains "locked" by your Python process. Other applications (Zoom, Teams, etc.) won't be able to access it until your process terminates.
- On some systems, re-running the same script without releasing will fail because the OS still thinks the camera is in use.
- At the OS level, this is a **resource leak** — similar to opening a file and never closing it. Python's garbage collector *may* eventually release it, but that behavior is not guaranteed or timely.

The broader principle: any time you acquire a hardware resource or file handle, always release it explicitly. This is also why Python's `with` statement exists for file I/O.

---

### Q2: Why do we use `cv2.waitKey(1)` instead of `cv2.waitKey(0)`, and what would happen without it?

**Answer:** The argument to `waitKey()` is the delay in milliseconds:
- **`waitKey(1)`**: Wait 1ms, then continue. This creates a ~1ms pause per frame, which:
  - Gives OpenCV's HighGUI event loop time to render the frame and handle window events (resize, close, etc.).
  - Allows keyboard input to be detected.
  - Results in the fastest possible frame rate (~30+ FPS depending on camera).
- **`waitKey(0)`**: Wait **indefinitely** until a key is pressed. The video would show one frame, freeze, wait for a keypress, then show the next frame. It becomes a "slideshow" — useless for real-time video.
- **Without `waitKey()` at all**: The `imshow()` window would never update. OpenCV relies on `waitKey()` to pump its internal message loop. You'd see a grey/blank window or the OS would report the application as "Not Responding."

---

### Q3: What is the `& 0xFF` bitmask doing, and why is it necessary on some platforms?

**Answer:** `cv2.waitKey()` returns a 32-bit integer on some platforms (particularly 64-bit Linux with GTK backends). The upper bits can contain modifier key flags (Shift, Ctrl, Alt, NumLock, etc.). For example, pressing 'q' might return `1048689` instead of `113` if NumLock is on.

The bitmask `& 0xFF` (which is `& 255` in decimal, or `& 0b11111111` in binary) strips away everything except the lowest 8 bits — the actual ASCII value of the key pressed.

- `1048689 & 0xFF = 113` → matches `ord("q")` ✓
- `113 & 0xFF = 113` → still matches ✓

On Windows, this is often not strictly necessary (the return value is usually clean), but including it is a **cross-platform safety practice** that costs nothing and prevents subtle bugs.
