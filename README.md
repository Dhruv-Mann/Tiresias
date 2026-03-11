# Tiresias

**An AI-powered assistive vision system for the visually impaired.**

Tiresias uses a webcam to perceive the world in real-time — detecting objects, estimating depth, and determining what's in the user's path. Named after the blind prophet of Greek mythology who could "see" what others couldn't.

---

## What It Does

- **Real-time object detection** using YOLO-World (open-vocabulary — detects any object, not just a fixed set)
- **Monocular depth estimation** using MiDaS (near = RED, far = BLUE)
- **Sensor fusion** — fuses detection + depth to classify objects as NEAR / MID / FAR
- **Audio alerts** — speaks warnings like *"person ahead, close"* using offline text-to-speech
- **Spatial awareness** — divides the frame into Left / Center / Right zones; objects in Center are flagged as obstacles
- **Custom class prompts** — define exactly which objects matter for navigation (stairs, curbs, potholes, traffic lights, etc.)

---

## Project Journey — Phase by Phase

### Phase 1: The Eyes (Basic Webcam Feed)

**Goal:** Get a live camera feed working with OpenCV.

**What we did:**
- Set up the Python project structure with a virtual environment (Python 3.12).
- Created `main.py` with OpenCV's `VideoCapture` to grab frames from the webcam.
- Displayed the raw live feed in a window with `cv2.imshow()`.
- Added proper cleanup (`cap.release()`, `destroyAllWindows()`) and a quit mechanism (`q` key).
- Set up `setup_env.bat` (Windows) and `setup_env.sh` (macOS/Linux) for one-command environment setup.

**Key decisions:**
- Resolution set to 640×480 — good balance of quality vs. processing speed.
- Used `cv2.waitKey(1)` (not `0`) so the feed runs in real-time instead of freezing per frame.

**Files created:**
- `main.py` — entry point, camera initialization, main loop
- `requirements.txt` — dependencies (just `opencv-python` and `numpy` at this point)
- `setup_env.bat` / `setup_env.sh` — environment setup scripts

---

### Phase 2: Depth Perception (MiDaS)

**Goal:** Give Tiresias the ability to perceive how far objects are using just a single camera — no LiDAR, no stereo cameras.

**What we did:**
- Integrated **MiDaS v2.1 Small** (Intel ISL) via `torch.hub.load()` for monocular depth estimation.
- Created `depth_estimation.py` with a `DepthEstimator` class that:
  - Loads the MiDaS Small model (once, at startup — not per frame).
  - Converts BGR→RGB (OpenCV vs ML convention mismatch).
  - Runs inference with `torch.no_grad()` for speed + memory savings.
  - Resizes output back to 640×480 with bicubic interpolation.
  - Applies JET colormap: **RED = near**, **BLUE = far**.
- Added a second `cv2.imshow()` window for the depth map alongside the live feed.
- Added `get_raw_depth()` method for future phases (returns grayscale depth values without colormap).

**Problems faced:**
- **BGR vs RGB confusion** — MiDaS expects RGB but OpenCV captures in BGR. Skipping conversion produces a subtly wrong depth map (silent failure, hard to catch). Fixed with `cv2.cvtColor()`.
- **Model loading time** — MiDaS takes 2–5 seconds to load. Initially considered putting it inside the loop (which would've made it unusable). Solution: load once before the loop.
- **PyTorch Hub downloads** — First run downloads ~80MB of weights. Added `trust_repo=True` to avoid interactive prompts.

**Key decisions:**
- Chose MiDaS **Small** over DPT_Large: ~5-10 FPS on CPU vs <1 FPS. For real-time assistive use, speed > precision.
- Used `model.eval()` to disable training-mode behaviors (BatchNorm, Dropout) for consistent predictions.
- CUDA auto-detection: uses GPU if available, falls back to CPU gracefully.

**Files created / modified:**
- `depth_estimation.py` — new module for depth estimation
- `main.py` — added depth estimator integration
- `requirements.txt` — added `torch`, `torchvision`, `timm`

---

### Phase 3: The Brain (Object Detection — YOLOv8)

**Goal:** Detect real-world objects in the camera feed and determine if they're in the user's path.

**What we did:**
- Integrated **YOLOv8** (Ultralytics) for object detection.
- Created `object_detection.py` with an `ObjectDetector` class that:
  - Loads a YOLOv8 model (started with nano, moved to extra-large for accuracy).
  - Runs inference on each frame, extracting bounding boxes, confidence scores, and class labels.
  - Calculates the **center point** of each detection.
  - Classifies detections into **Left / Center / Right** zones (frame divided into thirds).
  - Draws color-coded bounding boxes: **RED** for Center (danger), **GREEN** for sides (safer).
  - Draws zone divider lines, center dots, and labels with confidence + zone info.
- Two windows now run simultaneously: annotated live feed + depth map.

**Problems faced:**
- **YOLOv8 nano (yolov8n.pt) was too inaccurate** — missed many objects, frequent misclassifications. It's only 3.2M parameters trained on COCO's 80 classes.
- **Moved to YOLOv8x (yolov8x.pt)** — 68M+ parameters, much better accuracy, but still limited to 80 fixed COCO classes. This was a fundamental limitation: COCO doesn't include "stairs", "curb", "pothole", "door", etc. — all critical for blind navigation.
- **Console spam** — YOLO prints stats every frame by default. Fixed with `verbose=False`.

**Key decisions:**
- Confidence threshold set to 0.5 (balanced false positive/negative trade-off).
- Zone system uses simple thirds — center zone = direct path = danger.
- Color coding: red = danger (center), green = safe (sides) — intuitive warning system.

**Files created / modified:**
- `object_detection.py` — new module for object detection
- `main.py` — integrated detection pipeline
- `requirements.txt` — added `ultralytics`

---

### Phase 4: The Upgrade (YOLO-World — Open-Vocabulary Detection)

**Goal:** Replace YOLOv8's fixed 80-class detection with an open-vocabulary model that can detect *any* object described by text.

**Why the upgrade was critical:**
YOLOv8, even the extra-large variant with 68M+ parameters, is fundamentally limited to the 80 COCO dataset classes (person, car, chair, etc.). For a blind assistance tool, this is a dealbreaker:
- **Missing critical objects:** stairs, curbs, potholes, doors, traffic cones, barriers, fences, crosswalks, shopping carts, strollers — none of these are in COCO.
- **No customizability:** Can't add new classes without retraining the entire model on custom datasets.
- **Small training data:** COCO has ~330K images. Good for benchmarks, not enough for real-world diversity.

**What we did:**
- Replaced YOLOv8 with **YOLO-World v2** (`yolov8x-worldv2.pt`) — an open-vocabulary object detection model.
- YOLO-World uses a **vision-language model** approach: it takes text prompts describing what to detect, then finds those objects in the image. No retraining needed.
- Defined a comprehensive set of **30+ assistive navigation classes** including:
  - People, animals, vehicles
  - Indoor obstacles (chairs, tables, doors, cabinets)
  - Outdoor obstacles (benches, poles, fire hydrants, trash cans, cones, barriers, fences)
  - Navigation hazards (stairs, curbs, potholes)
  - Traffic infrastructure (traffic lights, stop signs, crosswalks)
  - Common objects (bags, umbrellas, strollers, wheelchairs, shopping carts)
- Made classes fully configurable — pass a custom list or use the defaults.
- Lowered confidence threshold to 0.4 (YOLO-World's open-vocabulary nature benefits from slightly lower thresholds).

**Key improvements over YOLOv8:**
| Feature | YOLOv8x | YOLO-World v2 |
|---|---|---|
| Detectable classes | 80 (fixed COCO) | **Unlimited** (text prompts) |
| Stairs, curbs, potholes | ❌ Not in COCO | ✅ Just add to class list |
| Add new classes | Requires retraining | Change one line of code |
| Architecture | Standard CNN | Vision-Language Model |
| Accuracy on custom classes | N/A | Strong zero-shot performance |

**Files modified:**
- `object_detection.py` — switched from `YOLO` to `YOLOWorld`, added class definitions
- `main.py` — updated docstrings
- `requirements.txt` — updated comments (same `ultralytics` package)

---

### Phase 5: Sensor Fusion & Audio Engine

**Goal:** Fuse detection + depth into a single perception pipeline and speak warnings aloud — making Tiresias actually usable by a blind person.

**What we did:**
- **Sensor fusion** — For each YOLO-World detection, we slice the MiDaS depth map at the bounding box ROI, compute the **median** depth, and classify it as NEAR (>170), MID (85-170), or FAR (<85).
- Created `audio_engine.py` with a **queue-based TTS system**:
  - A single dedicated daemon thread owns the pyttsx3 engine (avoids Windows COM threading crashes).
  - Main thread drops speech requests into a `queue.Queue` — never blocks.
  - Messages are spoken sequentially — no overlapping garbled audio.
- **Priority & cooldown logic** — Only speaks for NEAR objects. Center zone gets 3s cooldown (urgent), sides get 5s cooldown. Cooldown key is `label_zone` so "person Center" and "person Left" are tracked independently.
- **Color-coded threat visuals** — 4-level gradient: RED (near + center), ORANGE (near + side), YELLOW (mid distance), GREEN (far/safe).
- Labels now show `person - NEAR [Center]` instead of `person 87% [Center]` — proximity is more actionable than raw confidence.

**Problems faced:**
- **pyttsx3 threading crashes** — Gemini's original plan called for `threading.Thread(target=speak_alert)` creating a new engine per thread. pyttsx3 uses Windows COM objects with thread affinity — this causes crashes and hangs. Fixed with the single-worker-thread + queue pattern.
- **pyttsx3 engine caching bug** — Even with a single worker thread, `pyttsx3.init()` returns a cached singleton via an internal `_activeEngines` WeakValueDictionary. After one `runAndWait()`, the engine's COM event loop state corrupts on Windows, silently dropping all subsequent speech. Fixed by creating a fresh engine per speech and clearing `pyttsx3._activeEngines` after each call.
- **Mean vs Median depth** — Initially planned to use mean, but background pixels inside rectangular bounding boxes skew the average, making close objects appear farther. Median is robust to this (up to 49% outliers can't affect it).
- **Alert spam** — Without cooldowns, NEAR detections trigger 30 alerts/second. The cooldown dictionary with `time.monotonic()` (immune to clock changes) prevents this.
- **Only alerting Center zone was too restrictive** — A blind person turning should still hear about nearby obstacles on their left/right. Extended alerts to all zones with zone-appropriate cooldowns.

**Key decisions:**
- Median over mean for depth ROI (robust to background pixel contamination).
- `time.monotonic()` over `time.time()` (immune to NTP sync, DST, clock changes).
- Daemon thread (auto-killed on exit) as safety net alongside explicit `shutdown()`.
- Fusion logic lives in `main.py` (orchestrator), not in detection or depth modules (separation of concerns).

**Files created / modified:**
- `audio_engine.py` — new module: queue-based TTS with cooldowns (engine-per-speech + cache clearing for Windows COM reliability)
- `main.py` — added fusion logic, depth thresholds, audio integration
- `object_detection.py` — updated visuals: proximity labels, 4-color threat gradient
- `requirements.txt` — added `pyttsx3`

---

## Architecture

```
main.py                  ← Entry point: fusion orchestrator, wires everything
├── object_detection.py  ← YOLO-World: detects objects, zones, draws annotations
├── depth_estimation.py  ← MiDaS: monocular depth map (near=RED, far=BLUE)
└── audio_engine.py      ← pyttsx3: non-blocking spoken alerts (queue + thread)
```

**Data flow per frame:**
1. Camera captures a 640×480 BGR frame.
2. Frame → `ObjectDetector.detect()` → list of detections (label, confidence, box, center, zone).
3. Frame → `DepthEstimator.get_raw_depth()` → grayscale depth map (0-255, HIGH = near).
4. `fuse_detections_with_depth()` → samples depth at each bounding box ROI → adds proximity (NEAR/MID/FAR).
5. `AudioEngine.alert()` → if NEAR and cooldown elapsed, queues spoken warning.
6. Frame + detections → `ObjectDetector.draw_detections()` → annotated frame with color-coded threat levels.
7. Both windows displayed via `cv2.imshow()`.

---

## Setup

### Prerequisites
- Python 3.12+
- A webcam
- (Optional) NVIDIA GPU with CUDA for faster inference

### Quick Start

**Windows:**
```bash
setup_env.bat
venv\Scripts\activate
python main.py
```

**macOS/Linux:**
```bash
chmod +x setup_env.sh
./setup_env.sh
source venv/bin/activate
python main.py
```

### Manual Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
python main.py
```

> **Note:** On first run, YOLO-World and MiDaS models will be downloaded automatically. This is a one-time ~200MB download.

Press **q** to quit.

---

## Customizing Detection Classes

Edit the `DEFAULT_CLASSES` list in `object_detection.py`, or pass a custom list:

```python
from object_detection import ObjectDetector

# Detect only what you need
detector = ObjectDetector(classes=["person", "car", "stairs", "door"])
```

YOLO-World is open-vocabulary — any English noun or short phrase works as a class.

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture, image display, drawing |
| `numpy` | Array operations (frames are NumPy arrays) |
| `torch` + `torchvision` | PyTorch (MiDaS backend) |
| `timm` | PyTorch Image Models (MiDaS dependency) |
| `ultralytics` | YOLO-World object detection |
| `pyttsx3` | Offline text-to-speech (audio alerts) |

---

## Models Used

| Model | Task | Size | Speed |
|---|---|---|---|
| **YOLO-World v2 (x)** | Object Detection | ~200MB | Real-time on GPU |
| **MiDaS Small** | Depth Estimation | ~80MB | ~5-10 FPS (CPU), 30+ FPS (GPU) |

---

## License

See [LICENSE](LICENSE).
