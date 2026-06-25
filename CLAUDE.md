# CLAUDE.md — FaceLocker

## Project Overview

FaceLocker is a macOS real-time face detection overlay application. It captures
the screen in a background thread, runs YOLO face detection + KCF tracking, and
renders bounding boxes and optional aim-lines toward the mouse cursor as a
transparent overlay on every display.

- **Language:** Python 3
- **Platform:** macOS only (AppKit, Quartz, Metal/MPS)
- **Entry point:** `main.py` → `python main.py` or double-click `FaceLocker.command`
- **Virtual env:** `venv/` with dependencies in `requirements.txt`

## Architecture

```
main.py              — App controller (FaceLocker), overlay views, control panel, main loop
detector_yolo.py     — YOLO face detection (Ultralytics YOLO, face_yolov8n.pt, MPS accelerated)
tracker_faces.py     — KCF tracking bridge: YOLO every N frames, KCF in between
face_yolov8n.pt      — YOLOv8n face-detection model weights (6.2 MB)
FaceLocker.command   — Double-click launcher for macOS Finder
```

### Key Design Points

| Component              | Detail                                                         |
|------------------------|----------------------------------------------------------------|
| Screen capture         | `BackgroundCapture` — dedicated thread, `mss`, zero main-thread cost |
| Detection              | YOLOv8n face model, FP16 on MPS, every 4th tick (~15×/s)      |
| Tracking               | OpenCV KCF per face, runs every tick (60 Hz)                   |
| Overlay rendering      | Per-display `NSWindow` at `NSScreenSaverWindowLevel`, transparent, ignores mouse |
| Control panel          | Floating `NSWindow` with detection toggle, aim-lines toggle, box/line color wells, FPS counter |
| Timer                  | `NSTimer` at 60 Hz — detection limited to every 4th tick via `FaceTracker` |

### Data Flow

```
Screens → BackgroundCapture (thread) → get() → FaceTracker.update(bgr)
  → YOLO (every 4th) or KCF (intermediate) → normalised CGRects
  → FaceOverlayView → NSBezierPath boxes + aim-lines → GPU composited overlay
```

### Coordinate System

- YOLO outputs pixel coords with **top-left** origin
- Overlay view uses normalised coords with **bottom-left** origin (AppKit convention)
- `detector_yolo.py` flips Y during normalisation; `tracker_faces.py` does the same

## Setup & Running

```bash
# First time
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python main.py
# Or double-click FaceLocker.command in Finder
```

## Dependencies

- `pyobjc-framework-Quartz` — macOS screen capture via CGImage
- `pyobjc-framework-Vision` — (reserved for future Vision.framework path)
- `opencv-python>=4.8.0` — KCF tracking (`cv2.TrackerKCF_create`)
- `ultralytics>=8.0.0` — YOLO inference
- `mss>=9.0.0` — fast cross-platform screen capture

## Git Ignore Notes

`.claude/` is gitignored — CLAUDE.md itself should be tracked, but
`.claude/settings.json` and other harness state should not.

---

## Installed Skills — COMPASS (司南)

Three skills from [dongshuyan/compass-skills](https://github.com/dongshuyan/compass-skills)
are installed under `.agents/skills/` for both Codex and Claude Code.

### `/task-clarifier` — Task Clarifier
**When to use:** Before any non-trivial, ambiguous, costly, or preference-sensitive request.
Asks 1–3 focused questions (each with a recommended answer) to align goals, scope,
constraints, and acceptance criteria before execution. Does NOT fire for trivial edits
or single-fact lookups.

### `/task-forest` — Task Forest
**When to use:** To initialise, update, or query a repo-local task DAG. Maintains
goals, subtasks, dependencies, progress, deviations, todos, decisions, and session
history under `.agent-workbench/task-forest/`. Supports HTML export and live DAG view.
Invoke explicitly — do not auto-run.

### `/user-profile-keeper` — User Profile Keeper
**When to use:** Only when explicitly invoked by the user. Maintains a local,
auditable collaboration profile (communication preferences, risk style, recurring
context) under `~/.compass-skills/user-profiles/v1/`. Purely local plaintext — never
store passwords, tokens, or private keys.

### Skill Pipeline

```
user-profile-keeper  →  task-forest  →  task-clarifier
 (who you are /          (where the task    (what to do
  how to collaborate)     fits in the DAG)   next, concretely)
```

### CLI for task-forest

```bash
# Resolve skill dir from .agents/skills/task-forest/
SKILL_DIR=".agents/skills/task-forest"
python3 $SKILL_DIR/scripts/task_forest.py --help
python3 $SKILL_DIR/scripts/task_forest.py init
python3 $SKILL_DIR/scripts/task_forest.py add --title "Add feature X" --parent root
```

### CLI for user-profile-keeper

```bash
SKILL_DIR=".agents/skills/user-profile-keeper"
python3 $SKILL_DIR/scripts/profile_manager.py --help
```
