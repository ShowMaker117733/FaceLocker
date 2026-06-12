#!/usr/bin/env python3
"""
FaceLocker — Real-time Face Detection Overlay for macOS
========================================================
- Background screen capture (no main-thread blocking)
- GUI control panel: detection ON/OFF, aim-lines ON/OFF, line colour
- Face boxes + optional aim-lines toward the mouse cursor
- YOLO detection + KCF tracking for smooth 40-60 FPS
"""

import signal
import threading
import time

import mss
import numpy as np
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSBackingStoreBuffered,
    NSBorderlessWindowMask,
    NSButton,
    NSClosableWindowMask,
    NSColor,
    NSColorWell,
    NSFloatingWindowLevel,
    NSMakePoint,
    NSMakeRect,
    NSOnState,
    NSOffState,
    NSRectFill,
    NSRoundedBezelStyle,
    NSScreen,
    NSScreenSaverWindowLevel,
    NSSwitchButton,
    NSTextField,
    NSTitledWindowMask,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
)
from Foundation import NSBezierPath, NSObject, NSTimer
import objc
from AppKit import NSEvent

from detector_yolo import YOLODetector
from tracker_faces import FaceTracker

# ---------------------------------------------------------------------------
# Background screen capture  (runs in its own thread, zero main-thread cost)
# ---------------------------------------------------------------------------


class BackgroundCapture:
    """Continuously captures a display in a background thread.

    ``get()`` always returns the latest frame instantly — no blocking.
    """

    def __init__(self, monitor):
        self._monitor = monitor
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        sct = mss.MSS()
        mon = self._monitor
        while self._running:
            img = sct.grab(mon)
            bgr = np.array(img)[:, :, :3]
            with self._lock:
                self._frame = bgr

    def get(self):
        """Return the most recent frame (None if nothing captured yet)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)


# ---------------------------------------------------------------------------
# Overlay view
# ---------------------------------------------------------------------------


class FaceOverlayView(NSView):
    """Draws face boxes and (optionally) aim-lines toward the mouse."""

    def initWithFrame_(self, frame):
        self = objc.super(FaceOverlayView, self).initWithFrame_(frame)
        if self is not None:
            self._faces = []
            self._mouse = None
            self._show_lines = True
            self._line_color = NSColor.greenColor()
            self._box_color = NSColor.greenColor()
        return self

    def setFaces_(self, faces):
        self._faces = list(faces) if faces else []
        self.setNeedsDisplay_(True)

    def setMousePoint_(self, point):
        self._mouse = point
        self.setNeedsDisplay_(True)

    def setShowLines_(self, show):
        self._show_lines = bool(show)
        self.setNeedsDisplay_(True)

    def setLineColor_(self, color):
        self._line_color = color.copy()
        self.setNeedsDisplay_(True)

    def setBoxColor_(self, color):
        self._box_color = color.copy()
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSRectFill(rect)
        if not self._faces:
            return

        vw = self.frame().size.width
        vh = self.frame().size.height

        # ---- face boxes (single path → single stroke) ----
        boxes = NSBezierPath.bezierPath()
        for box in self._faces:
            x = box.origin.x * vw
            y = box.origin.y * vh
            w = box.size.width * vw
            h = box.size.height * vh
            boxes.appendBezierPath_(
                NSBezierPath.bezierPathWithRect_(NSMakeRect(x, y, w, h))
            )
        bc = self._box_color or NSColor.greenColor()
        b_rgb = bc.colorUsingColorSpaceName_("NSDeviceRGBColorSpace") or bc
        NSColor.colorWithRed_green_blue_alpha_(
            b_rgb.redComponent(), b_rgb.greenComponent(),
            b_rgb.blueComponent(), b_rgb.alphaComponent()
        ).set()
        boxes.setLineWidth_(2.5)
        boxes.stroke()

        # ---- aim lines + dots (one path each) ----
        if not self._show_lines or self._mouse is None:
            return

        lc = self._line_color or NSColor.greenColor()
        rgb = lc.colorUsingColorSpaceName_("NSDeviceRGBColorSpace") or lc
        r, g, b, a = (rgb.redComponent(), rgb.greenComponent(),
                       rgb.blueComponent(), rgb.alphaComponent())

        lines = NSBezierPath.bezierPath()
        dots = NSBezierPath.bezierPath()
        for box in self._faces:
            cx = (box.origin.x + box.size.width / 2.0) * vw
            cy = (box.origin.y + box.size.height / 2.0) * vh
            lines.moveToPoint_(NSMakePoint(cx, cy))
            lines.lineToPoint_(self._mouse)
            dots.appendBezierPath_(
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(cx - 3, cy - 3, 6, 6)
                )
            )

        NSColor.colorWithRed_green_blue_alpha_(r, g, b, a * 0.55).set()
        lines.setLineWidth_(1.2)
        lines.stroke()

        NSColor.colorWithRed_green_blue_alpha_(r, g, b, a * 0.9).set()
        dots.fill()


# ---------------------------------------------------------------------------
# Per-display overlay + capture
# ---------------------------------------------------------------------------


class ScreenOverlay:
    def __init__(self, screen, tracker, capture):
        self.screen = screen
        self._tracker = tracker
        self._capture = capture  # BackgroundCapture
        self.view = None
        self.window = None
        self._setup_window()

    def _setup_window(self):
        frame = self.screen.frame()
        self.view = FaceOverlayView.alloc().initWithFrame_(
            NSMakeRect(0, 0, frame.size.width, frame.size.height)
        )
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            NSBorderlessWindowMask,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setLevel_(NSScreenSaverWindowLevel)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setIgnoresMouseEvents_(True)
        self.window.setHasShadow_(False)
        self.window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorIgnoresCycle,
        )
        self.window.setContentView_(self.view)
        self.window.orderFrontRegardless()

    def process_frame(self):
        """Grab latest frame (instant) → track/detect → return faces."""
        bgr = self._capture.get()
        if bgr is None:
            return []
        return self._tracker.update(bgr)

    def update(self, faces):
        self.view.setFaces_(faces)

    def set_show_lines(self, show):
        self.view.setShowLines_(show)

    def set_line_color(self, color):
        self.view.setLineColor_(color)

    def set_box_color(self, color):
        self.view.setBoxColor_(color)

    def show(self):
        self.window.orderFrontRegardless()

    def hide(self):
        self.window.orderOut_(None)

    def close(self):
        if self.window:
            self.window.close()
            self.window = None


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------


class ControlPanel(NSObject):
    """Floating panel: detection toggle, aim-lines toggle, colour picker, status."""

    def init(self):
        self = objc.super(ControlPanel, self).init()
        if self is not None:
            self.window = None
            self._detect_toggle = None
            self._lines_toggle = None
            self._color_well = None
            self._status = None
            self._locker = None
        return self

    def setupWithLocker_(self, locker):
        self._locker = locker

        win_w, win_h = 300, 250
        screen = NSScreen.mainScreen()
        sf = screen.frame()
        wx = sf.origin.x + sf.size.width - win_w - 40
        wy = sf.origin.y + sf.size.height - win_h - 40

        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(wx, wy, win_w, win_h),
            NSTitledWindowMask | NSClosableWindowMask,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_("FaceLocker")
        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setReleasedWhenClosed_(False)

        content = self.window.contentView()

        self._detect_toggle = NSButton.alloc().initWithFrame_(NSMakeRect(18, 210, 264, 28))
        self._detect_toggle.setButtonType_(NSSwitchButton)
        self._detect_toggle.setTitle_("Detection Active")
        self._detect_toggle.setState_(NSOnState)
        self._detect_toggle.setTarget_(self)
        self._detect_toggle.setAction_("detectToggleAction:")
        content.addSubview_(self._detect_toggle)

        self._lines_toggle = NSButton.alloc().initWithFrame_(NSMakeRect(18, 178, 264, 28))
        self._lines_toggle.setButtonType_(NSSwitchButton)
        self._lines_toggle.setTitle_("Show Aim Lines")
        self._lines_toggle.setState_(NSOnState)
        self._lines_toggle.setTarget_(self)
        self._lines_toggle.setAction_("linesToggleAction:")
        content.addSubview_(self._lines_toggle)

        # Box colour
        box_lbl = NSTextField.alloc().initWithFrame_(NSMakeRect(18, 148, 80, 24))
        box_lbl.setStringValue_("Box Color:")
        box_lbl.setEditable_(False)
        box_lbl.setBordered_(False)
        box_lbl.setDrawsBackground_(False)
        content.addSubview_(box_lbl)

        self._box_well = NSColorWell.alloc().initWithFrame_(NSMakeRect(100, 144, 44, 32))
        self._box_well.setColor_(NSColor.greenColor())
        self._box_well.setTarget_(self)
        self._box_well.setAction_("boxColorChanged:")
        content.addSubview_(self._box_well)

        # Line colour
        line_lbl = NSTextField.alloc().initWithFrame_(NSMakeRect(18, 114, 80, 24))
        line_lbl.setStringValue_("Line Color:")
        line_lbl.setEditable_(False)
        line_lbl.setBordered_(False)
        line_lbl.setDrawsBackground_(False)
        content.addSubview_(line_lbl)

        self._line_well = NSColorWell.alloc().initWithFrame_(NSMakeRect(100, 110, 44, 32))
        self._line_well.setColor_(NSColor.greenColor())
        self._line_well.setTarget_(self)
        self._line_well.setAction_("lineColorChanged:")
        content.addSubview_(self._line_well)

        self._status = NSTextField.alloc().initWithFrame_(NSMakeRect(18, 70, 264, 24))
        self._status.setStringValue_("FPS: --  |  Faces: 0")
        self._status.setEditable_(False)
        self._status.setBordered_(False)
        self._status.setDrawsBackground_(False)
        content.addSubview_(self._status)

        quit_btn = NSButton.alloc().initWithFrame_(NSMakeRect(182, 14, 100, 32))
        quit_btn.setTitle_("Quit")
        quit_btn.setBezelStyle_(NSRoundedBezelStyle)
        quit_btn.setTarget_(NSApplication.sharedApplication())
        quit_btn.setAction_("terminate:")
        content.addSubview_(quit_btn)

        self.window.makeKeyAndOrderFront_(None)

    def detectToggleAction_(self, sender):
        self._locker.set_active(sender.state() == NSOnState)

    def linesToggleAction_(self, sender):
        self._locker.set_show_lines(sender.state() == NSOnState)

    def boxColorChanged_(self, sender):
        self._locker.set_box_color(sender.color())

    def lineColorChanged_(self, sender):
        self._locker.set_line_color(sender.color())

    def update_status(self, fps, face_count):
        if self._status is not None:
            self._status.setStringValue_(f"FPS: {fps:.0f}  |  Faces: {face_count}")


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------


class FaceLocker(NSObject):
    TICK_INTERVAL = 1.0 / 60.0   # 60 Hz timer — tracking keeps up, YOLO every 4th

    def init(self):
        self = objc.super(FaceLocker, self).init()
        if self is not None:
            self._timer = None
            self._overlays = []
            self._captures = []
            self._control = None
            self._active = True
            self._show_aim_lines = True
            self._line_color = NSColor.greenColor()
            self._box_color = NSColor.greenColor()
            self._frame_count = 0
            self._fps_time = time.time()
        return self

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_overlays(self):
        print("🔧 Loading YOLO face detection model...")
        yolo = YOLODetector(conf=0.25, iou=0.45, imgsz=640)
        tracker = FaceTracker(yolo, detect_interval=4)

        screens = NSScreen.screens()
        for screen in screens:
            idx = FaceLocker._match_mss_monitor(screen)

            sct = mss.MSS()
            mon = sct.monitors[idx]
            cap = BackgroundCapture(mon)
            cap.start()
            self._captures.append(cap)

            ov = ScreenOverlay(screen, tracker, cap)
            self._overlays.append(ov)

        print(f"✅ {len(screens)} display(s) — background capture active")

    def _setup_control_panel(self):
        self._control = ControlPanel.alloc().init()
        self._control.setupWithLocker_(self)

    @staticmethod
    def _match_mss_monitor(screen):
        frame = screen.frame()
        scale = int(screen.backingScaleFactor())
        primary_h = NSScreen.screens()[0].frame().size.height
        px_left = int(frame.origin.x * scale)
        px_top = int((primary_h - frame.origin.y - frame.size.height) * scale)
        px_w = int(frame.size.width * scale)
        px_h = int(frame.size.height * scale)
        sct = mss.MSS()
        for i, m in enumerate(sct.monitors):
            if i == 0:
                continue
            if (m["left"] == px_left and m["top"] == px_top
                    and m["width"] == px_w and m["height"] == px_h):
                return i
        return 1

    # ------------------------------------------------------------------
    # Control panel callbacks
    # ------------------------------------------------------------------

    def set_active(self, active):
        was = self._active
        self._active = active
        if active and not was:
            for ov in self._overlays:
                ov.show()
        elif not active and was:
            for ov in self._overlays:
                ov.view.setFaces_([])
                ov.hide()

    def set_show_lines(self, show):
        self._show_aim_lines = show
        for ov in self._overlays:
            ov.set_show_lines(show)

    def set_line_color(self, color):
        self._line_color = color
        for ov in self._overlays:
            ov.set_line_color(color)

    def set_box_color(self, color):
        self._box_color = color
        for ov in self._overlays:
            ov.set_box_color(color)

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick_(self, timer):
        total_faces = 0
        mouse_global = NSEvent.mouseLocation()

        if self._active:
            for ov in self._overlays:
                try:
                    faces = ov.process_frame()  # instant grab + track/detect
                    mouse_local = ov.view.convertPoint_fromView_(mouse_global, None)
                    ov.view.setMousePoint_(mouse_local)
                    ov.update(faces)
                    total_faces += len(faces)
                except Exception:
                    pass

        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            self._control.update_status(fps, total_faces)
            if self._active:
                status = f"🎯 {total_faces} face(s)" if total_faces else "👀 scanning..."
                print(f"\r  FaceLocker — {fps:4.0f} FPS | {status}   ",
                      end="", flush=True)
            else:
                print("\r  FaceLocker — paused                          ",
                      end="", flush=True)
            self._frame_count = 0
            self._fps_time = now

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyRegular)

        self._setup_overlays()
        self._setup_control_panel()

        signal.signal(signal.SIGINT, lambda *_: app.terminate_(self))

        self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            self.TICK_INTERVAL, self, "tick:", None, True
        )

        print("=" * 52)
        print("🔒  FaceLocker started")
        print("    Capture   : background thread (non-blocking)")
        print("    Detection : YOLO every 4th tick + KCF tracking")
        print("    Timer     : 60 Hz")
        print("    Ctrl+C    : quit")
        print("=" * 52)

        app.run()

        if self._timer:
            self._timer.invalidate()
        for c in self._captures:
            c.stop()
        for ov in self._overlays:
            ov.close()
        print("\n👋  FaceLocker stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    locker = FaceLocker.alloc().init()
    locker.run()


if __name__ == "__main__":
    main()
