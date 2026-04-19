"""
Real-Time Object Detection Streaming Client
Captures webcam frames → sends to server → displays annotated results
"""

import asyncio
import base64
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("yolo-client")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
SERVER_URI       = "ws://localhost:8765/ws/stream"
CAMERA_INDEX     = 0
CAPTURE_WIDTH    = 640
CAPTURE_HEIGHT   = 480
TARGET_FPS       = 30               # capture rate
SEND_EVERY_N     = 2                # skip N-1 frames between sends
JPEG_QUALITY     = 75               # encode quality before send
RECONNECT_DELAY  = 3.0              # seconds between reconnect attempts
DISPLAY_WINDOW   = "YOLO Detection Stream  [Q to quit]"
LATENCY_WINDOW   = 60               # samples for rolling average


# ──────────────────────────────────────────────
# Shared state between threads
# ──────────────────────────────────────────────
@dataclass
class SharedState:
    # Latest annotated frame from server (BGR numpy array)
    annotated_frame: Optional[np.ndarray] = None
    # Raw capture frame (for display when no annotated yet)
    raw_frame: Optional[np.ndarray] = None

    # Stats
    send_fps:         float = 0.0
    recv_fps:         float = 0.0
    latency_ms:       float = 0.0
    inference_ms:     float = 0.0
    num_objects:      int   = 0
    frames_sent:      int   = 0
    frames_received:  int   = 0
    frames_dropped:   int   = 0
    connected:        bool  = False

    latency_samples:  deque = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_annotated(self, frame: np.ndarray, meta: dict):
        with self._lock:
            self.annotated_frame = frame
            self.inference_ms    = meta.get("inference_ms", 0)
            self.num_objects     = meta.get("num_objects", 0)
            rtt = meta.get("round_trip_ms", 0)
            self.latency_samples.append(rtt)
            self.latency_ms    = sum(self.latency_samples) / len(self.latency_samples)
            self.frames_received += 1

    def get_display_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self.annotated_frame is not None:
                return self.annotated_frame.copy()
            if self.raw_frame is not None:
                return self.raw_frame.copy()
            return None


state = SharedState()


# ──────────────────────────────────────────────
# FPS Counter
# ──────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self):
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0] + 1e-9)


send_fps_counter = FPSCounter()
recv_fps_counter = FPSCounter()


# ──────────────────────────────────────────────
# Camera Capture Thread
# ──────────────────────────────────────────────
class CaptureThread(threading.Thread):
    """Continuously reads frames from the webcam into a single-slot buffer."""

    def __init__(self):
        super().__init__(daemon=True, name="capture")
        self._frame: Optional[np.ndarray] = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

        if not cap.isOpened():
            logger.error("Cannot open camera %d", CAMERA_INDEX)
            return

        logger.info("Camera opened  %dx%d @ %dfps",
                    CAPTURE_WIDTH, CAPTURE_HEIGHT, TARGET_FPS)

        interval = 1.0 / TARGET_FPS
        while not self._stop.is_set():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                with state._lock:
                    state.raw_frame = frame
            sleep = interval - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

        cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop.set()


# ──────────────────────────────────────────────
# WebSocket Send/Receive (async)
# ──────────────────────────────────────────────
async def receive_loop(ws):
    """Receive annotated frames from server."""
    async for message in ws:
        try:
            msg = json.loads(message)
            if msg.get("type") != "annotated_frame":
                continue

            img_bytes = base64.b64decode(msg["image"])
            arr       = np.frombuffer(img_bytes, dtype=np.uint8)
            frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            state.update_annotated(frame, msg)
            recv_fps_counter.tick()
            state.recv_fps = recv_fps_counter.fps

        except Exception as exc:
            logger.warning("Receive error: %s", exc)


async def send_loop(ws, capture: CaptureThread):
    """Capture → encode → send frames with frame-skip strategy."""
    frame_idx    = 0
    interval     = 1.0 / TARGET_FPS

    while True:
        t0    = time.perf_counter()
        frame = capture.get_frame()

        if frame is not None:
            frame_idx += 1
            if frame_idx % SEND_EVERY_N == 0:
                ok, buf = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if ok:
                    payload = json.dumps({
                        "type":      "frame",
                        "image":     base64.b64encode(buf.tobytes()).decode(),
                        "timestamp": time.time(),
                    })
                    try:
                        await ws.send(payload)
                        state.frames_sent += 1
                        send_fps_counter.tick()
                        state.send_fps = send_fps_counter.fps
                    except ConnectionClosed:
                        return

        elapsed = time.perf_counter() - t0
        await asyncio.sleep(max(0.0, interval - elapsed))


async def run_client(capture: CaptureThread):
    """Main async loop: connect, stream, reconnect on failure."""
    while True:
        try:
            logger.info("Connecting to %s …", SERVER_URI)
            async with websockets.connect(
                SERVER_URI,
                ping_interval=20,
                ping_timeout=30,
                max_size=10 * 1024 * 1024,   # 10 MB
            ) as ws:
                state.connected = True
                logger.info("Connected ✓")
                await asyncio.gather(
                    send_loop(ws, capture),
                    receive_loop(ws),
                )
        except (ConnectionClosed, OSError) as exc:
            state.connected = False
            logger.warning("Connection lost (%s) – retrying in %.1fs …",
                           exc, RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as exc:
            state.connected = False
            logger.error("Unexpected error: %s", exc)
            await asyncio.sleep(RECONNECT_DELAY)


# ──────────────────────────────────────────────
# HUD overlay
# ──────────────────────────────────────────────
def draw_hud(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent black bar at top
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    status_color = (0, 220, 80) if state.connected else (0, 60, 220)
    status_text  = "● LIVE" if state.connected else "● OFFLINE"

    def put(text, x, y, scale=0.55, color=(220, 220, 220), thickness=1):
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness,
                    cv2.LINE_AA)

    put(status_text,                          10, 25, 0.65, status_color, 2)
    put(f"Send  {state.send_fps:5.1f} fps",   10, 50, 0.52)
    put(f"Recv  {state.recv_fps:5.1f} fps",   10, 72, 0.52)

    put(f"Latency   {state.latency_ms:6.1f} ms",  200, 30, 0.55, (180, 220, 255))
    put(f"Inference {state.inference_ms:6.1f} ms", 200, 52, 0.55, (180, 220, 255))
    put(f"Objects   {state.num_objects:3d}",       200, 74, 0.55, (180, 220, 255))

    put(f"Sent {state.frames_sent}  |  Rcvd {state.frames_received}",
        w - 270, 30, 0.50, (160, 160, 160))
    put(f"Skip 1/{SEND_EVERY_N}  |  Q={JPEG_QUALITY}%",
        w - 270, 52, 0.50, (160, 160, 160))

    return frame


# ──────────────────────────────────────────────
# Display Thread
# ──────────────────────────────────────────────
def display_thread_fn():
    cv2.namedWindow(DISPLAY_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISPLAY_WINDOW, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    while True:
        frame = state.get_display_frame()
        if frame is not None:
            frame = draw_hud(frame)
            cv2.imshow(DISPLAY_WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            logger.info("User quit")
            break

    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    capture = CaptureThread()
    capture.start()

    # Run display in main thread (required by OpenCV on macOS/Windows)
    display = threading.Thread(target=display_thread_fn, daemon=True)
    display.start()

    # Run async send/receive in a background thread
    def async_runner():
        asyncio.run(run_client(capture))

    net_thread = threading.Thread(target=async_runner, daemon=True, name="ws-client")
    net_thread.start()

    display.join()          # blocks until user presses Q
    capture.stop()
    logger.info("Client shut down.")


if __name__ == "__main__":
    main()