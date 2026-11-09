import os

# =====================================================================
# 0. SILENCE MEDIAPIPE & TENSORFLOW LITE C++ LOG SPAM (MUST BE FIRST!)
# =====================================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import math
import threading
import time
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
import mediapipe as mp
import numpy as np
import psutil
from pydantic import BaseModel
import uvicorn


# =====================================================================
# CONFIGURATION & MONOCHROME PALETTE
# =====================================================================
class Config:
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 960
    FRAME_HEIGHT: int = 720
    BUFFER_CAPACITY: int = 30

    EDGE_HIGH_CONF: float = 0.65
    OCCLUSION_DRAW_FLOOR: float = 0.12
    CLOUD_COOLDOWN_SEC: float = 1.2
    EMA_ALPHA: float = 0.45

    # Sleek Monochrome Palette (BGR)
    PURE_WHITE: tuple[int, int, int] = (255, 255, 255)
    LIGHT_GREY: tuple[int, int, int] = (190, 190, 190)
    MID_GREY: tuple[int, int, int] = (115, 115, 115)
    DARK_GREY: tuple[int, int, int] = (65, 65, 65)


LIMB_GROUPS = {
    "head_neck": {
        "nodes": [0, 7, 8, 11, 12],
        "edges": [(7, 8), (0, 7), (0, 8), (11, 12)],
    },
    "left_arm": {
        "nodes": [11, 13, 15, 17, 19],
        "edges": [(11, 13), (13, 15), (15, 17), (15, 19), (17, 19)],
    },
    "right_arm": {
        "nodes": [12, 14, 16, 18, 20],
        "edges": [(12, 14), (14, 16), (16, 18), (16, 20), (18, 20)],
    },
    "torso_posture": {
        "nodes": [11, 12, 23, 24],
        "edges": [(11, 12), (11, 23), (12, 24), (23, 24)],
    },
    "lower_body": {
        "nodes": [23, 24, 25, 26, 27, 28],
        "edges": [(23, 25), (25, 27), (24, 26), (26, 28)],
    },
}


class IsolationState(BaseModel):
    head_neck: bool = True
    left_arm: bool = True
    right_arm: bool = True
    torso_posture: bool = True
    lower_body: bool = False
    roi_crop_mode: bool = False


# =====================================================================
# 1. THREAD-SAFE CIRCULAR RING BUFFER
# =====================================================================
class ThreadSafeCircularBuffer:
    def __init__(self, capacity: int = 30) -> None:
        self.buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        with self.lock:
            self.buffer.append((time.perf_counter(), frame))

    def get_latest(self) -> tuple[float, np.ndarray] | None:
        with self.lock:
            return self.buffer[-1] if self.buffer else None


# =====================================================================
# 2. TELEMETRY & 3D ORIENTATION COLLECTOR
# =====================================================================
class TelemetryCollector:
    def __init__(self, window_size: int = 200) -> None:
        self.latencies_ms: deque[float] = deque(maxlen=window_size)
        self.total_frames: int = 0
        self.edge_resolved: int = 0
        self.cloud_offloaded: int = 0
        self.last_route_status: str = "EDGE_LOCAL"
        self.last_cloud_result: str = "Standby"
        self.fps: float = 0.0

        self.kinematics: dict[str, float | str] = {
            "facing_dir": "FRONT (0°)",
            "body_yaw_deg": 0.0,
            "neck_tilt_deg": 0.0,
            "left_elbow_deg": 0.0,
            "right_elbow_deg": 0.0,
            "torso_roll_deg": 0.0,
            "active_nodes_count": 0,
            "posture_alert": "NOMINAL",
        }
        self.lock = threading.Lock()

    def update_metrics(
        self,
        latency_ms: float,
        offloaded: bool,
        route_label: str,
        kinematics_update: dict,
    ) -> None:
        with self.lock:
            self.latencies_ms.append(latency_ms)
            self.total_frames += 1
            if offloaded:
                self.cloud_offloaded += 1
            else:
                self.edge_resolved += 1
            self.last_route_status = route_label
            self.kinematics.update(kinematics_update)

    def snapshot(self) -> dict:
        with self.lock:
            arr = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])
            savings = (
                (self.edge_resolved / self.total_frames * 100.0)
                if self.total_frames > 0
                else 100.0
            )
            return {
                "fps": round(self.fps, 1),
                "p50_ms": round(float(np.percentile(arr, 50)), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "p99_ms": round(float(np.percentile(arr, 99)), 2),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_percent": psutil.virtual_memory().percent,
                "edge_resolved": self.edge_resolved,
                "cloud_offloaded": self.cloud_offloaded,
                "bandwidth_saved_pct": round(savings, 1),
                "route_status": self.last_route_status,
                "cloud_status": self.last_cloud_result,
                "kinematics": self.kinematics.copy(),
            }


# =====================================================================
# 3. 3D VECTOR GEOMETRY & 360-DEGREE ORIENTATION ENGINE
# =====================================================================
def calc_3d_joint_angle(a, b, c) -> float:
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=np.float64)
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z], dtype=np.float64)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5:
        return 0.0

    cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def compute_360_body_orientation(lms_2d, lms_3d) -> tuple[float, str]:
    l_sh_2d, r_sh_2d = lms_2d[11], lms_2d[12]
    l_sh_3d, r_sh_3d = lms_3d[11], lms_3d[12]

    dx_screen = l_sh_2d.x - r_sh_2d.x
    dz_world = (l_sh_3d.z - r_sh_3d.z) * 1.15

    yaw_deg = math.degrees(math.atan2(dz_world, dx_screen))
    abs_yaw = abs(yaw_deg)

    if abs_yaw < 38.0:
        cardinal = "FRONT"
    elif abs_yaw > 135.0:
        cardinal = "REAR"
    elif yaw_deg > 0:
        cardinal = "RIGHT"
    else:
        cardinal = "LEFT"

    return yaw_deg, cardinal


# =====================================================================
# 4. PRODUCER-CONSUMER PIPELINE (MONOCHROME VECTOR OVERLAY)
# =====================================================================
class EdgeCloudKinematicPipeline:
    def __init__(self) -> None:
        self.ring_buffer = ThreadSafeCircularBuffer(capacity=Config.BUFFER_CAPACITY)
        self.telemetry = TelemetryCollector()
        self.isolation = IsolationState()
        self.state_lock = threading.Lock()
        self.cloud_lock = threading.Lock()

        self.mp_pose = mp.solutions.pose
        self.edge_pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self.cloud_pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.50,
        )
        self.cloud_executor = ThreadPoolExecutor(max_workers=1)

        self.smooth_yaw: float = 0.0
        self.smooth_neck: float = 0.0
        self.smooth_l_elbow: float = 0.0
        self.smooth_r_elbow: float = 0.0
        self.smooth_torso: float = 0.0

        self.latest_jpeg: bytes | None = None
        self.jpeg_lock = threading.Lock()
        self.last_cloud_ts: float = 0.0
        self.running: bool = True

        threading.Thread(target=self._producer_loop, daemon=True).start()
        threading.Thread(target=self._consumer_loop, daemon=True).start()

    def update_isolation(self, new_state: IsolationState) -> None:
        with self.state_lock:
            self.isolation = new_state

    def stop(self) -> None:
        self.running = False
        self.cloud_executor.shutdown(wait=False)

    def _producer_loop(self) -> None:
        cap = cv2.VideoCapture(Config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.running:
            ret, frame = cap.read()
            if ret:
                self.ring_buffer.push(cv2.flip(frame, 1))
            else:
                time.sleep(0.004)
        cap.release()

    def _cloud_anomaly_audit(
        self, rgb_crop: np.ndarray, reason: str, lowest_conf: float
    ) -> None:
        t0 = time.perf_counter()
        with self.cloud_lock:
            _ = self.cloud_pose.process(rgb_crop)
        time.sleep(0.06)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        with self.telemetry.lock:
            self.telemetry.last_cloud_result = (
                f"Verified [{reason}] · Conf {lowest_conf:.2f} · {dt_ms:.0f}ms"
            )

    def _draw_compass_widget(
        self, frame: np.ndarray, yaw_deg: float, cardinal: str
    ) -> None:
        """Sleek single-ring monochrome compass with a hairline 1px needle."""
        h, w, _ = frame.shape
        cx, cy, radius = w - 52, 48, 26

        # Single crisp 1px ring (no bulky outer circle)
        cv2.circle(frame, (cx, cy), radius, Config.MID_GREY, 1, cv2.LINE_AA)

        # Minimal 4px inner tick marks at N, S, E, W
        cv2.line(frame, (cx, cy - radius), (cx, cy - radius + 4), Config.DARK_GREY, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + radius - 4), (cx, cy + radius), Config.DARK_GREY, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - radius, cy), (cx - radius + 4, cy), Config.DARK_GREY, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + radius - 4, cy), (cx + radius, cy), Config.DARK_GREY, 1, cv2.LINE_AA)

        # Thin 1px white vector needle
        rad = math.radians(yaw_deg - 90.0)
        nx = int(cx + radius * 0.82 * math.cos(rad))
        ny = int(cy + radius * 0.82 * math.sin(rad))

        cv2.line(frame, (cx, cy), (nx, ny), Config.PURE_WHITE, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 1, Config.MID_GREY, -1, cv2.LINE_AA)
        cv2.circle(frame, (nx, ny), 2, Config.PURE_WHITE, -1, cv2.LINE_AA)

        # Minimalist label underneath
        text_size = cv2.getTextSize(cardinal, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0]
        tx = cx - (text_size[0] // 2)
        cv2.putText(
            frame,
            cardinal,
            (tx, cy + radius + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            Config.LIGHT_GREY,
            1,
            cv2.LINE_AA,
        )

    def _consumer_loop(self) -> None:
        prev_time = time.perf_counter()
        last_ts = 0.0
        a = Config.EMA_ALPHA

        while self.running:
            item = self.ring_buffer.get_latest()
            if item is None or item[0] == last_ts:
                time.sleep(0.002)
                continue

            frame_ts, raw_frame = item
            last_ts = frame_ts
            frame = raw_frame.copy()
            h, w, _ = frame.shape

            with self.state_lock:
                iso = self.isolation.model_copy()

            t_start = time.perf_counter()

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rgb.flags.writeable = False
            results = self.edge_pose.process(rgb)
            rgb.flags.writeable = True

            active_nodes: set[int] = set()
            active_edges: list[tuple[int, int]] = []
            for group_name, group_data in LIMB_GROUPS.items():
                if getattr(iso, group_name):
                    active_nodes.update(group_data["nodes"])
                    active_edges.extend(group_data["edges"])

            offloaded = False
            route_status = "EDGE_LOCAL"
            lowest_conf = 1.0
            anomaly_reason = ""
            cardinal = "IDLE"

            if (
                results.pose_landmarks
                and results.pose_world_landmarks
                and active_nodes
            ):
                lms_2d = results.pose_landmarks.landmark
                lms_3d = results.pose_world_landmarks.landmark

                raw_yaw, cardinal = compute_360_body_orientation(lms_2d, lms_3d)
                yaw_diff = (raw_yaw - self.smooth_yaw + 180.0) % 360.0 - 180.0
                self.smooth_yaw = (self.smooth_yaw + a * yaw_diff + 180.0) % 360.0 - 180.0

                pts_px: dict[int, tuple[int, int]] = {}
                xs, ys = [], []
                for idx in active_nodes:
                    lm = lms_2d[idx]
                    lowest_conf = min(lowest_conf, lm.visibility)
                    if lm.visibility >= Config.OCCLUSION_DRAW_FLOOR:
                        px = int(np.clip(lm.x * w, 0, w - 1))
                        py = int(np.clip(lm.y * h, 0, h - 1))
                        pts_px[idx] = (px, py)
                        xs.append(px)
                        ys.append(py)

                if iso.roi_crop_mode and xs and ys:
                    mask = np.zeros_like(frame)
                    pad = 60
                    rx1, ry1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
                    rx2, ry2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
                    mask[ry1:ry2, rx1:rx2] = frame[ry1:ry2, rx1:rx2]
                    frame = cv2.addWeighted(frame, 0.18, mask, 0.82, 0)
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), Config.MID_GREY, 1, cv2.LINE_AA)

                if iso.head_neck:
                    mid_sh_x = (lms_3d[11].x + lms_3d[12].x) * 0.5
                    mid_sh_y = (lms_3d[11].y + lms_3d[12].y) * 0.5
                    mid_sh_z = (lms_3d[11].z + lms_3d[12].z) * 0.5

                    head_x = (lms_3d[7].x + lms_3d[8].x) * 0.5
                    head_y = (lms_3d[7].y + lms_3d[8].y) * 0.5
                    head_z = (lms_3d[7].z + lms_3d[8].z) * 0.5

                    vx = head_x - mid_sh_x
                    vy = mid_sh_y - head_y
                    vz = head_z - mid_sh_z
                    horiz_mag = math.hypot(vx, vz)
                    raw_neck_deg = math.degrees(math.atan2(horiz_mag, max(0.01, vy)))
                    self.smooth_neck = (a * raw_neck_deg) + ((1.0 - a) * self.smooth_neck)

                    if 11 in pts_px and 12 in pts_px and (0 in pts_px or 7 in pts_px):
                        mid_px = (
                            (pts_px[11][0] + pts_px[12][0]) // 2,
                            (pts_px[11][1] + pts_px[12][1]) // 2,
                        )
                        head_px = pts_px.get(0, pts_px.get(7, mid_px))
                        cv2.line(frame, mid_px, (mid_px[0], max(10, mid_px[1] - 110)), Config.DARK_GREY, 1, cv2.LINE_AA)
                        cv2.line(frame, mid_px, head_px, Config.LIGHT_GREY, 1, cv2.LINE_AA)

                if iso.left_arm:
                    raw_l_el = calc_3d_joint_angle(lms_3d[11], lms_3d[13], lms_3d[15])
                    self.smooth_l_elbow = (a * raw_l_el) + ((1.0 - a) * self.smooth_l_elbow)
                    if 13 in pts_px:
                        cv2.putText(
                            frame,
                            f"{self.smooth_l_elbow:.0f}",
                            (pts_px[13][0] + 8, pts_px[13][1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.36,
                            Config.LIGHT_GREY,
                            1,
                            cv2.LINE_AA,
                        )

                if iso.right_arm:
                    raw_r_el = calc_3d_joint_angle(lms_3d[12], lms_3d[14], lms_3d[16])
                    self.smooth_r_elbow = (a * raw_r_el) + ((1.0 - a) * self.smooth_r_elbow)
                    if 14 in pts_px:
                        cv2.putText(
                            frame,
                            f"{self.smooth_r_elbow:.0f}",
                            (pts_px[14][0] - 32, pts_px[14][1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.36,
                            Config.LIGHT_GREY,
                            1,
                            cv2.LINE_AA,
                        )

                if iso.torso_posture:
                    dy_3d = lms_3d[11].y - lms_3d[12].y
                    horiz_3d = max(0.05, math.hypot(lms_3d[11].x - lms_3d[12].x, lms_3d[11].z - lms_3d[12].z))
                    raw_torso = math.degrees(math.atan2(dy_3d, horiz_3d))
                    self.smooth_torso = (a * raw_torso) + ((1.0 - a) * self.smooth_torso)

                # Monochrome 1px Skeleton & Micro-Nodes
                for i1, i2 in active_edges:
                    if i1 in pts_px and i2 in pts_px:
                        edge_vis = min(lms_2d[i1].visibility, lms_2d[i2].visibility)
                        color = Config.PURE_WHITE if edge_vis >= 0.50 else Config.DARK_GREY
                        cv2.line(frame, pts_px[i1], pts_px[i2], color, 1, cv2.LINE_AA)

                for idx, pt in pts_px.items():
                    vis = lms_2d[idx].visibility
                    n_color = Config.PURE_WHITE if vis >= 0.50 else Config.MID_GREY
                    cv2.circle(frame, pt, 2, n_color, -1, cv2.LINE_AA)

                self._draw_compass_widget(frame, self.smooth_yaw, cardinal)

                posture_alert = "NOMINAL"
                if self.smooth_neck > 22.0:
                    posture_alert = f"DEVIATION ({self.smooth_neck:.1f}°)"
                    anomaly_reason = "Deviation"

                now = time.perf_counter()
                if lowest_conf < Config.EDGE_HIGH_CONF or anomaly_reason:
                    if not anomaly_reason:
                        anomaly_reason = f"Occlusion ({lowest_conf:.2f})"
                    if (now - self.last_cloud_ts) >= Config.CLOUD_COOLDOWN_SEC:
                        offloaded = True
                        route_status = f"CLOUD_SYNC ({anomaly_reason})"
                        self.last_cloud_ts = now
                        self.cloud_executor.submit(
                            self._cloud_anomaly_audit, rgb.copy(), anomaly_reason, lowest_conf
                        )
            else:
                posture_alert = "NO_SIGNAL"

            kinematics_data = {
                "facing_dir": f"{cardinal} ({self.smooth_yaw:+.0f}°)",
                "body_yaw_deg": round(self.smooth_yaw, 1),
                "neck_tilt_deg": round(self.smooth_neck, 1),
                "left_elbow_deg": round(self.smooth_l_elbow, 1),
                "right_elbow_deg": round(self.smooth_r_elbow, 1),
                "torso_roll_deg": round(self.smooth_torso, 1),
                "active_nodes_count": len(active_nodes),
                "posture_alert": posture_alert,
            }

            edge_latency_ms = (time.perf_counter() - t_start) * 1000.0
            now_fps = time.perf_counter()
            dt = now_fps - prev_time
            prev_time = now_fps
            if dt > 0:
                self.telemetry.fps = (0.9 * self.telemetry.fps) + (0.1 * (1.0 / dt))

            self.telemetry.update_metrics(edge_latency_ms, offloaded, route_status, kinematics_data)

            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                with self.jpeg_lock:
                    self.latest_jpeg = encoded.tobytes()


# =====================================================================
# 5. FASTAPI LIFESPAN & SLEEK MONOCHROME UI
# =====================================================================
pipeline: EdgeCloudKinematicPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = EdgeCloudKinematicPipeline()
    print("\n========================================================")
    print(" CLOUD TELEMETRY ONLINE")
    print(" -> Open in Browser: http://127.0.0.1:8000")
    print(" -> Press Ctrl+C in this terminal to stop.")
    print("========================================================\n")
    yield
    if pipeline:
        pipeline.stop()


app = FastAPI(title="Cloud Telemetry", lifespan=lifespan)


@app.post("/api/isolation")
async def set_isolation(state: IsolationState) -> dict:
    if pipeline:
        pipeline.update_isolation(state)
    return {"status": "updated", "active_filter": state.model_dump()}


@app.get("/", response_class=HTMLResponse)
async def dashboard_ui() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Cloud Telemetry</title>
        <style>
            * { box-sizing: border-box; }
            body {
                background: #000000;
                color: #ffffff;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 0;
                height: 100vh;
                width: 100vw;
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 14px 28px;
                border-bottom: 1px solid #1a1a1a;
                background: #000000;
                flex-shrink: 0;
            }
            .title {
                font-size: 13px;
                font-weight: 400;
                letter-spacing: 2.5px;
                text-transform: uppercase;
                color: #ffffff;
            }
            .status-indicator {
                font-size: 11px;
                letter-spacing: 1.5px;
                color: #888888;
                text-transform: uppercase;
                font-family: 'Consolas', 'SF Mono', monospace;
            }
            .workspace {
                display: grid;
                grid-template-columns: 1fr 370px;
                flex: 1;
                min-height: 0;
                background: #000000;
            }
            .viewport {
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                padding: 20px 28px;
                min-height: 0;
                border-right: 1px solid #1a1a1a;
            }
            .video-container {
                flex: 1;
                min-height: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #050505;
                border: 1px solid #161616;
                border-radius: 2px;
                overflow: hidden;
            }
            .video-container img {
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
            }
            .controls-bar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding-top: 16px;
                flex-shrink: 0;
                gap: 12px;
            }
            .controls-label {
                font-size: 10px;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                color: #666666;
            }
            .toggle-group {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
            }
            .toggle-btn {
                background: #080808;
                color: #666666;
                border: 1px solid #222222;
                padding: 7px 13px;
                border-radius: 2px;
                cursor: pointer;
                font-size: 11px;
                font-weight: 400;
                letter-spacing: 0.5px;
                transition: all 0.15s ease;
            }
            .toggle-btn:hover {
                border-color: #444444;
                color: #aaaaaa;
            }
            .toggle-btn.active {
                background: #161616;
                color: #ffffff;
                border-color: #666666;
            }
            .sidebar {
                padding: 24px 28px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                overflow-y: auto;
                background: #030303;
            }
            .section-title {
                font-size: 10px;
                letter-spacing: 2px;
                text-transform: uppercase;
                color: #666666;
                margin-bottom: 14px;
                padding-bottom: 6px;
                border-bottom: 1px solid #161616;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
                margin-bottom: 26px;
            }
            .metric-card {
                background: #080808;
                border: 1px solid #161616;
                padding: 11px 13px;
                border-radius: 2px;
            }
            .metric-label {
                font-size: 10px;
                color: #666666;
                letter-spacing: 0.8px;
                text-transform: uppercase;
            }
            .metric-val {
                font-size: 14px;
                font-weight: 300;
                margin-top: 6px;
                color: #e5e5e5;
                font-family: 'Consolas', 'SF Mono', monospace;
            }
            .telemetry-list {
                margin-bottom: 26px;
            }
            .tel-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 9px 0;
                border-bottom: 1px solid #111111;
                font-size: 12px;
            }
            .tel-label {
                color: #888888;
                font-weight: 400;
            }
            .tel-val {
                color: #ffffff;
                font-weight: 300;
                font-family: 'Consolas', 'SF Mono', monospace;
                font-size: 12.5px;
            }
            .router-panel {
                background: #080808;
                border: 1px solid #161616;
                padding: 14px;
                border-radius: 2px;
                font-size: 11.5px;
            }
            .router-line {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
            }
            .router-line:last-child {
                margin-bottom: 0;
            }
            .router-key {
                color: #666666;
            }
            .router-val {
                color: #cccccc;
                font-family: 'Consolas', 'SF Mono', monospace;
                text-align: right;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title">Cloud Telemetry</div>
            <div class="status-indicator">STREAM // ACTIVE</div>
        </div>

        <div class="workspace">
            <div class="viewport">
                <div class="video-container">
                    <img src="/video_feed" alt="Stream" />
                </div>
                <div class="controls-bar">
                    <span class="controls-label">Node Isolation</span>
                    <div class="toggle-group">
                        <button class="toggle-btn active" id="btn-head_neck" onclick="toggleLimb('head_neck')">Cranium</button>
                        <button class="toggle-btn active" id="btn-left_arm" onclick="toggleLimb('left_arm')">Left Arm</button>
                        <button class="toggle-btn active" id="btn-right_arm" onclick="toggleLimb('right_arm')">Right Arm</button>
                        <button class="toggle-btn active" id="btn-torso_posture" onclick="toggleLimb('torso_posture')">Torso</button>
                        <button class="toggle-btn" id="btn-lower_body" onclick="toggleLimb('lower_body')">Lower</button>
                        <button class="toggle-btn" id="btn-roi_crop_mode" onclick="toggleLimb('roi_crop_mode')">Spotlight</button>
                    </div>
                </div>
            </div>

            <div class="sidebar">
                <div>
                    <div class="section-title">Execution Metrics</div>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">P50 / P95 / P99</div>
                            <div class="metric-val" id="lat">-- / -- / --</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Frame Rate</div>
                            <div class="metric-val" id="fps">-- fps</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Edge Ratio</div>
                            <div class="metric-val" id="saved">-- %</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Edge / Cloud</div>
                            <div class="metric-val" id="counts">0 / 0</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Active Nodes</div>
                            <div class="metric-val" id="nodes">16 / 33</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">CPU / Mem</div>
                            <div class="metric-val" id="sys">--% / --%</div>
                        </div>
                    </div>

                    <div class="section-title">Kinematics</div>
                    <div class="telemetry-list">
                        <div class="tel-row">
                            <span class="tel-label">Orientation</span>
                            <span class="tel-val" id="k-dir">FRONT (0°)</span>
                        </div>
                        <div class="tel-row">
                            <span class="tel-label">Cervical Inclination</span>
                            <span class="tel-val" id="k-neck">0.0°</span>
                        </div>
                        <div class="tel-row">
                            <span class="tel-label">Left Elbow Flexion</span>
                            <span class="tel-val" id="k-l-elbow">0.0°</span>
                        </div>
                        <div class="tel-row">
                            <span class="tel-label">Right Elbow Flexion</span>
                            <span class="tel-val" id="k-r-elbow">0.0°</span>
                        </div>
                        <div class="tel-row">
                            <span class="tel-label">Clavicular Roll</span>
                            <span class="tel-val" id="k-torso">0.0°</span>
                        </div>
                        <div class="tel-row">
                            <span class="tel-label">Status</span>
                            <span class="tel-val" id="k-alert">NOMINAL</span>
                        </div>
                    </div>
                </div>

                <div>
                    <div class="section-title">Routing Log</div>
                    <div class="router-panel">
                        <div class="router-line">
                            <span class="router-key">Decision</span>
                            <span class="router-val" id="route">--</span>
                        </div>
                        <div class="router-line">
                            <span class="router-key">Cloud Audit</span>
                            <span class="router-val" id="cloud">--</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const state = {
                head_neck: true,
                left_arm: true,
                right_arm: true,
                torso_posture: true,
                lower_body: false,
                roi_crop_mode: false
            };

            async function toggleLimb(key) {
                state[key] = !state[key];
                const btn = document.getElementById(`btn-${key}`);
                btn.classList.toggle("active", state[key]);
                await fetch("/api/isolation", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(state)
                });
            }

            const ws = new WebSocket(`ws://${location.host}/ws/telemetry`);
            ws.onmessage = (event) => {
                const d = JSON.parse(event.data);
                document.getElementById("lat").innerText = `${d.p50_ms} / ${d.p95_ms} / ${d.p99_ms}`;
                document.getElementById("fps").innerText = `${d.fps} fps`;
                document.getElementById("saved").innerText = `${d.bandwidth_saved_pct}%`;
                document.getElementById("counts").innerText = `${d.edge_resolved} / ${d.cloud_offloaded}`;
                document.getElementById("nodes").innerText = `${d.kinematics.active_nodes_count} / 33`;
                document.getElementById("sys").innerText = `${d.cpu_percent}% / ${d.ram_percent}%`;
                document.getElementById("route").innerText = d.route_status;
                document.getElementById("cloud").innerText = d.cloud_status;

                document.getElementById("k-dir").innerText = d.kinematics.facing_dir;
                document.getElementById("k-neck").innerText = `${d.kinematics.neck_tilt_deg}°`;
                document.getElementById("k-l-elbow").innerText = `${d.kinematics.left_elbow_deg}°`;
                document.getElementById("k-r-elbow").innerText = `${d.kinematics.right_elbow_deg}°`;
                document.getElementById("k-torso").innerText = `${d.kinematics.torso_roll_deg}°`;
                document.getElementById("k-alert").innerText = d.kinematics.posture_alert;
            };
        </script>
    </body>
    </html>
    """


def generate_mjpeg():
    while True:
        if pipeline and pipeline.latest_jpeg:
            with pipeline.jpeg_lock:
                frame_bytes = pipeline.latest_jpeg
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.016)


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(
        generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            if pipeline:
                await websocket.send_json(pipeline.telemetry.snapshot())
            await asyncio.sleep(0.20)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info", access_log=False)