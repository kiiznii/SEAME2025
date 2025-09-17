# common/lk_algo.py
# ORIGINAL-STYLE lane-keeping helpers extracted from LK_0903_1.py
# - HoughP → slope/intercept avg per side
# - Construct lane endpoints at y_bottom=0.95*h, y_top=0.70*h
# - Fuse with short-term memory
# - Speed-based gains/clip + lookahead ratio

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import math
import numpy as np
import cv2

# -------------------- Geometry / Utils --------------------

def x_at_y(m: float, b: float, y: float) -> float:
    """Return x coordinate on the line y = m*x + b at given y."""
    denom = m if abs(m) > 1e-9 else 1e-9
    return (y - b) / denom

def x_at_y_(line: Tuple[int, int, int, int], y: int) -> int:
    """Return x along a finite segment (x1,y1,x2,y2) at given y (linear interp)."""
    x1, y1, x2, y2 = line
    if y2 == y1:
        return int(x1)
    t = (y - y1) / float(y2 - y1)
    return int(round(x1 + t * (x2 - x1)))

def lane_mid_x(left_line: Tuple[int,int,int,int],
               right_line: Tuple[int,int,int,int],
               y: int) -> Tuple[int,int,int]:
    """Return (center_x, left_x, right_x) at scanline y."""
    xl = x_at_y_(left_line, y)
    xr = x_at_y_(right_line, y)
    return (xl + xr) // 2, xl, xr

def speed_of(vehicle) -> float:
    """CARLA vehicle speed in m/s."""
    if vehicle is None:
        return 0.0
    v = vehicle.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

# -------------------- Lane Detection (original style) --------------------

def detect_lanes_and_center(
    bgr: np.ndarray,
    roi_vertices: Optional[np.ndarray] = None,
    canny_low: int = 60, canny_high: int = 180,
    hough_thresh: int = 40, hough_min_len: int = 40, hough_max_gap: int = 120,
) -> Dict:
    """
    Return dict with:
      edges, masked, roi_vertices, line_segs,
      left_line, right_line, center_line, center_x_bottom

    left_line/right_line are (x1,y1,x2,y2) built from averaged slope/intercept
    and evaluated at y_bottom=0.95*h, y_top=0.70*h.
    """
    h, w = bgr.shape[:2]

    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    if roi_vertices is None:
        roi_vertices = np.array([[
            (int(0.08 * w), h),
            (int(0.38 * w), int(0.6 * h)),
            (int(0.62 * w), int(0.6 * h)),
            (int(0.92 * w), h),
        ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked, 1, np.pi/180,
        threshold=hough_thresh,
        minLineLength=hough_min_len,
        maxLineGap=hough_max_gap,
    )

    left_params: List[Tuple[float, float]] = []
    right_params: List[Tuple[float, float]] = []
    line_segs: List[Tuple[int,int,int,int]] = []

    if lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0, :]:
            if x2 == x1:
                continue
            m = (y2 - y1) / float(x2 - x1)
            b = y1 - m * x1
            if m < 0 and x1 < w*0.5 and x2 < w*0.5:
                left_params.append((m, b))
            elif m > 0 and x1 > w*0.5 and x2 > w*0.5:
                right_params.append((m, b))
            line_segs.append((int(x1), int(y1), int(x2), int(y2)))

    # Build representative lines at two vertical positions
    y_bottom = int(0.95 * h)
    y_top    = int(0.67 * h) # prev -> 0.70

    left_line_pts: Optional[Tuple[int,int,int,int]] = None
    right_line_pts: Optional[Tuple[int,int,int,int]] = None

    if left_params:
        ms, bs = zip(*left_params)
        m_avg, b_avg = float(np.mean(ms)), float(np.mean(bs))
        lx1, lx2 = int(x_at_y(m_avg, b_avg, y_bottom)), int(x_at_y(m_avg, b_avg, y_top))
        left_line_pts = (lx1, y_bottom, lx2, y_top)

    if right_params:
        ms, bs = zip(*right_params)
        m_avg, b_avg = float(np.mean(ms)), float(np.mean(bs))
        rx1, rx2 = int(x_at_y(m_avg, b_avg, y_bottom)), int(x_at_y(m_avg, b_avg, y_top))
        right_line_pts = (rx1, y_bottom, rx2, y_top)

    center_line = None
    center_x_bottom = None
    if left_line_pts and right_line_pts:
        cx1 = (left_line_pts[0] + right_line_pts[0]) // 2
        cx2 = (left_line_pts[2] + right_line_pts[2]) // 2
        center_x_bottom = cx1
        center_line = (cx1, y_bottom, cx2, y_top)

    return dict(
        edges=edges,
        masked=masked,
        roi_vertices=roi_vertices,
        line_segs=line_segs,
        left_line=left_line_pts,
        right_line=right_line_pts,
        center_line=center_line,
        center_x_bottom=center_x_bottom,
    )

# -------------------- Memory fuse --------------------

def fuse_lanes_with_memory(lanes: Dict, frame_id: int, mem: Dict, ttl_frames: int = 80):
    """
    mem keys: "left","right","t_left","t_right","center_x"
    Returns: used_left, used_right, center_line, center_x
    """
    if lanes["left_line"] is not None:
        mem["left"] = lanes["left_line"];   mem["t_left"]  = frame_id
    if lanes["right_line"] is not None:
        mem["right"] = lanes["right_line"]; mem["t_right"] = frame_id

    def alive(ts): return (ts is not None) and (frame_id - ts <= ttl_frames)

    used_left  = lanes["left_line"]  if lanes["left_line"]  is not None else (mem.get("left")  if alive(mem.get("t_left"))  else None)
    used_right = lanes["right_line"] if lanes["right_line"] is not None else (mem.get("right") if alive(mem.get("t_right")) else None)

    center_line = None
    center_x    = None
    if used_left and used_right:
        lx1, ly1, lx2, ly2 = used_left
        rx1, ry1, rx2, ry2 = used_right
        cx1 = (lx1 + rx1) // 2
        cx2 = (lx2 + rx2) // 2
        center_x = cx1
        center_line = (cx1, ly1, cx2, ly2)
        mem["center_x"] = center_x
    else:
        center_x = mem.get("center_x", None)

    return used_left, used_right, center_line, center_x

# -------------------- Control helpers --------------------

def gains_for_speed(v_mps: float):
    """Return (kp, clip) based on speed in m/s (converted to km/h)."""
    v = v_mps * 3.6
    if v <= 25:
        return 0.0030, 0.15
    elif v <= 50:
        return 0.0020, 0.13
    else:
        return 0.0010, 0.10

def lookahead_ratio(v_kmh: float) -> float:
    """
    y_anchor / h ratio:
      0.90 @<=25 → 0.80 @50 → 0.70 @75 → 0.62 @>=100
    """
    if v_kmh <= 25.0:
        return 0.90
    if v_kmh <= 50.0:
        t = (v_kmh - 25.0) / (50.0 - 25.0)
        return 0.90*(1.0 - t) + 0.70*t
    if v_kmh <= 75.0:
        t = (v_kmh - 50.0) / (75.0 - 50.0)
        return 0.80*(1.0 - t) + 0.65*t
    if v_kmh <= 100.0:
        t = (v_kmh - 75.0) / (100.0 - 75.0)
        return 0.70*(1.0 - t) + 0.62*t
    return 0.62

