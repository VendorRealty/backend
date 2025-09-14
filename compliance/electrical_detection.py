"""
Electrical symbol detection utilities.
Approximate detection of common plan symbols:
- Duplex receptacles (circle near wall, small radius)
- Ceiling lights (circle in room interior, medium radius)
- Switches (letter 'S' by OCR, near wall)

This is a pragmatic first pass to get electrical working; it follows contours by
snapping detected devices to the nearest wall segment when drawing.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import math
import numpy as np
import cv2
from PIL import Image
import os

try:
    import easyocr  # type: ignore
except Exception:  # optional dependency
    easyocr = None  # type: ignore


def _non_max_suppression_points(points: List[Tuple[float, float, float]], min_dist: float) -> List[Tuple[float, float, float]]:
    """Greedy distance-based NMS for circles: (x, y, r)."""
    if not points:
        return []
    pts = sorted(points, key=lambda p: p[2], reverse=True)  # larger radius first
    kept: List[Tuple[float, float, float]] = []
    for x, y, r in pts:
        ok = True
        for X, Y, R in kept:
            if (X - x) * (X - x) + (Y - y) * (Y - y) < min_dist * min_dist:
                ok = False
                break
        if ok:
            kept.append((x, y, r))
    return kept


def _distance_to_edges(edges: np.ndarray, x: float, y: float, max_search: int = 8) -> float:
    """Approximate distance in pixels from (x,y) to the nearest edge pixel using a small window."""
    h, w = edges.shape
    xi, yi = int(round(x)), int(round(y))
    r = int(max(1, max_search))
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    window = edges[y0:y1, x0:x1]
    ys, xs = np.where(window > 0)
    if xs.size == 0:
        return float('inf')
    xs = xs + x0
    ys = ys + y0
    d2 = (xs - x) * (xs - x) + (ys - y) * (ys - y)
    return float(math.sqrt(float(d2.min())))


def detect_electrical_symbols(background: Image.Image, wall_mask: np.ndarray) -> Dict[str, List[Dict]]:
    """Detect electrical devices from the floor plan.

    Args:
        background: original RGB image (PIL), aligned to mask
        wall_mask: processed mask (float or uint8) with walls/features

    Returns: dict with keys 'outlets', 'switches', 'lights'
    """
    img = background.convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    mask_u8 = (wall_mask * 255.0).astype(np.uint8) if wall_mask.dtype != np.uint8 else wall_mask
    edges = cv2.Canny(mask_u8, 50, 150)

    # Hough circles to detect common circular symbols
    circles_small = cv2.HoughCircles(
        arr, cv2.HOUGH_GRADIENT, dp=1.2, minDist=12,
        param1=120, param2=15, minRadius=4, maxRadius=10
    )
    circles_medium = cv2.HoughCircles(
        arr, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=120, param2=18, minRadius=10, maxRadius=22
    )

    candidates: List[Tuple[float, float, float]] = []
    if circles_small is not None:
        for c in circles_small[0, :]:
            candidates.append((float(c[0]), float(c[1]), float(c[2])))
    if circles_medium is not None:
        for c in circles_medium[0, :]:
            candidates.append((float(c[0]), float(c[1]), float(c[2])))

    candidates = _non_max_suppression_points(candidates, min_dist=10.0)

    outlets: List[Dict] = []
    lights: List[Dict] = []

    for x, y, r in candidates:
        d = _distance_to_edges(edges, x, y, max_search=10)
        if r <= 10 and d <= 6.0:
            outlets.append({"position": [float(x), float(y)], "radius": float(r)})
        elif r >= 10 and d > 6.0:
            lights.append({"position": [float(x), float(y)], "radius": float(r)})

    # Switches via OCR if available (look for isolated 'S', 'S3', 'S4')
    switches: List[Dict] = []
    FAST = bool(int(os.environ.get("FAST", "1")))
    if easyocr is not None and not FAST:
        try:
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            ocr = reader.readtext(np.asarray(background), paragraph=False)
            for bbox, text, conf in ocr:
                if conf < 0.4:
                    continue
                t = text.strip().upper()
                if t in {"S", "S3", "S4"}:
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x = float(sum(xs) / len(xs))
                    y = float(sum(ys) / len(ys))
                    # require near wall
                    if _distance_to_edges(edges, x, y, max_search=12) <= 8.0:
                        switches.append({"position": [x, y], "type": t})
        except Exception:
            pass

    return {"outlets": outlets, "switches": switches, "lights": lights}


def snap_to_nearest_wall(point: Tuple[float, float], walls: List[Dict], max_snap_dist: float = 12.0) -> Tuple[float, float]:
    """Project a point to the nearest segment if within max_snap_dist (pixels)."""
    px, py = float(point[0]), float(point[1])
    best = (px, py)
    best_d = float('inf')
    for wall in walls or []:
        x1, y1 = map(float, wall.get('start', [px, py]))
        x2, y2 = map(float, wall.get('end', [px, py]))
        vx, vy = x2 - x1, y2 - y1
        L2 = vx * vx + vy * vy
        if L2 == 0:
            continue
        t = max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / L2))
        qx, qy = x1 + t * vx, y1 + t * vy
        d = math.hypot(qx - px, qy - py)
        if d < best_d:
            best_d = d
            best = (qx, qy)
    if best_d <= max_snap_dist:
        return best
    return (px, py)
