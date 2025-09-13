from io import BytesIO
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import trimesh
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import cv2
import easyocr

# Import compliance engine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compliance.compliance_engine import OntarioBuildingCodeEngine

origins = [
    "*"
]

app = FastAPI(title="CAD Processing & Compliance API", version="0.2.0")

# Initialize compliance engine
compliance_engine = OntarioBuildingCodeEngine()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024


# Conservative preprocessing defaults (tuned for floor plans)
DEFAULT_INVERT = True
DEFAULT_BINARY = True
DEFAULT_SMOOTH_RADIUS = 0.25
DEFAULT_MEDIAN_SIZE = 3
DEFAULT_OPEN_SIZE = 2
DEFAULT_CLOSE_SIZE = 3
DEFAULT_DILATE_SIZE = 1
DEFAULT_KEEP_LARGEST = 150
DEFAULT_MIN_AREA_RATIO = 0.00003
DEFAULT_BASE_THICKNESS_MM = 1.0

# Initialize EasyOCR reader once (global to avoid repeated initialization)
_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader

def _remove_text_with_ocr(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    """Remove text regions detected by OCR from the mask.
    
    Args:
        image: Original grayscale PIL image 
        mask: Binary mask to remove text from
    Returns:
        mask with text regions removed
    """
    try:
        # Convert PIL image to numpy array for OCR
        img_array = np.array(image)
        
        # Get OCR reader
        reader = _get_ocr_reader()
        
        # Detect text (returns list of [bbox, text, confidence])
        results = reader.readtext(img_array, paragraph=False)
        
        # Create output mask
        out_mask = mask.copy()
        
        # Remove each detected text region with padding
        for detection in results:
            bbox = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            confidence = detection[2]
            
            # Only remove if confidence is high enough
            if confidence < 0.3:
                continue
                
            # Get bounding box coordinates
            xs = [int(pt[0]) for pt in bbox]
            ys = [int(pt[1]) for pt in bbox]
            x_min, x_max = max(0, min(xs) - 2), min(mask.shape[1], max(xs) + 2)
            y_min, y_max = max(0, min(ys) - 2), min(mask.shape[0], max(ys) + 2)
            
            # Clear text region from mask
            out_mask[y_min:y_max, x_min:x_max] = False
            
        return out_mask
    except Exception as e:
        # If OCR fails, return original mask
        print(f"OCR text removal failed: {e}")
        return mask

def _remove_small_components(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    """Fallback: Remove only very small noise components."""
    h, w = mask.shape
    total = h * w
    min_area = max(1, int(total * max(0.0, float(min_area_ratio))))
    
    visited = np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if row[x] and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                indices = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                                indices.append((ny, nx))
                # Only remove if truly tiny
                if len(indices) >= min_area:
                    for (yy, xx) in indices:
                        out[yy, xx] = True
    return out

def _filter_components_by_area(mask: np.ndarray, keep_largest: int, min_area_ratio: float) -> np.ndarray:
    """Keep only the largest connected components (8-connectivity).
    mask: boolean array (H, W)
    keep_largest: number of components to keep
    min_area_ratio: minimum area relative to image to keep (0..1)
    """
    h, w = mask.shape
    total = h * w
    min_area = max(1, int(total * max(0.0, float(min_area_ratio))))

    visited = np.zeros_like(mask, dtype=bool)
    comps = []  # list of (area, indices_list)

    # Neighbor offsets for 8-connectivity
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(h):
        row = mask[y]
        for x in range(w):
            if row[x] and not visited[y, x]:
                # BFS/DFS
                stack = [(y, x)]
                visited[y, x] = True
                indices = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                                indices.append((ny, nx))
                comps.append((len(indices), indices))

    if not comps:
        return np.zeros_like(mask, dtype=bool)

    # Sort by area desc and keep top K meeting min_area
    comps.sort(key=lambda t: t[0], reverse=True)
    kept = 0
    out = np.zeros_like(mask, dtype=bool)
    for area, indices in comps:
        if kept >= int(max(1, keep_largest)):
            break
        if area < min_area:
            continue
        for (yy, xx) in indices:
            out[yy, xx] = True
        kept += 1
    return out


def _load_png_as_grayscale(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image: {e}",
        )
    # Validate PNG specifically
    if (getattr(img, "format", None) or "").upper() != "PNG":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PNG images are supported.",
        )
    return img.convert("L")  # grayscale


def _downsample(img: Image.Image, max_dim: int) -> Image.Image:
    if max_dim is None or max_dim <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.LANCZOS)


def _ensure_odd(k: int) -> int:
    k = int(k)
    if k < 1:
        return 1
    return k if k % 2 == 1 else k + 1


def _otsu_threshold(arr_u8: np.ndarray) -> int:
    # arr_u8: uint8 array (H, W)
    hist = np.bincount(arr_u8.ravel(), minlength=256).astype(np.float64)
    total = arr_u8.size
    if total == 0:
        return 128
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    return idx


def _preprocess_to_heightmap(
    img_gray: Image.Image,
    invert: bool,
    binary: bool,
    smooth_radius: float,
    open_size: int,
    close_size: int,
    dilate_size: int,
    median_size: int,
    keep_largest: int,
    min_area_ratio: float,
) -> np.ndarray:
    """Return float32 height factor in [0,1] from PIL L image using PIL filters only."""
    im = img_gray

    # Optional invert first to make features of interest bright
    if invert:
        im = ImageOps.invert(im)

    # Contrast enhancement to spread dynamic range
    im = ImageOps.autocontrast(im)

    # Optional denoise and smooth
    if median_size and median_size >= 3:
        im = im.filter(ImageFilter.MedianFilter(_ensure_odd(median_size)))
    if smooth_radius and smooth_radius > 0.0:
        im = im.filter(ImageFilter.GaussianBlur(radius=float(smooth_radius)))

    if binary:
        arr_u8_a = np.asarray(im, dtype=np.uint8)
        thr_a = _otsu_threshold(arr_u8_a)
        mask_a = (arr_u8_a >= thr_a).astype(np.uint8)
        occ_a = mask_a.mean() if mask_a.size else 0.0

        # Try inverted variant and choose the one with lower but non-trivial occupancy
        arr_u8_b = 255 - arr_u8_a
        thr_b = _otsu_threshold(arr_u8_b)
        mask_b = (arr_u8_b >= thr_b).astype(np.uint8)
        occ_b = mask_b.mean() if mask_b.size else 0.0

        chosen = mask_a
        if 0.001 < occ_b < occ_a:
            chosen = mask_b
        # If occupancy still huge (> 40%), pick the smaller of the two
        if occ_a > 0.4 or occ_b > 0.4:
            chosen = mask_a if occ_a <= occ_b else mask_b

        bim = Image.fromarray((chosen * 255).astype(np.uint8), mode="L")
        orig_bim = bim.copy()

        # NOTE: Keep it simple to avoid grid artifacts: rely on global Otsu only.

        # Morphology (opening then closing). We'll adapt if this is too destructive.
        def _apply_morph(img_l: Image.Image, os: int, cs: int) -> Image.Image:
            os = _ensure_odd(os)
            cs = _ensure_odd(cs)
            out = img_l.filter(ImageFilter.MinFilter(os))
            out = out.filter(ImageFilter.MaxFilter(os))
            out = out.filter(ImageFilter.MaxFilter(cs))
            out = out.filter(ImageFilter.MinFilter(cs))
            return out

        # Slightly stronger despeckle to kill grid noise
        os_eff = max(3, open_size)
        bim = _apply_morph(bim, os_eff, close_size)
        # If after morph the area is extremely low, relax morphology
        pre_mask = (np.asarray(bim, dtype=np.uint8) > 0)
        area_frac = pre_mask.mean() if pre_mask.size else 0.0
        if area_frac < 0.01:
            bim = _apply_morph(orig_bim, 3, 3)
            pre_mask = (np.asarray(bim, dtype=np.uint8) > 0)

        # Optional thickness via dilation
        dsz = _ensure_odd(dilate_size)
        if dsz > 1:
            bim = bim.filter(ImageFilter.MaxFilter(dsz))
            pre_mask = (np.asarray(bim, dtype=np.uint8) > 0)

        # Remove fancy fusions; stick to morphology + component pruning only.

        # First remove tiny noise
        clean_mask = _remove_small_components(pre_mask, min_area_ratio=max(min_area_ratio, 0.00015))
        
        # Then use OCR to detect and remove text
        clean_mask = _remove_text_with_ocr(img_gray, clean_mask)
        
        # If removal was too aggressive, fall back to pre_mask
        if clean_mask.mean() < 0.005:
            clean_mask = pre_mask
        return clean_mask.astype(np.float32)
    else:
        # Grayscale heightmap
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr


def _heightmap_mesh(z_map: np.ndarray, scale_mm_per_px: float) -> trimesh.Trimesh:
    # z_map: (H, W) float32 in mm
    h, w = z_map.shape
    # Create grid of XY in mm
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    x = x_idx * scale_mm_per_px
    y = y_idx * scale_mm_per_px

    # Flatten vertices
    vertices = np.column_stack((x.ravel(), y.ravel(), z_map.ravel())).astype(np.float32)

    # Build faces: two triangles per grid cell
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    v00 = idx[:-1, :-1].ravel()
    v01 = idx[:-1, 1:].ravel()
    v10 = idx[1:, :-1].ravel()
    v11 = idx[1:, 1:].ravel()

    faces1 = np.column_stack((v00, v01, v10))
    faces2 = np.column_stack((v11, v10, v01))
    faces = np.vstack((faces1, faces2)).astype(np.int64)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def convert_png_to_stl(
    image_bytes: bytes,
    max_height_mm: float,
    scale_mm_per_px: float,
    downsample: int,
    invert: bool,
    binary: bool,
    smooth_radius: float,
    open_size: int,
    close_size: int,
    dilate_size: int,
    median_size: int,
    keep_largest: int,
    min_area_ratio: float,
) -> bytes:
    # 1) Load and validate
    img = _load_png_as_grayscale(image_bytes)

    # 2) Downsample
    downsample = int(max(16, min(4096, downsample)))
    img = _downsample(img, downsample)

    # 3) Preprocess (denoise, threshold/morph) to reduce spikes
    arr01 = _preprocess_to_heightmap(
        img_gray=img,
        invert=invert,
        binary=binary,
        smooth_radius=smooth_radius,
        open_size=open_size,
        close_size=close_size,
        dilate_size=dilate_size,
        median_size=median_size,
        keep_largest=keep_largest,
        min_area_ratio=min_area_ratio,
    )
    # No final blur: keep flat base and avoid grid ripples
    arr01 = np.clip(arr01, 0.0, 1.0)
    z_map = DEFAULT_BASE_THICKNESS_MM + arr01 * float(max_height_mm)

    # 4) Triangulate grid
    mesh = _heightmap_mesh(z_map, scale_mm_per_px=float(scale_mm_per_px))

    # 5) Export binary STL
    buffer = BytesIO()
    mesh.export(buffer, file_type="stl")
    return buffer.getvalue()


@app.post("/api/convert")
async def convert_endpoint(
    image: UploadFile = File(..., description="PNG image file"),
    max_height_mm: float = Form(10.0),
    scale_mm_per_px: float = Form(0.2),
    downsample: int = Form(1024),
    debug: str | None = Form(None),
):
    # Basic validation of numeric params
    if max_height_mm <= 0:
        raise HTTPException(status_code=400, detail="max_height_mm must be > 0")
    if scale_mm_per_px <= 0:
        raise HTTPException(status_code=400, detail="scale_mm_per_px must be > 0")

    try:
        contents = await image.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_BYTES // (1024*1024)} MB")
        # (Optional) Debug outputs
        if debug in {"mask", "contours"}:
            img = _load_png_as_grayscale(contents)
            img = _downsample(img, int(max(16, min(4096, downsample))))
            arr01 = _preprocess_to_heightmap(
                img_gray=img,
                invert=DEFAULT_INVERT,
                binary=DEFAULT_BINARY,
                smooth_radius=DEFAULT_SMOOTH_RADIUS,
                open_size=DEFAULT_OPEN_SIZE,
                close_size=DEFAULT_CLOSE_SIZE,
                dilate_size=DEFAULT_DILATE_SIZE,
                median_size=DEFAULT_MEDIAN_SIZE,
                keep_largest=DEFAULT_KEEP_LARGEST,
                min_area_ratio=DEFAULT_MIN_AREA_RATIO,
            )
            mask_bool = (arr01 >= 0.5)
            if debug == "mask":
                mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
                buf = BytesIO()
                mask_img.save(buf, format="PNG")
                headers = {"Content-Disposition": 'attachment; filename="mask.png"'}
                return Response(content=buf.getvalue(), media_type="image/png", headers=headers)
            else:
                # contours overlay
                overlay = Image.merge("RGB", (img, img, img))
                draw = ImageDraw.Draw(overlay)
                mask_u8 = (mask_bool.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                eps = 1.0
                for cnt in contours:
                    if cnt.shape[0] < 2:
                        continue
                    cnt = cv2.approxPolyDP(cnt, eps, True)
                    pts = cnt.squeeze(1)
                    pts_list = [tuple(map(float, p)) for p in pts]
                    # draw closed poly
                    if len(pts_list) >= 2:
                        draw.line(pts_list + [pts_list[0]], fill=(255, 0, 0), width=2)
                # limit output size to keep file small and viewable
                overlay.thumbnail((1600, 1600), Image.LANCZOS)
                buf = BytesIO()
                overlay.save(buf, format="PNG")
                headers = {"Content-Disposition": 'attachment; filename="contours.png"'}
                return Response(content=buf.getvalue(), media_type="image/png", headers=headers)

        # Use conservative internal defaults for preprocessing
        stl_bytes = convert_png_to_stl(
            image_bytes=contents,
            max_height_mm=max_height_mm,
            scale_mm_per_px=scale_mm_per_px,
            downsample=downsample,
            invert=DEFAULT_INVERT,
            binary=DEFAULT_BINARY,
            smooth_radius=DEFAULT_SMOOTH_RADIUS,
            open_size=DEFAULT_OPEN_SIZE,
            close_size=DEFAULT_CLOSE_SIZE,
            dilate_size=DEFAULT_DILATE_SIZE,
            median_size=DEFAULT_MEDIAN_SIZE,
            keep_largest=DEFAULT_KEEP_LARGEST,
            min_area_ratio=DEFAULT_MIN_AREA_RATIO,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")

    headers = {
        "Content-Disposition": 'attachment; filename="model.stl"'
    }
    return Response(content=stl_bytes, media_type="application/sla", headers=headers)


@app.get("/")
async def root():
    return {"status": "ok", "version": "0.2.0", "features": ["2d-to-3d", "cad-processing", "compliance"]}


def detect_room_boundaries_from_image(processed_mask: np.ndarray) -> List[Dict]:
    """Extract room boundaries from processed floor plan image"""
    # Find contours for room boundaries
    contours, _ = cv2.findContours(
        processed_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    rooms = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 1000:  # Skip very small areas
            continue
            
        # Calculate bounding box and room properties
        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else x + w // 2
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else y + h // 2
        
        rooms.append({
            "id": i + 1,
            "area_pixels": float(area),
            "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "centroid": [cx, cy],
            "contour": contour.tolist()
        })
    
    return rooms


def extract_walls_from_image(processed_mask: np.ndarray) -> List[Dict]:
    """Extract wall segments from floor plan image"""
    # Detect edges
    edges = cv2.Canny(processed_mask.astype(np.uint8), 50, 150)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    
    walls = []
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            walls.append({
                "id": i + 1,
                "type": "wall",
                "start": [int(x1), int(y1)],
                "end": [int(x2), int(y2)],
                "length": float(length)
            })
    
    return walls


def generate_intelligent_openings(walls: List[Dict], rooms: List[Dict]) -> List[Dict]:
    """
    Intelligently generate door and window placements based on room layout
    Since floor plan data quality is often poor, we use heuristics to place openings
    """
    openings = []
    
    for room in rooms:
        room_area = room.get('area_pixels', 0) * 0.01  # Convert to approximate sq ft
        bbox = room.get('bounding_box', {})
        
        # Generate doors - typically one per room, two for large rooms
        num_doors = 1 if room_area < 200 else 2
        for door_idx in range(num_doors):
            # Place door on longer wall
            if bbox.get('width', 0) > bbox.get('height', 0):
                door_x = bbox['x'] + bbox['width'] // (num_doors + 1) * (door_idx + 1)
                door_y = bbox['y'] if door_idx == 0 else bbox['y'] + bbox['height']
            else:
                door_x = bbox['x'] if door_idx == 0 else bbox['x'] + bbox['width']
                door_y = bbox['y'] + bbox['height'] // (num_doors + 1) * (door_idx + 1)
            
            openings.append({
                "type": "door",
                "width": 36,  # Standard door width in inches
                "height": 80,  # Standard door height
                "position": [door_x, door_y],
                "room_id": room['id'],
                "generated": True
            })
        
        # Generate windows based on room size and building code requirements
        # Minimum window area = 10% of floor area for natural light (OBC requirement)
        required_window_area = room_area * 0.1 * 144  # Convert to sq inches
        window_size = 48 * 36  # Standard window 48"x36"
        num_windows = max(1, int(required_window_area / window_size))
        
        # Place windows evenly on exterior walls
        for window_idx in range(num_windows):
            # Assume top and right walls are more likely exterior
            if window_idx % 2 == 0:
                window_x = bbox['x'] + bbox['width'] // (num_windows + 1) * (window_idx + 1)
                window_y = bbox['y']
            else:
                window_x = bbox['x'] + bbox['width']
                window_y = bbox['y'] + bbox['height'] // (num_windows + 1) * (window_idx + 1)
            
            openings.append({
                "type": "window",
                "width": 48,
                "height": 36,
                "sill_height": 30,  # Standard sill height from floor
                "position": [window_x, window_y],
                "room_id": room['id'],
                "generated": True,
                "egress_compliant": window_size >= 619  # Min egress area in sq inches
            })
    
    return openings


@app.post("/api/process_floorplan")
async def process_floorplan_endpoint(
    image: UploadFile = File(..., description="Floor plan image (PNG, JPG)"),
    generate_3d: bool = Form(False),
    check_compliance: bool = Form(True)
):
    """
    Process floor plan image to extract architectural elements
    Returns rooms, walls, and intelligently placed doors/windows
    """
    contents = await image.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_BYTES // (1024*1024)} MB")
    
    # Load and process image
    img = _load_png_as_grayscale(contents)
    img = _downsample(img, 1024)
    
    # Extract architectural elements
    processed_mask = _preprocess_to_heightmap(
        img_gray=img,
        invert=True,
        binary=True,
        smooth_radius=0.25,
        open_size=2,
        close_size=3,
        dilate_size=1,
        median_size=3,
        keep_largest=150,
        min_area_ratio=0.00003,
    )
    
    # Detect rooms and walls
    rooms = detect_room_boundaries_from_image(processed_mask)
    walls = extract_walls_from_image(processed_mask)
    
    # Generate intelligent door/window placements
    openings = generate_intelligent_openings(walls, rooms)
    
    floor_plan_data = {
        "type": "floor_plan_analysis",
        "rooms": rooms,
        "walls": walls,
        "openings": openings,
        "metadata": {
            "filename": image.filename,
            "processed_at": datetime.now().isoformat(),
            "openings_generated": True,
            "total_rooms": len(rooms),
            "total_walls": len(walls),
            "total_openings": len(openings)
        }
    }
    
    response = {"floor_plan_analysis": floor_plan_data}
    
    # Check compliance if requested
    if check_compliance:
        compliance_results = compliance_engine.check_compliance(floor_plan_data)
        routing_suggestions = compliance_engine.generate_compliance_suggestions(floor_plan_data)
        
        response["compliance_analysis"] = {
            "compliance_status": "pass" if not any(i.severity == "error" for i in compliance_results) else "fail",
            "issues": [
                {
                    "system": issue.system,
                    "code_reference": issue.code_reference,
                    "severity": issue.severity,
                    "message": issue.message,
                    "location": issue.location,
                    "suggested_fix": issue.suggested_fix
                }
                for issue in compliance_results
            ],
            "summary": {
                "total_issues": len(compliance_results),
                "errors": sum(1 for i in compliance_results if i.severity == "error"),
                "warnings": sum(1 for i in compliance_results if i.severity == "warning"),
                "info": sum(1 for i in compliance_results if i.severity == "info")
            },
            "routing_suggestions": routing_suggestions
        }
    
    # Generate 3D model if requested
    if generate_3d:
        # Convert to 3D heightmap
        z_map = 1.0 + processed_mask * 10.0  # Base + walls height
        mesh = _heightmap_mesh(z_map, scale_mm_per_px=0.2)
        buffer = BytesIO()
        mesh.export(buffer, file_type="stl")
        # Convert to base64 for JSON response
        import base64
        stl_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response["model_3d"] = {
            "format": "stl",
            "encoding": "base64",
            "data": stl_base64
        }
    
    return response


class ComplianceCheckRequest(BaseModel):
    floor_plan_data: Dict
    system_type: Optional[str] = None
    generate_suggestions: bool = True


@app.post("/api/check_compliance")
async def check_compliance_endpoint(request: ComplianceCheckRequest):
    """
    Check building code compliance for processed floor plan data
    """
    try:
        # Run compliance checks
        issues = compliance_engine.check_compliance(request.floor_plan_data, request.system_type)
        
        response = {
            "compliance_status": "pass" if not any(i.severity == "error" for i in issues) else "fail",
            "issues": [
                {
                    "system": issue.system,
                    "code_reference": issue.code_reference,
                    "severity": issue.severity,
                    "message": issue.message,
                    "location": issue.location,
                    "suggested_fix": issue.suggested_fix
                }
                for issue in issues
            ],
            "summary": {
                "total_issues": len(issues),
                "errors": sum(1 for i in issues if i.severity == "error"),
                "warnings": sum(1 for i in issues if i.severity == "warning"),
                "info": sum(1 for i in issues if i.severity == "info")
            }
        }
        
        # Generate routing suggestions if requested
        if request.generate_suggestions:
            response["routing_suggestions"] = compliance_engine.generate_compliance_suggestions(request.floor_plan_data)
        
        return response
        
    except Exception as e:
        raise HTTPException(500, f"Compliance check failed: {e}")

