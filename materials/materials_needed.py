#!/usr/bin/env python3
"""
Rule-of-thumb materials takeoff for a single-story wood-framed plan.

This script estimates common framing materials from high-level inputs
(exterior perimeter, interior wall length, wall height, stud spacing, etc.).

It outputs a JSON array in the exact format requested, e.g.
[
  {
    "name": "2x4x8 SPF lumber",
    "quantity": 200,
    "unit": "piece",
    "spec": "SPF stud grade",
    "vendor_preferences": ["Home Depot Canada", "RONA", "Lowe's Canada"]
  },
  {
    "name": "OSB sheathing 7/16 in 4x8",
    "quantity": 80,
    "unit": "sheet",
    "spec": "7/16 in OSB",
    "vendor_preferences": ["Home Depot Canada", "RONA", "Lowe's Canada"]
  }
]

Notes and assumptions (adjust via CLI flags as needed):
- Stud spacing defaults to 16 in OC.
- Stud count approximates both exterior and interior walls using total wall length.
- Double top plate + single bottom plate are implicitly included in the 2x4x8 count
  via a configurable extra percentage (plates_waste_pct), since many residential
  builds buy studs and plates together as generic 2x4x8 stock.
- Exterior sheathing uses perimeter * wall_height to compute area; divided by 32 sq ft
  per 4x8 sheet; plus waste.
- Door/window openings can be accounted for with a simple adjustment.

This estimator is intentionally simple and transparent so you can tweak
inputs quickly without complex CAD parsing. It is also designed to feed the
downstream pricing pipeline (material_search.py) if you want vendor pricing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import glob
from typing import List, Dict

# Optional OCR/image deps. If not available, we gracefully fallback to CLI inputs.
try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore
try:
    import pytesseract  # Requires system tesseract binary installed
except Exception:
    pytesseract = None  # type: ignore


DEFAULT_VENDORS = ["Home Depot Canada", "RONA", "Lowe's Canada"]


def ceil_int(x: float) -> int:
    return int(math.ceil(x))


def estimate_studs(
    total_wall_length_ft: float,
    stud_spacing_in: float = 16.0,
    extra_studs_for_openings: int = 0,
    waste_pct: float = 10.0,
    plates_waste_pct: float = 10.0,
) -> int:
    """Estimate 2x4x8 count for studs + a simple allowance for plates.

    total_wall_length_ft: total linear feet of walls (exterior + interior)
    stud_spacing_in: on-center spacing (inches)
    extra_studs_for_openings: add trimmers/kings etc. for doors/windows
    waste_pct: percent extra studs for waste, cuts, corners
    plates_waste_pct: additional percent to roughly cover double top and
                      single bottom plates using the same 2x4x8 stock.
    """
    if stud_spacing_in <= 0:
        stud_spacing_in = 16.0

    spacing_ft = stud_spacing_in / 12.0

    # Base studs from spacing (very rough; ignores segment endpoints)
    base_studs = total_wall_length_ft / spacing_ft

    # Add opening studs (kings + jacks). User supplies a lump sum.
    base_studs += float(extra_studs_for_openings)

    # Apply waste/allowances
    base_studs *= (1.0 + waste_pct / 100.0)

    # Rough-in plates allowance using the same 2x4x8 stock bucket.
    base_studs *= (1.0 + plates_waste_pct / 100.0)

    return ceil_int(base_studs)


def estimate_osb_sheets(
    exterior_perimeter_ft: float,
    wall_height_ft: float = 8.0,
    waste_pct: float = 10.0,
) -> int:
    """Estimate 7/16 in 4x8 OSB sheets for exterior walls.

    Assumes full height sheathing: area = perimeter * wall_height.
    Each sheet covers 32 sq ft.
    """
    if exterior_perimeter_ft <= 0 or wall_height_ft <= 0:
        return 0

    area_sqft = exterior_perimeter_ft * wall_height_ft
    sheets = area_sqft / 32.0
    sheets *= (1.0 + waste_pct / 100.0)
    return ceil_int(sheets)


def build_materials_payload(
    total_wall_length_ft: float,
    exterior_perimeter_ft: float,
    wall_height_ft: float,
    stud_spacing_in: float,
    opening_count: int,
    studs_waste_pct: float,
    plates_waste_pct: float,
    osb_waste_pct: float,
    vendors: List[str],
) -> List[Dict]:
    # Simple allowance: assume each opening consumes ~4 extra studs
    extra_studs = max(0, int(opening_count)) * 4

    studs_qty = estimate_studs(
        total_wall_length_ft=total_wall_length_ft,
        stud_spacing_in=stud_spacing_in,
        extra_studs_for_openings=extra_studs,
        waste_pct=studs_waste_pct,
        plates_waste_pct=plates_waste_pct,
    )

    osb_qty = estimate_osb_sheets(
        exterior_perimeter_ft=exterior_perimeter_ft,
        wall_height_ft=wall_height_ft,
        waste_pct=osb_waste_pct,
    )

    materials = [
        {
            "name": "2x4x8 SPF lumber",
            "quantity": studs_qty,
            "unit": "piece",
            "spec": "SPF stud grade",
            "vendor_preferences": vendors,
        },
        {
            "name": "OSB sheathing 7/16 in 4x8",
            "quantity": osb_qty,
            "unit": "sheet",
            "spec": "7/16 in OSB",
            "vendor_preferences": vendors,
        },
    ]

    return materials


def _ftin_to_ft(ft: int | float, inches: int | float) -> float:
    try:
        return float(ft) + float(inches) / 12.0
    except Exception:
        return float(ft)


def _parse_dimensions_from_text(text: str) -> list[tuple[float, float]]:
    """Parse dimension strings like 22'11" x 18'0" into (w_ft, h_ft).

    Returns a list of width/height in feet.
    """
    dims: list[tuple[float, float]] = []

    # Normalize some unicode
    t = text.replace("\u2032", "'").replace("\u2033", '"').replace("\u00D7", "x")

    # Pattern with feet and inches on both sides
    pat_full = re.compile(
        r"(\d{1,2})\s*'\s*(\d{1,2})\s*\"?\s*[xX]\s*(\d{1,2})\s*'\s*(\d{1,2})\s*\"?"
    )
    # Pattern with only feet (no inches) on both sides
    pat_feet_only = re.compile(r"(\d{1,2})\s*'\s*[xX]\s*(\d{1,2})\s*'")

    for m in pat_full.finditer(t):
        a_ft, a_in, b_ft, b_in = m.groups()
        try:
            w = _ftin_to_ft(int(a_ft), int(a_in))
            h = _ftin_to_ft(int(b_ft), int(b_in))
            # Filter obviously bogus small/large rooms
            if 3 <= w <= 50 and 3 <= h <= 50:
                dims.append((w, h))
        except Exception:
            continue

    for m in pat_feet_only.finditer(t):
        a_ft, b_ft = m.groups()
        try:
            w = float(a_ft)
            h = float(b_ft)
            if 3 <= w <= 50 and 3 <= h <= 50:
                dims.append((w, h))
        except Exception:
            continue

    return dims


def _ocr_text_from_image(image_path: str) -> str | None:
    if Image is None or pytesseract is None:
        return None
    try:
        img = Image.open(image_path)
        # Simple pre-processing: convert to grayscale for OCR
        img = img.convert("L")
        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return None


def _find_floorplan_image(floorplan_path: str | None, floorplan_dir: str) -> str | None:
    if floorplan_path:
        return floorplan_path if os.path.exists(floorplan_path) else None
    # Common names
    candidates = [
        os.path.join(floorplan_dir, "floorplan.png"),
        os.path.join(floorplan_dir, "floorplan.jpg"),
        os.path.join(floorplan_dir, "floorplan.jpeg"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fallback: first image in dir
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        matches = glob.glob(os.path.join(floorplan_dir, ext))
        if matches:
            return matches[0]
    return None


def estimate_from_floorplan(
    image_path: str,
    footprint_aspect_ratio: float = 1.6,
    interior_share_factor: float = 0.55,
) -> dict:
    """Estimate geometry metrics from a floorplan image via OCR heuristics.

    - Extract room dimension pairs using OCR (best-effort).
    - Approximate total footprint area from sum of room rectangles.
    - Derive an "equivalent rectangle" of aspect_ratio to estimate exterior perimeter.
    - Estimate interior wall length by scaling the sum of room perimeters
      to account for shared walls (interior_share_factor ~ 0.5-0.6 typical).

    Returns dict with keys: exterior_perimeter_ft, interior_wall_length_ft, openings
    """
    text = _ocr_text_from_image(image_path) or ""

    # Named dimension parsing: e.g., "LIVING ROOM 22'11\" x 18'0\""
    named_rooms: list[dict] = []
    try:
        # Normalize unicode and collapse spaces
        t = text.replace("\u2032", "'").replace("\u2033", '"').replace("\u00D7", "x")
        t = re.sub(r"\s+", " ", t)

        # Pattern with feet and inches both sides
        pat_named_full = re.compile(
            r"([A-Z][A-Z ]{2,}?)\s+(\d{1,2})\s*'\s*(\d{1,2})\s*\"?\s*[xX]\s*(\d{1,2})\s*'\s*(\d{1,2})\s*\"?"
        )
        for m in pat_named_full.finditer(t):
            name, a_ft, a_in, b_ft, b_in = m.groups()
            try:
                w = _ftin_to_ft(int(a_ft), int(a_in))
                h = _ftin_to_ft(int(b_ft), int(b_in))
                if 3 <= w <= 60 and 3 <= h <= 60:
                    named_rooms.append({"name": name.strip(), "w": w, "h": h})
            except Exception:
                continue

        # Fallback: feet-only
        if not named_rooms:
            pat_named_feet = re.compile(r"([A-Z][A-Z ]{2,}?)\s+(\d{1,2})\s*'\s*[xX]\s*(\d{1,2})\s*'")
            for m in pat_named_feet.finditer(t):
                name, a_ft, b_ft = m.groups()
                try:
                    w = float(a_ft)
                    h = float(b_ft)
                    if 3 <= w <= 60 and 3 <= h <= 60:
                        named_rooms.append({"name": name.strip(), "w": w, "h": h})
                except Exception:
                    continue
    except Exception:
        pass

    # Anonymous dimensions list as secondary signal
    dims = _parse_dimensions_from_text(text)

    if not dims and not named_rooms:
        # Heuristic fallback: use image size as proxy area and 8ft grid.
        try:
            if Image is not None:
                img = Image.open(image_path)
                w_px, h_px = img.size
                # Assume arbitrary scale: 25 px per foot (very rough)
                area_est = (w_px * h_px) / (25.0 * 25.0)
            else:
                area_est = 1500.0
        except Exception:
            area_est = 1500.0
        # Derive exterior perimeter from area and aspect ratio
        w = math.sqrt(area_est * footprint_aspect_ratio)
        h = area_est / w
        exterior_perimeter = 2 * (w + h)
        interior_wall_length = interior_share_factor * exterior_perimeter * 2.0
        openings = 16
        return {
            "exterior_perimeter_ft": float(exterior_perimeter),
            "interior_wall_length_ft": float(interior_wall_length),
            "openings": int(openings),
            "rooms_detected": 0,
            "named_rooms": [],
            "conditioned_ceiling_area_sqft": float(area_est * 0.7),
            "garage_ceiling_area_sqft": float(area_est * 0.3),
        }

    # Build room stats
    room_stats = []
    seen_any = False
    if named_rooms:
        for r in named_rooms:
            w = float(r["w"])
            h = float(r["h"])
            room_stats.append({
                "name": r["name"],
                "w": w,
                "h": h,
                "area": w * h,
                "perimeter": 2.0 * (w + h),
            })
            seen_any = True
    if not seen_any and dims:
        # Anonymous rooms; still capture areas for totals
        for (w, h) in dims:
            room_stats.append({
                "name": "ROOM",
                "w": w,
                "h": h,
                "area": w * h,
                "perimeter": 2.0 * (w + h),
            })

    # Compute total area and perimeters
    total_area = sum(r["area"] for r in room_stats)
    sum_room_perims = sum(r["perimeter"] for r in room_stats)

    # Equivalent rectangle
    w_equiv = math.sqrt(total_area * footprint_aspect_ratio)
    h_equiv = total_area / w_equiv if w_equiv > 0 else 0.0
    exterior_perimeter = 2.0 * (w_equiv + h_equiv)

    # Interior walls approximated from combined room perimeters with a share factor
    interior_wall_length = interior_share_factor * sum_room_perims

    # Determine basic counts for openings and areas
    names_lower = [r["name"].lower() for r in room_stats]
    conditioned_rooms = [r for r in room_stats if r["name"].lower() not in ("garage", "porch")]
    garage_rooms = [r for r in room_stats if r["name"].lower() == "garage"]
    porch_rooms = [r for r in room_stats if r["name"].lower() == "porch"]

    conditioned_area = sum(r["area"] for r in conditioned_rooms)
    garage_area = sum(r["area"] for r in garage_rooms) or 0.0

    # Heuristic openings
    likely_door_rooms = {"bedroom", "bath", "closet", "laundry room", "laundry", "office", "pantry", "room"}
    interior_door_count = sum(1 for r in room_stats if r["name"].lower() in likely_door_rooms)
    interior_door_count = max(8, min(interior_door_count, 20))
    exterior_door_count = 2
    window_count = max(8, min(int(round(exterior_perimeter / 16.0)), 20))
    openings = interior_door_count + exterior_door_count + window_count

    return {
        "exterior_perimeter_ft": float(exterior_perimeter),
        "interior_wall_length_ft": float(interior_wall_length),
        "openings": int(openings),
        "rooms_detected": len(room_stats),
        "named_rooms": room_stats,
        "conditioned_ceiling_area_sqft": float(conditioned_area),
        "garage_ceiling_area_sqft": float(garage_area),
        "interior_doors": int(interior_door_count),
        "exterior_doors": int(exterior_door_count),
        "windows": int(window_count),
    }


def build_detailed_materials_payload(
    inferred: dict,
    wall_height_ft: float,
    stud_spacing_in: float,
    studs_waste_pct: float,
    plates_waste_pct: float,
    osb_waste_pct: float,
    drywall_waste_pct: float,
    insulation_waste_pct: float,
    finish_waste_pct: float,
    vendors: List[str],
) -> List[Dict]:
    """Produce a richer materials list using floorplan-derived metrics."""
    ext_perim = float(inferred.get("exterior_perimeter_ft", 0.0))
    interior_len = float(inferred.get("interior_wall_length_ft", 0.0))
    total_wall_len = ext_perim + interior_len

    interior_doors = int(inferred.get("interior_doors", 10))
    exterior_doors = int(inferred.get("exterior_doors", 2))
    windows = int(inferred.get("windows", max(8, int(round(ext_perim / 16.0)))))

    # Studs (9ft walls use ~104-5/8" studs). Do NOT include plate allowance here.
    extra_studs = (interior_doors + exterior_doors + windows) * 4
    studs_qty = estimate_studs(
        total_wall_length_ft=total_wall_len,
        stud_spacing_in=stud_spacing_in,
        extra_studs_for_openings=extra_studs,
        waste_pct=studs_waste_pct,
        plates_waste_pct=0.0,
    )

    # Plates: double top + single bottom across all walls
    plates_linear = 3.0 * total_wall_len
    plates_pieces_10 = ceil_int(plates_linear * (1.0 + plates_waste_pct / 100.0) / 10.0)

    # Exterior sheathing area (no opening subtraction to stay conservative)
    osb_area = ext_perim * wall_height_ft
    osb_sheets = ceil_int(osb_area * (1.0 + osb_waste_pct / 100.0) / 32.0)

    # Drywall wall area (interior sides)
    interior_wall_area = interior_len * wall_height_ft * 2.0
    exterior_interior_area = ext_perim * wall_height_ft
    interior_door_area = 16.7 * interior_doors  # 30x80
    exterior_door_area = 21.0 * exterior_doors  # 36x84 approx
    window_area = 12.0 * windows                # 36x48 approx
    interior_wall_area_adj = max(0.0, interior_wall_area - interior_door_area)
    exterior_interior_area_adj = max(0.0, exterior_interior_area - exterior_door_area - window_area)
    drywall_walls_area = interior_wall_area_adj + exterior_interior_area_adj
    drywall_walls_sheets = ceil_int(drywall_walls_area * (1.0 + drywall_waste_pct / 100.0) / 32.0)

    # Ceilings
    conditioned_area = float(inferred.get("conditioned_ceiling_area_sqft", 0.0))
    garage_area = float(inferred.get("garage_ceiling_area_sqft", 0.0))
    drywall_living_ceiling = ceil_int(conditioned_area * (1.0 + drywall_waste_pct / 100.0) / 32.0)
    drywall_garage_ceiling = ceil_int(garage_area * (1.0 + drywall_waste_pct / 100.0) / 32.0)

    # Insulation + wraps for exterior walls (subtract openings)
    ext_wall_area = ext_perim * wall_height_ft
    insul_area = max(0.0, ext_wall_area - window_area - exterior_door_area)
    insul_area = insul_area * (1.0 + insulation_waste_pct / 100.0)
    wrap_area = ext_wall_area * (1.0 + insulation_waste_pct / 100.0)

    # Baseboard: sum perimeters of conditioned rooms
    named_rooms = inferred.get("named_rooms") or []
    baseboard_lf = 0.0
    for r in named_rooms:
        nm = str(r.get("name", "")).lower()
        if nm in ("garage", "porch"):
            continue
        try:
            baseboard_lf += float(r.get("perimeter", 0.0))
        except Exception:
            pass
    baseboard_lf *= (1.0 + finish_waste_pct / 100.0)
    baseboard_lf_qty = ceil_int(baseboard_lf)

    materials: List[Dict] = [
        {
            "name": "2x4x104-5/8 in SPF studs",
            "quantity": studs_qty,
            "unit": "piece",
            "spec": "SPF stud grade; 9 ft walls; 16 in OC; includes openings/waste",
            "vendor_preferences": vendors,
        },
        {
            "name": "2x4x10 SPF lumber (plates)",
            "quantity": plates_pieces_10,
            "unit": "piece",
            "spec": "Double top + single bottom plates across all walls",
            "vendor_preferences": vendors,
        },
        {
            "name": "OSB sheathing 7/16 in 4x8",
            "quantity": osb_sheets,
            "unit": "sheet",
            "spec": "Exterior walls ~9 ft height",
            "vendor_preferences": vendors,
        },
        {
            "name": "Drywall 1/2 in 4x8 (walls)",
            "quantity": drywall_walls_sheets,
            "unit": "sheet",
            "spec": "Interior walls both sides + interior face of exterior walls",
            "vendor_preferences": vendors,
        },
        {
            "name": "Drywall 1/2 in 4x8 (ceilings, living areas)",
            "quantity": drywall_living_ceiling,
            "unit": "sheet",
            "spec": "Conditioned area ceilings",
            "vendor_preferences": vendors,
        },
        {
            "name": "Drywall 5/8 in Type X 4x8 (garage ceiling)",
            "quantity": drywall_garage_ceiling,
            "unit": "sheet",
            "spec": "Fire-rated garage ceiling",
            "vendor_preferences": vendors,
        },
        {
            "name": "Fiberglass batts R-14 (3.5 in) for 2x4 walls",
            "quantity": ceil_int(insul_area),
            "unit": "sqft",
            "spec": "Exterior wall cavities",
            "vendor_preferences": vendors,
        },
        {
            "name": "House wrap (weather barrier)",
            "quantity": ceil_int(wrap_area),
            "unit": "sqft",
            "spec": "Tyvek-style wrap for exterior walls",
            "vendor_preferences": vendors,
        },
        {
            "name": "Polyethylene vapor barrier 6 mil",
            "quantity": ceil_int(wrap_area),
            "unit": "sqft",
            "spec": "Interior side of exterior walls (where required)",
            "vendor_preferences": vendors,
        },
        {
            "name": "MDF baseboard 3-1/2 in",
            "quantity": baseboard_lf_qty,
            "unit": "linear_ft",
            "spec": "Perimeter of finished rooms",
            "vendor_preferences": vendors,
        },
        {
            "name": "Interior doors (prehung hollow-core 30x80)",
            "quantity": interior_doors,
            "unit": "piece",
            "spec": "Assorted swings",
            "vendor_preferences": vendors,
        },
        {
            "name": "Exterior doors (insulated steel 36x80)",
            "quantity": exterior_doors,
            "unit": "piece",
            "spec": "Front/rear or house-to-garage",
            "vendor_preferences": vendors,
        },
        {
            "name": "Vinyl windows (avg 36x48)",
            "quantity": windows,
            "unit": "piece",
            "spec": "Double-pane, new-construction flanged",
            "vendor_preferences": vendors,
        }
    ]

    return materials


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Estimate basic framing materials from high-level inputs or from a floorplan image. "
            "Outputs a JSON array suitable for the pricing step."
        )
    )

    # Floorplan inputs
    p.add_argument("--from-floorplan", action="store_true", help="Attempt to infer geometry from a floorplan image via OCR.")
    p.add_argument("--floorplan", help="Path to floorplan image. If not provided, searches floorplans/ for floorplan.png.")
    p.add_argument("--floorplan-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "floorplans"), help="Directory to search for a floorplan image.")
    p.add_argument("--aspect-ratio", type=float, default=1.6, help="Assumed footprint aspect ratio when inferring from area (default: 1.6).")
    p.add_argument("--interior-share-factor", type=float, default=0.55, help="Portion of summed room perimeters treated as interior walls (default: 0.55).")

    # Geometry inputs (manual overrides). If omitted, will try floorplan inference.
    p.add_argument(
        "--exterior-perimeter",
        type=float,
        help="Exterior perimeter in feet (sum of all exterior wall lengths).",
    )
    p.add_argument(
        "--interior-wall-length",
        type=float,
        help="Total interior wall length in feet (sum of all interior partitions).",
    )
    p.add_argument(
        "--wall-height",
        type=float,
        default=8.0,
        help="Framed wall height in feet for exterior walls (default: 8).",
    )

    # Framing assumptions
    p.add_argument("--stud-spacing", type=float, default=16.0, help="Stud spacing in inches OC (default: 16).")
    p.add_argument("--openings", type=int, help="Approximate count of total door+window openings. If omitted, will be inferred.")
    p.add_argument("--studs-waste-pct", type=float, default=10.0, help="Stud waste/extra percentage (default: 10).")
    p.add_argument(
        "--plates-waste-pct",
        type=float,
        default=10.0,
        help="Additional percentage to roughly cover double top + bottom plates (default: 10).",
    )
    p.add_argument("--osb-waste-pct", type=float, default=10.0, help="OSB waste/extra percentage (default: 10).")
    p.add_argument("--drywall-waste-pct", type=float, default=10.0, help="Drywall waste percentage (default: 10).")
    p.add_argument("--insulation-waste-pct", type=float, default=10.0, help="Insulation/wrap waste percentage (default: 10).")
    p.add_argument("--finish-waste-pct", type=float, default=10.0, help="Finish trim waste percentage (default: 10).")

    # Output
    p.add_argument(
        "--output",
        "-o",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "materials_test.json"),
        help="Destination JSON file (default: backend/materials_test.json).",
    )
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Print JSON to stdout only (do not write file).",
    )

    # Vendors
    p.add_argument(
        "--vendors",
        nargs="*",
        default=DEFAULT_VENDORS,
        help="Preferred vendors to include in the payload.",
    )

    # Output detail
    p.add_argument("--detailed", action="store_true", help="Produce a detailed materials list (drywall, insulation, doors, windows, etc.).")

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve geometry, either manual or inferred from floorplan
    exterior_perimeter = args.exterior_perimeter
    interior_wall_length = args.interior_wall_length
    openings = args.openings
    inferred = None

    need_infer = args.from_floorplan or (exterior_perimeter is None or interior_wall_length is None)
    if need_infer:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        floorplan_dir = args.floorplan_dir or os.path.join(script_dir, "floorplans")
        fp_path = _find_floorplan_image(args.floorplan, floorplan_dir)

        if fp_path is None:
            raise SystemExit(
                "No floorplan image found. Provide --floorplan PATH or place floorplan.png in floorplans/."
            )

        inferred = estimate_from_floorplan(
            fp_path,
            footprint_aspect_ratio=float(args.aspect_ratio),
            interior_share_factor=float(args.interior_share_factor),
        )

        if exterior_perimeter is None:
            exterior_perimeter = inferred.get("exterior_perimeter_ft")
        if interior_wall_length is None:
            interior_wall_length = inferred.get("interior_wall_length_ft")
        if openings is None:
            openings = inferred.get("openings")

    # Final validation/fallbacks
    if exterior_perimeter is None or interior_wall_length is None:
        raise SystemExit(
            "Missing geometry. Provide --exterior-perimeter and --interior-wall-length or enable --from-floorplan."
        )
    if openings is None:
        openings = 12

    total_wall_length_ft = float(exterior_perimeter) + float(interior_wall_length)

    use_detailed = bool(args.detailed or args.from_floorplan)
    if use_detailed:
        if inferred is None:
            # Build a minimal inference record from manual geometry
            windows = max(8, min(int(round(float(exterior_perimeter) / 16.0)), 20))
            inferred = {
                "exterior_perimeter_ft": float(exterior_perimeter),
                "interior_wall_length_ft": float(interior_wall_length),
                "conditioned_ceiling_area_sqft": 0.0,
                "garage_ceiling_area_sqft": 0.0,
                "interior_doors": max(8, min(int(round(float(interior_wall_length) / 25.0)), 20)),
                "exterior_doors": 2,
                "windows": windows,
                "named_rooms": [],
            }
        materials = build_detailed_materials_payload(
            inferred=inferred,
            wall_height_ft=float(args.wall_height),
            stud_spacing_in=float(args.stud_spacing),
            studs_waste_pct=float(args.studs_waste_pct),
            plates_waste_pct=float(args.plates_waste_pct),
            osb_waste_pct=float(args.osb_waste_pct),
            drywall_waste_pct=float(args.drywall_waste_pct),
            insulation_waste_pct=float(args.insulation_waste_pct),
            finish_waste_pct=float(args.finish_waste_pct),
            vendors=list(args.vendors),
        )
    else:
        materials = build_materials_payload(
            total_wall_length_ft=total_wall_length_ft,
            exterior_perimeter_ft=float(exterior_perimeter),
            wall_height_ft=float(args.wall_height),
            stud_spacing_in=float(args.stud_spacing),
            opening_count=int(openings),
            studs_waste_pct=float(args.studs_waste_pct),
            plates_waste_pct=float(args.plates_waste_pct),
            osb_waste_pct=float(args.osb_waste_pct),
            vendors=list(args.vendors),
        )

    # Ensure directory exists
    if not args.print_only:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    payload = json.dumps(materials, indent=2)

    if args.print_only:
        print(payload)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"Wrote materials JSON to {args.output}")


if __name__ == "__main__":
    main()

