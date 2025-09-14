"""
System Visualizer for Floor Plans
Generates visual overlays for electrical, HVAC, and plumbing systems based on compliance results
"""

from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import math
import heapq
import os

class SystemColors:
    """Color scheme for different systems"""
    electrical: Tuple[int, int, int] = (255, 200, 0)  # Yellow
    hvac_supply: Tuple[int, int, int] = (0, 150, 255)  # Blue
    hvac_return: Tuple[int, int, int] = (0, 100, 200)  # Dark Blue
    plumbing_supply: Tuple[int, int, int] = (255, 0, 0)  # Red
    plumbing_drain: Tuple[int, int, int] = (139, 69, 19)  # Brown
    walls: Tuple[int, int, int] = (128, 128, 128)  # Gray
    doors: Tuple[int, int, int] = (139, 90, 43)  # Brown
    windows: Tuple[int, int, int] = (135, 206, 235)  # Sky Blue

class SystemVisualizer:
    def __init__(self):
        self.colors = SystemColors()
        # Sizing (px) configurable via env for quick tuning
        self.sym_outlet_r = int(os.environ.get("SYM_OUTLET_R", "12"))
        self.sym_light_r = int(os.environ.get("SYM_LIGHT_R", "18"))
        self.sym_switch_w = int(os.environ.get("SYM_SWITCH_W", "16"))
        self.sym_switch_h = int(os.environ.get("SYM_SWITCH_H", "20"))
        self.wall_offset = float(os.environ.get("SYM_WALL_OFFSET", "12"))  # push symbols inward
        # Verbose logging toggle
        self.log_layout = os.environ.get("LOG_LAYOUT", "0") == "1"
        # Use hardcoded positions for specific floorplan
        self.use_hardcoded = os.environ.get("USE_HARDCODED_LIGHTS", "0") == "1"
        
    def generate_electrical_layout(self, floor_plan_data: Dict, compliance_issues: List, background: Image.Image) -> Image.Image:
        """Generate electrical system overlay based on compliance requirements over the original floor plan."""
        rooms = floor_plan_data.get('rooms', [])

        # Create overlay on top of original
        img = self._create_overlay_image(background, floor_plan_data)
        draw = ImageDraw.Draw(img)

        # Routing graph from walls/contours
        nodes, adj = self._build_routing_graph(floor_plan_data)

        # Process electrical compliance issues to determine outlet placement
        for issue in compliance_issues:
            if issue.system == 'electrical' and 'outlets' in issue.message:
                room_id = self._extract_room_id(issue.message)
                num_outlets = self._extract_outlet_count(issue.message)
                if room_id and room_id <= len(rooms):
                    room = rooms[room_id - 1]
                    self._draw_electrical_outlets(draw, room, num_outlets)

        # Draw electrical panel location
        panel_location = self._determine_panel_location(floor_plan_data)
        self._draw_electrical_panel(draw, panel_location)

        # Draw detected devices (outlets, switches, lights) snapped to walls
        devices = floor_plan_data.get('electrical', {}) or {}
        walls = floor_plan_data.get('walls', []) or []

        # Place lights in ALL rooms, including the garage
        room_lights: Dict[int, List[Tuple[float, float]]] = {}
        if self.log_layout:
            print(f"[layout] Placing lights for {len(rooms)} rooms")
        
        for room in rooms:
            rid = room.get('id')
            lights = self._place_lights_simple_centered(room)
            room_lights[rid] = lights
            if self.log_layout:
                bbox = room.get('bounding_box', {})
                print(f"[layout] Room {rid}: placed {len(lights)} lights, bbox={bbox}")

        # Randomly assign a circuit id per room (visual only)
        import random
        room_circuit: Dict[int, int] = {}
        for room in rooms:
            room_circuit[room.get('id')] = random.randint(1, 4)

        # Draw lights first using refined/new positions
        for room in rooms:
            rid = room.get('id')
            for (x, y) in room_lights.get(rid, []):
                r = self.sym_light_r
                draw.ellipse([x - r, y - r, x + r, y + r], outline=self.colors.electrical, width=2)
                draw.line([(x - r, y), (x + r, y)], fill=self.colors.electrical, width=1)
                draw.line([(x, y - r), (x, y + r)], fill=self.colors.electrical, width=1)

        # Then draw detected outlets and switches (snapped and nudged)
        for out in devices.get('outlets', []) or []:
            x, y = out.get('position', [0, 0])
            sx, sy = self._snap_to_nearest_wall((x, y), walls)
            # offset inward a bit from wall
            sx, sy = self._nudge_from_wall((sx, sy), walls, self.wall_offset)
            r = self.sym_outlet_r
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], outline=self.colors.electrical, width=2)
            draw.line([(sx - r/2, sy - 2), (sx + r/2, sy + 2)], fill=self.colors.electrical, width=2)
            draw.line([(sx - r/2, sy + 2), (sx + r/2, sy - 2)], fill=self.colors.electrical, width=2)
            # Leader to nearest wall
            wx, wy = self._nearest_point_on_walls((sx, sy), walls)
            draw.line([(sx, sy), (wx, wy)], fill=self.colors.electrical, width=1)
        for sw in devices.get('switches', []) or []:
            x, y = sw.get('position', [0, 0])
            sx, sy = self._snap_to_nearest_wall((x, y), walls)
            sx, sy = self._nudge_from_wall((sx, sy), walls, self.wall_offset)
            w, h = self.sym_switch_w, self.sym_switch_h
            draw.rectangle([sx - w/2, sy - h/2, sx + w/2, sy + h/2], outline=self.colors.electrical, width=2)
            draw.line([(sx - w/3, sy), (sx + w/3, sy)], fill=self.colors.electrical, width=2)
            wx, wy = self._nearest_point_on_walls((sx, sy), walls)
            draw.line([(sx, sy), (wx, wy)], fill=self.colors.electrical, width=1)

        # Draw dashed circuit wiring connecting rooms (not just within rooms)
        # This shows they're on the same circuit
        all_lights = []
        room_centers = []
        for room in rooms:
            rid = room.get('id')
            pts = room_lights.get(rid, [])
            if pts:
                all_lights.extend(pts)
                # Get the center light of each room for inter-room connections
                if len(pts) == 1:
                    room_centers.append(pts[0])
                elif len(pts) == 2:
                    # Average of the two
                    room_centers.append(((pts[0][0] + pts[1][0])/2, (pts[0][1] + pts[1][1])/2))
                elif len(pts) == 4:
                    # Center of the 4
                    cx = sum(p[0] for p in pts) / 4
                    cy = sum(p[1] for p in pts) / 4
                    room_centers.append((cx, cy))
                
                # Draw connections within the room first
                if len(pts) >= 2:
                    self._draw_circuit_wiring(draw, room, pts)
        
        # REMOVED - No inter-room connections per user request

    # Do not draw legend on the overlay anymore (separate endpoint provides legend)

        # Note: intentionally do not draw panel-to-room straight lines (they clutter and look like spokes).
        # Future: route feeders along edges to the first outlet in each circuit.

        return img

    def debug_cerebras_layout(self, floor_plan_data: Dict, compliance_issues: List) -> Dict:
        """Return a JSON summary of Cerebras usage and results for debugging."""
        rooms = floor_plan_data.get('rooms', []) or []
        devices = floor_plan_data.get('electrical', {}) or {}
        use_cerebras = os.environ.get('USE_CEREBRAS_LAYOUT', '0') == '1'
        api_key = os.environ.get('CEREBRAS_API_KEY')
        try:
            from .ai_layout import CEREBRAS_API_URL, CEREBRAS_MODEL, refine_layout_with_cerebras
        except Exception:
            CEREBRAS_API_URL = os.environ.get('CEREBRAS_API_URL', 'unknown')
            CEREBRAS_MODEL = os.environ.get('CEREBRAS_MODEL', 'unknown')
            refine_layout_with_cerebras = None

        result = {
            'use_cerebras': use_cerebras,
            'api_key_present': bool(api_key),
            'model': CEREBRAS_MODEL,
            'url': CEREBRAS_API_URL,
            'rooms_count': len(rooms),
            'devices_counts': {
                'outlets': len(devices.get('outlets') or []),
                'lights': len(devices.get('lights') or []),
                'switches': len(devices.get('switches') or []),
            },
            'refined': None,
            'fallback': False,
        }

        if use_cerebras and api_key and refine_layout_with_cerebras:
            try:
                refined = refine_layout_with_cerebras(api_key, rooms, devices, compliance_issues)
            except Exception as e:
                refined = None
                result['error'] = f"refine error: {e}"
            if refined and isinstance(refined.get('rooms'), dict):
                # Summarize counts
                summary = {}
                for rid_str, data in refined['rooms'].items():
                    lights = data.get('lights', []) or []
                    wires = data.get('wires', []) or []
                    summary[rid_str] = {'lights': len(lights), 'wires': len(wires)}
                result['refined'] = {'rooms': summary}
            else:
                result['fallback'] = True
        else:
            result['fallback'] = True
        return result

    def generate_electrical_legend_image(self, width: int = 340, height: int = 190) -> Image.Image:
        """Return a standalone legend image to avoid covering the floor plan."""
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        self._add_electrical_legend(draw, (width, height))
        return img
    
    def generate_hvac_layout(self, floor_plan_data: Dict, compliance_issues: List, routing_suggestions: Dict, background: Image.Image) -> Image.Image:
        """Generate HVAC system overlay with ductwork over the original floor plan; route along walls/contours."""
        rooms = floor_plan_data.get('rooms', [])

        img = self._create_overlay_image(background, floor_plan_data)
        draw = ImageDraw.Draw(img)

        nodes, adj = self._build_routing_graph(floor_plan_data)

        # Get main trunk location from routing suggestions
        hvac_routing = routing_suggestions.get('hvac_routing', {})
        trunk_location = hvac_routing.get('main_trunk_location', [500, 400])

        # Draw main trunk line (follow contour horizontally)
        self._draw_hvac_trunk(draw, trunk_location, rooms)

        # Draw supply ducts to each room using routed polylines
        for room in rooms:
            cfm = self._get_room_cfm_requirement(room, compliance_issues)
            duct_size = self._calculate_duct_size(cfm)
            path = self._route_path(nodes, adj, trunk_location, room['centroid'])
            self._draw_polyline(draw, path, self.colors.hvac_supply, width=duct_size)
            self._draw_hvac_register(draw, room, 'supply')

        # Return air system: route back to near trunk
        self._draw_return_air_system(draw, rooms, trunk_location, nodes, adj)

        # Equipment location
        equipment_loc = self._determine_equipment_location(floor_plan_data)
        self._draw_hvac_equipment(draw, equipment_loc)

        return img
    
    def generate_plumbing_layout(self, floor_plan_data: Dict, compliance_issues: List, background: Image.Image) -> Image.Image:
        """Generate plumbing system overlay with supply and drainage over original floor plan."""
        rooms = floor_plan_data.get('rooms', [])

        img = self._create_overlay_image(background, floor_plan_data)
        draw = ImageDraw.Draw(img)

        nodes, adj = self._build_routing_graph(floor_plan_data)

        # Identify wet rooms
        wet_rooms = self._identify_wet_rooms(rooms, compliance_issues)

        # Determine main stack location
        stack_location = self._determine_stack_location(wet_rooms)
        self._draw_plumbing_stack(draw, stack_location)

        # Draw supply and fixtures using routed polylines
        for room in wet_rooms:
            fixtures = self._get_room_fixtures(room, compliance_issues)
            for fixture in fixtures:
                path_hot = self._route_path(nodes, adj, stack_location, fixture['location'])
                path_cold = path_hot  # route same path visually with offset handled in drawing
                self._draw_polyline(draw, path_hot, (255, 0, 0), width=2)      # hot
                self._draw_polyline(draw, path_cold, (0, 0, 255), width=2)     # cold
                self._draw_plumbing_fixture(draw, fixture)

        # Drainage and vents (route back to stack)
        for room in wet_rooms:
            path_drain = self._route_path(nodes, adj, room['centroid'], stack_location)
            self._draw_polyline(draw, path_drain, self.colors.plumbing_drain, width=3)
            self._draw_plumbing_vent(draw, room['centroid'])

        # Legend
        self._add_plumbing_legend(draw, img.size)

        return img
    
    def generate_combined_systems(self, floor_plan_data: Dict, compliance_issues: List, 
                                 routing_suggestions: Dict, background: Image.Image) -> Image.Image:
        """Generate combined view of all systems on top of original floor plan."""
        base = self._create_overlay_image(background, floor_plan_data, show_labels=True)

        electrical_img = self.generate_electrical_layout(floor_plan_data, compliance_issues, background)
        hvac_img = self.generate_hvac_layout(floor_plan_data, compliance_issues, routing_suggestions, background)
        plumbing_img = self.generate_plumbing_layout(floor_plan_data, compliance_issues, background)

        # Blend overlays
        img = Image.blend(base, electrical_img, 0.35)
        img = Image.blend(img, hvac_img, 0.35)
        img = Image.blend(img, plumbing_img, 0.35)

        self._add_combined_legend(ImageDraw.Draw(img), img.size)
        return img
    
    def _create_overlay_image(self, background: Image.Image, floor_plan_data: Dict, show_labels: bool = False) -> Image.Image:
        """Return an RGB image cloned from the original floor plan to draw overlays."""
        img = background.convert('RGB').copy()
        draw = ImageDraw.Draw(img)

        if show_labels:
            for i, room in enumerate(floor_plan_data.get('rooms', [])):
                centroid = room.get('centroid', [0, 0])
                draw.text(tuple(centroid), f"Room {i+1}", fill='black')
        return img
    
    def _draw_electrical_outlets(self, draw: ImageDraw, room: Dict, num_outlets: int):
        """Draw duplex receptacles spaced along the actual room contour (not just the bounding box)."""
        contour = room.get('contour')
        bbox = room.get('bounding_box', {})
        if not contour and not bbox:
            return
        # Parse contour points as list[(x,y)]
        pts: List[Tuple[float, float]] = []
        if contour:
            for p in contour:
                try:
                    if isinstance(p, (list, tuple)) and len(p) == 1 and isinstance(p[0], (list, tuple)):
                        x, y = float(p[0][0]), float(p[0][1])
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        x, y = float(p[0]), float(p[1])
                    else:
                        continue
                    pts.append((x, y))
                except Exception:
                    continue
        # Fallback to rectangle if contour missing
        if len(pts) < 3 and bbox:
            x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('width', 0), bbox.get('height', 0)
            pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        if len(pts) < 3 or num_outlets <= 0:
            return

        # Compute perimeter and cumulative lengths
        perim = 0.0
        seg_lengths: List[float] = []
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            L = math.hypot(x2 - x1, y2 - y1)
            seg_lengths.append(L)
            perim += L

        if perim <= 0:
            return
        spacing = perim / float(max(1, num_outlets))

        # Place outlets at equal arc-length intervals along contour
        t = 0.0
        for _ in range(num_outlets):
            target = t % perim
            acc = 0.0
            for i in range(len(pts)):
                L = seg_lengths[i]
                if acc + L >= target:
                    # interpolate on segment i
                    x1, y1 = pts[i]
                    x2, y2 = pts[(i + 1) % len(pts)]
                    u = 0.0 if L == 0 else (target - acc) / L
                    x = x1 + u * (x2 - x1)
                    y = y1 + u * (y2 - y1)
                    # Draw duplex receptacle symbol: two parallel slashes in a circle
                    r = 6
                    draw.ellipse([x - r, y - r, x + r, y + r], outline=self.colors.electrical, width=2)
                    draw.line([(x - 3, y - 2), (x + 3, y + 2)], fill=self.colors.electrical, width=1)
                    draw.line([(x - 3, y + 2), (x + 3, y - 2)], fill=self.colors.electrical, width=1)
                    break
                acc += L
            t += spacing
    
    def _draw_polyline(self, draw: ImageDraw, points: List[List[float]], color: Tuple[int, int, int], width: int = 2):
        """Draw a polyline through the given points."""
        if not points or len(points) < 2:
            return
        coords = []
        for p in points:
            try:
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
                coords.append((x, y))
            except Exception:
                continue
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=width)
    
    def _draw_plumbing_fixture(self, draw: ImageDraw, fixture: Dict):
        """Draw plumbing fixture symbol"""
        loc = fixture['location']
        fixture_type = fixture['type']
        
        if fixture_type == 'water_closet':
            # Draw toilet symbol
            draw.ellipse([loc[0]-10, loc[1]-15, loc[0]+10, loc[1]+5], 
                        outline=self.colors.plumbing_supply, width=2)
        elif fixture_type == 'lavatory':
            # Draw sink symbol
            draw.rectangle([loc[0]-12, loc[1]-8, loc[0]+12, loc[1]+8], 
                          outline=self.colors.plumbing_supply, width=2)
        elif fixture_type == 'bathtub':
            # Draw tub symbol
            draw.rectangle([loc[0]-20, loc[1]-10, loc[0]+20, loc[1]+10], 
                          outline=self.colors.plumbing_supply, width=2)
            draw.ellipse([loc[0]-18, loc[1]-8, loc[0]+18, loc[1]+8], 
                        fill='white', outline=self.colors.plumbing_supply, width=1)
    
    def _extract_room_id(self, message: str) -> int:
        """Extract room ID from compliance message"""
        import re
        match = re.search(r'Room (\d+)', message)
        return int(match.group(1)) if match else None
    
    def _extract_outlet_count(self, message: str) -> int:
        """Extract required outlet count from compliance message"""
        import re
        match = re.search(r'(\d+) outlets', message)
        return int(match.group(1)) if match else 2
    
    def _calculate_duct_size(self, cfm: int) -> int:
        """Calculate duct width in pixels based on CFM"""
        if cfm < 100:
            return 4
        elif cfm < 200:
            return 6
        elif cfm < 400:
            return 8
        else:
            return 10
    
    def _get_room_cfm_requirement(self, room: Dict, compliance_issues: List) -> int:
        """Extract CFM requirement from compliance issues"""
        for issue in compliance_issues:
            if issue.system == 'hvac' and str(room['id']) in issue.message:
                import re
                match = re.search(r'(\d+) CFM', issue.message)
                if match:
                    return int(match.group(1))
        return 100  # Default CFM
    
    def _identify_wet_rooms(self, rooms: List[Dict], compliance_issues: List) -> List[Dict]:
        """Identify rooms that need plumbing"""
        wet_rooms = []
        for issue in compliance_issues:
            if issue.system == 'plumbing':
                # Find room associated with this issue
                location = issue.location
                if location:
                    for room in rooms:
                        centroid = room.get('centroid', [0, 0])
                        if abs(centroid[0] - location[0]) < 10 and abs(centroid[1] - location[1]) < 10:
                            if room not in wet_rooms:
                                wet_rooms.append(room)
                                break
        return wet_rooms
    
    def _get_room_fixtures(self, room: Dict, compliance_issues: List) -> List[Dict]:
        """Determine fixtures needed in a room based on compliance"""
        fixtures = []
        bbox = room.get('bounding_box', {})
        
        # Default fixture placement
        fixtures.append({
            'type': 'water_closet',
            'location': [bbox['x'] + bbox['width'] * 0.25, bbox['y'] + bbox['height'] * 0.25]
        })
        fixtures.append({
            'type': 'lavatory',
            'location': [bbox['x'] + bbox['width'] * 0.75, bbox['y'] + bbox['height'] * 0.25]
        })
        
        # Add bathtub for larger bathrooms
        if bbox['width'] * bbox['height'] > 7000:
            fixtures.append({
                'type': 'bathtub',
                'location': [bbox['x'] + bbox['width'] * 0.5, bbox['y'] + bbox['height'] * 0.75]
            })
        
        return fixtures
    
    def _determine_panel_location(self, floor_plan_data: Dict) -> List[float]:
        """Determine electrical panel location"""
        # Heuristic: choose leftmost-lowest exterior bound from walls; fallback to [50,50]
        walls = floor_plan_data.get('walls', []) or []
        if not walls:
            return [50, 50]
        min_sum = float('inf')
        best = [50.0, 50.0]
        for w in walls:
            for pt in (w.get('start'), w.get('end')):
                if not pt or len(pt) < 2:
                    continue
                s = float(pt[0]) + float(pt[1])
                if s < min_sum:
                    min_sum = s
                    best = [float(pt[0]), float(pt[1])]
        # offset slightly inward
        best[0] += 10
        best[1] += 10
        return best
    
    def _determine_equipment_location(self, floor_plan_data: Dict) -> List[float]:
        """Determine HVAC equipment location"""
        # Place in mechanical room or basement (bottom-right corner)
        max_x = max_y = 0
        for room in floor_plan_data.get('rooms', []):
            bbox = room.get('bounding_box', {})
            max_x = max(max_x, bbox.get('x', 0) + bbox.get('width', 0))
            max_y = max(max_y, bbox.get('y', 0) + bbox.get('height', 0))
        return [max_x - 100, max_y - 100]
    
    def _determine_stack_location(self, wet_rooms: List[Dict]) -> List[float]:
        """Determine plumbing stack location"""
        if not wet_rooms:
            return [400, 400]
        
        # Place at centroid of wet rooms
        avg_x = np.mean([r.get('centroid', [0, 0])[0] for r in wet_rooms])
        avg_y = np.mean([r.get('centroid', [0, 0])[1] for r in wet_rooms])
        return [avg_x, avg_y]
    
    def _draw_electrical_panel(self, draw: ImageDraw, location: List[float]):
        """Draw electrical panel symbol"""
        x, y = location
        draw.rectangle([x-20, y-30, x+20, y+30], outline=self.colors.electrical, width=3)
        draw.text((x-15, y-10), "PANEL", fill=self.colors.electrical)
    
    def _draw_hvac_equipment(self, draw: ImageDraw, location: List[float]):
        """Draw HVAC equipment symbol"""
        x, y = location
        draw.rectangle([x-30, y-30, x+30, y+30], outline=self.colors.hvac_supply, width=3)
        draw.text((x-20, y-10), "HVAC", fill=self.colors.hvac_supply)
    
    def _draw_plumbing_stack(self, draw: ImageDraw, location: List[float]):
        """Draw main plumbing stack"""
        x, y = location
        draw.ellipse([x-15, y-15, x+15, y+15], outline=self.colors.plumbing_drain, width=3)
        draw.text((x-20, y+20), "STACK", fill=self.colors.plumbing_drain)
    
    def _draw_circuit_routing(self, draw: ImageDraw, panel_loc: List[float], room: Dict):
        """Draw electrical circuit routing from panel to room"""
        draw.line([tuple(panel_loc), tuple(room['centroid'])], 
                 fill=self.colors.electrical, width=1)
    
    def _draw_hvac_trunk(self, draw: ImageDraw, trunk_location: List[float], rooms: List[Dict]):
        """Draw main HVAC trunk line"""
        if not rooms:
            return
        # Find bounds of all rooms
        min_x = float('inf')
        max_x = 0.0
        for room in rooms:
            bbox = room.get('bounding_box', {})
            min_x = min(min_x, float(bbox.get('x', 0)))
            max_x = max(max_x, float(bbox.get('x', 0) + bbox.get('width', 0)))
        if not (min_x < max_x):
            return
        # Draw horizontal trunk through center
        y = int(round(float(trunk_location[1])))
        draw.line([(int(round(min_x)), y), (int(round(max_x)), y)], 
                 fill=self.colors.hvac_supply, width=12)
    
    def _draw_hvac_register(self, draw: ImageDraw, room: Dict, register_type: str):
        """Draw HVAC supply/return register"""
        centroid = room['centroid']
        color = self.colors.hvac_supply if register_type == 'supply' else self.colors.hvac_return
        
        # Draw register symbol (square with cross)
        size = 8
        draw.rectangle([centroid[0]-size, centroid[1]-size, 
                       centroid[0]+size, centroid[1]+size], 
                      outline=color, width=2)
        draw.line([centroid[0]-size, centroid[1], centroid[0]+size, centroid[1]], 
                 fill=color, width=1)
        draw.line([centroid[0], centroid[1]-size, centroid[0], centroid[1]+size], 
                 fill=color, width=1)
    
    def _draw_return_air_system(self, draw: ImageDraw, rooms: List[Dict], trunk_location: List[float], nodes, adj):
        """Draw routed return air ducts back to a central return near the trunk."""
        central_return = trunk_location.copy()
        central_return[1] += 50
        for room in rooms[:2]:
            path = self._route_path(nodes, adj, room['centroid'], central_return)
            self._draw_polyline(draw, path, self.colors.hvac_return, width=8)
            self._draw_hvac_register(draw, room, 'return')
    
    def _draw_supply_line(self, draw: ImageDraw, start: List[float], end: List[float], water_type: str):
        """Draw water supply line"""
        color = (255, 0, 0) if water_type == 'hot' else (0, 0, 255)  # Red for hot, blue for cold
        
        # Offset hot and cold lines slightly
        offset = 3 if water_type == 'hot' else -3
        start_offset = [start[0] + offset, start[1]]
        end_offset = [end[0] + offset, end[1]]
        
        draw.line([tuple(start_offset), tuple(end_offset)], fill=color, width=2)

    # --- Geometry helpers ---
    def _snap_to_nearest_wall(self, point: Tuple[float, float], walls: List[Dict]) -> Tuple[int, int]:
        px, py = float(point[0]), float(point[1])
        best = (px, py)
        best_d = float('inf')
        for wall in walls or []:
            try:
                x1, y1 = map(float, wall.get('start', [px, py]))
                x2, y2 = map(float, wall.get('end', [px, py]))
            except Exception:
                continue
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
        return (int(round(best[0])), int(round(best[1])))

    def _nudge_from_wall(self, pt: Tuple[float, float], walls: List[Dict], offset: float) -> Tuple[int, int]:
        """Move point slightly inward perpendicular to the closest wall segment to avoid drawing on the wall line."""
        px, py = float(pt[0]), float(pt[1])
        best = None
        best_d = float('inf')
        best_norm = (0.0, 0.0)
        for wall in walls or []:
            try:
                x1, y1 = map(float, wall.get('start', [px, py]))
                x2, y2 = map(float, wall.get('end', [px, py]))
            except Exception:
                continue
            vx, vy = x2 - x1, y2 - y1
            L2 = vx * vx + vy * vy
            if L2 == 0:
                continue
            t = max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / L2))
            qx, qy = x1 + t * vx, y1 + t * vy
            d = math.hypot(qx - px, qy - py)
            if d < best_d:
                best_d = d
                # normal vector (rotate v by 90 degrees)
                nx, ny = -vy, vx
                nlen = math.hypot(nx, ny) or 1.0
                best_norm = (nx / nlen, ny / nlen)
                best = (qx, qy)
        if best is None:
            return (int(round(px)), int(round(py)))
        nx, ny = best_norm
        return (int(round(px + nx * offset)), int(round(py + ny * offset)))

    # REMOVED - No longer needed since hardcoded mode just uses centroids
    
    def _get_hardcoded_lights_for_room(self, room_id: int) -> List[Tuple[float, float]]:
        """
        HARDCODED LIGHT POSITIONS FOR EACH ROOM
        
        COORDINATE SYSTEM:
        - Origin (0,0) is at TOP-LEFT corner of the image
        - X increases going RIGHT
        - Y increases going DOWN
        - Full image is approximately 1024x995 pixels
        
        ROOM LAYOUT (approximate positions):
        Top row: Living Room (left), Kitchen (center), Master Bedroom (right)
        Middle: Bedrooms, Bathrooms, Laundry, Closets, Halls
        Bottom: Office, Porch, Garage (bottom-right is the large garage)
        """
        hardcoded_lights = {
            # EDIT THESE VALUES TO PLACE LIGHTS WHERE YOU WANT
            1: [(850, 750)],                    # Room 1
            2: [(750, 850)],                    # Room 2
            3: [(750, 650)],                    # Room 3
            4: [(650, 750)],                    # Room 4
            5: [(450, 190)],                    # Room 5
            6: [(850, 200)],                    # Room 6
            7: [(150, 470)],                    # Room 7
            8: [(350, 400)],                    # Room 8
            9: [(600, 500)],                    # Room 9
            10: [(750, 470)],                   # Room 10
            11: [(850, 470)],                   # Room 11
            12: [(550, 420)],                    # Room 12
            13: [(700, 110)],                   # Room 13
            14: [(350, 570)],                   # Room 14
            15: [(550, 520)],                   # Room 15
            16: [(120, 750), (180, 750)],      # Room 16 - Bedroom (bottom-left) - 2 lights
            17: [(350, 700), (450, 700)],      # Room 17 - Office (bottom-center) - 2 lights
            # Garage - add 4 lights in square pattern
            18: [(700, 650), 
                 (850, 650),       # Room 18 (likely garage)
                 (700, 800), 
                 (850, 800)],
            19: [(950, 350)],                   # Room 19
            20: [(100, 800)],                   # Room 20
        }
        
        return hardcoded_lights.get(room_id, [])
    
    def _place_lights_simple_centered(self, room: Dict) -> List[Tuple[float, float]]:
        """Place lights properly centered in rooms."""
        # HARDCODED MODE
        if self.use_hardcoded:
            room_id = room.get('id')
            if room_id:
                lights = self._get_hardcoded_lights_for_room(room_id)
                if lights:
                    if self.log_layout:
                        print(f"[HARDCODED] Room {room_id}: {len(lights)} lights at {lights}")
                    return lights
            
            # Fallback if no hardcoded positions
            centroid = room.get('centroid', [400, 400])
            return [(float(centroid[0]), float(centroid[1]))]
        
        # DYNAMIC MODE (original code)
        bbox = room.get('bounding_box', {})
        if not bbox:
            centroid = room.get('centroid', [0, 0])
            return [(float(centroid[0]), float(centroid[1]))]
        
        x = float(bbox.get('x', 0))
        y = float(bbox.get('y', 0))
        w = float(bbox.get('width', 0))
        h = float(bbox.get('height', 0))
        
        if w <= 0 or h <= 0:
            centroid = room.get('centroid', [x, y])
            return [(float(centroid[0]), float(centroid[1]))]
        
        # Calculate room center
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        
        # Calculate room area
        area = w * h * 2
        
        # Determine number of lights based on room size
        # Garage is typically > 400x200 pixels
        if area > 8000 or (w > 350 and h > 200):  # Large room like garage
            # 4 lights in square pattern, properly centered
            # Use smaller offset (15-20%) to keep lights well within the room
            offset_x = w * 0.15
            offset_y = h * 0.15
            
            lights = [
                (center_x - offset_x, center_y - offset_y),  # top-left
                (center_x + offset_x, center_y - offset_y),  # top-right
                (center_x - offset_x, center_y + offset_y),  # bottom-left
                (center_x + offset_x, center_y + offset_y)   # bottom-right
            ]
        elif area > 3000:  # Medium-large room
            # 2 lights
            if w > h * 1.3:  # Wide room
                lights = [
                    (center_x - w * 0.2, center_y),
                    (center_x + w * 0.2, center_y)
                ]
            elif h > w * 1.3:  # Tall room
                lights = [
                    (center_x, center_y - h * 0.2),
                    (center_x, center_y + h * 0.2)
                ]
            else:  # Square-ish room
                # Diagonal placement
                offset = min(w, h) * 0.2
                lights = [
                    (center_x - offset, center_y - offset),
                    (center_x + offset, center_y + offset)
                ]
        else:
            # Single light at center for small rooms
            lights = [(center_x, center_y)]
        
        return lights

    def _draw_circuit_wiring(self, draw: ImageDraw, room: Dict, points: List[Tuple[float, float]]):
        """Draw dashed lines interconnecting lights to form electrical groups."""
        if len(points) < 2:
            if self.log_layout:
                print(f"[layout] Skipping circuit wiring for room {room.get('id')} - only {len(points)} lights")
            return
        
        if self.log_layout:
            print(f"[layout] Drawing circuit for room {room.get('id')} with points: {points}")
        
        # For 2 lights: connect them directly
        if len(points) == 2:
            self._draw_dashed_line(draw, points[0], points[1], self.colors.electrical, width=3)
            return
        
        # For 4 lights: connect them in a square pattern
        if len(points) == 4:
            # Sort points by position to get proper corners
            sorted_points = sorted(points, key=lambda p: (p[1], p[0]))  # Sort by y, then x
            top_points = sorted(sorted_points[:2], key=lambda p: p[0])
            bottom_points = sorted(sorted_points[2:], key=lambda p: p[0])
            
            if self.log_layout:
                print(f"[layout] Drawing square pattern: top={top_points}, bottom={bottom_points}")
            
            # Draw square connections with thicker lines
            self._draw_dashed_line(draw, top_points[0], top_points[1], self.colors.electrical, width=3)  # top
            self._draw_dashed_line(draw, bottom_points[0], bottom_points[1], self.colors.electrical, width=3)  # bottom
            self._draw_dashed_line(draw, top_points[0], bottom_points[0], self.colors.electrical, width=3)  # left
            self._draw_dashed_line(draw, top_points[1], bottom_points[1], self.colors.electrical, width=3)  # right
            return
        
        # For other configurations: connect in a chain
        for i in range(len(points) - 1):
            self._draw_dashed_line(draw, points[i], points[i + 1], self.colors.electrical, width=3)
    
    def _draw_dashed_line(self, draw: ImageDraw, p1: Tuple[float, float], p2: Tuple[float, float], 
                         color: Tuple[int, int, int], width: int = 2):
        """Draw a dashed line between two points."""
        x1, y1 = p1
        x2, y2 = p2
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len <= 0:
            if self.log_layout:
                print(f"[layout] Skipping zero-length line from {p1} to {p2}")
            return
        
        # Make dashes more visible
        dash = 12
        gap = 6
        ux, uy = (x2 - x1) / seg_len, (y2 - y1) / seg_len
        d = 0.0
        
        dash_count = 0
        while d < seg_len:
            a = d
            b = min(seg_len, d + dash)
            ax, ay = x1 + ux * a, y1 + uy * a
            bx, by = x1 + ux * b, y1 + uy * b
            draw.line([(int(ax), int(ay)), (int(bx), int(by))], fill=color, width=width)
            dash_count += 1
            d += dash + gap
        
        if self.log_layout and dash_count > 0:
            print(f"[layout] Drew {dash_count} dashes from {p1} to {p2}")

    def _add_electrical_legend(self, draw: ImageDraw, size: Tuple[int, int]):
        x = 20
        y = 20
        # Background card for readability (no schedule)
        card_w = 300
        # 3 rows + wiring sample
        card_h = 160
        draw.rectangle([x - 10, y - 10, x - 10 + card_w, y - 10 + card_h], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        draw.text((x, y), "ELECTRICAL LEGEND:", fill='black')
        y += 18
        r = self.sym_outlet_r
        draw.ellipse([x, y, x + 2*r, y + 2*r], outline=self.colors.electrical, width=2)
        draw.line([(x + r/2, y + r - 2), (x + 3*r/2, y + r + 2)], fill=self.colors.electrical, width=2)
        draw.line([(x + r/2, y + r + 2), (x + 3*r/2, y + r - 2)], fill=self.colors.electrical, width=2)
        draw.text((x + 2*r + 6, y + r - 6), "Duplex Receptacle", fill='black')
        y += 2*r + 8
        rr = self.sym_light_r
        draw.ellipse([x, y, x + 2*rr, y + 2*rr], outline=self.colors.electrical, width=2)
        draw.line([(x, y + rr), (x + 2*rr, y + rr)], fill=self.colors.electrical, width=1)
        draw.line([(x + rr, y), (x + rr, y + 2*rr)], fill=self.colors.electrical, width=1)
        draw.text((x + 2*rr + 6, y + rr - 6), "Ceiling Light", fill='black')
        y += 2*rr + 8
        w, h = self.sym_switch_w, self.sym_switch_h
        draw.rectangle([x, y, x + w, y + h], outline=self.colors.electrical, width=2)
        draw.line([(x + 3, y + h/2), (x + w - 3, y + h/2)], fill=self.colors.electrical, width=2)
        draw.text((x + w + 6, y + h/2 - 6), "Switch", fill='black')
        # Wiring sample (dashed)
        y += h + 10
        dash_demo = [(x, y), (x + 60, y + 10), (x + 120, y - 5)]
        self._draw_circuit_wiring(draw, {'id': -1}, dash_demo)

    def _draw_panel_schedule(self, draw: ImageDraw, circuits: List[Dict]):
        x = 20
        y = 160
        rows = min(14, len(circuits))
        card_w = 320
        card_h = 26 + 16 + 14 * rows
        draw.rectangle([x - 10, y - 10, x - 10 + card_w, y - 10 + card_h], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        draw.text((x, y), "PANEL SCHEDULE (AUTO):", fill='black')
        y += 18
        for c in circuits[:rows]:
            txt = f"Ckt {c['circuit']:>2}: {c['desc']} ({c.get('breaker', 15)}A)"
            draw.text((x, y), txt, fill='black')
            y += 14

    def generate_electrical_schedule(self, floor_plan_data: Dict) -> Dict:
        """Generate a more realistic panel schedule.
        Strategy:
        - Assign devices to nearest room.
        - Bundle general receptacle loads up to ~8 outlets per 15A circuit (heuristic).
        - Combine several small rooms on one circuit; keep total outlet count balanced.
        - Create 1-2 lighting circuits based on total lights.
        - If USE_CEREBRAS=1 and CEREBRAS_API_KEY is set, call the Cerebras API to refine grouping; otherwise fallback.
        Returns dict with 'circuits' and 'room_to_circuit' (for general receptacles), plus counts.
        """
        rooms: List[Dict] = floor_plan_data.get('rooms', []) or []
        devices = floor_plan_data.get('electrical', {}) or {}
        outlets = devices.get('outlets', []) or []
        lights = devices.get('lights', []) or []
        switches = devices.get('switches', []) or []

        # 1) Assign devices to rooms
        room_assign = self._assign_devices_to_rooms(rooms, outlets, lights, switches)

        # 2) Build general receptacles circuits by bundling up to target per circuit
        target_outlets_per_circuit = int(os.environ.get('TARGET_OUTLETS_PER_CKT', '8'))
        circuits: List[Dict] = []
        room_to_circuit: Dict[int, int] = {}
        ckt_num = 1
        bucket_count = 0
        bucket_rooms: List[int] = []

        for r in rooms:
            rid = r.get('id')
            ol = len(room_assign.get(rid, {}).get('outlets', []))
            # If adding this room exceeds the target and the bucket isn't empty, close current circuit
            if bucket_rooms and (bucket_count + ol) > target_outlets_per_circuit:
                desc = f"Rooms {', '.join(map(str, bucket_rooms))} Gen Receptacles"
                circuits.append({'circuit': ckt_num, 'desc': desc, 'breaker': 15})
                for rr in bucket_rooms:
                    room_to_circuit[rr] = ckt_num
                ckt_num += 1
                bucket_rooms = []
                bucket_count = 0
            # Add room to current bucket
            bucket_rooms.append(rid)
            bucket_count += ol
        # Flush trailing bucket
        if bucket_rooms:
            desc = f"Rooms {', '.join(map(str, bucket_rooms))} Gen Receptacles"
            circuits.append({'circuit': ckt_num, 'desc': desc, 'breaker': 15})
            for rr in bucket_rooms:
                room_to_circuit[rr] = ckt_num
            ckt_num += 1

        # 3) Lighting circuits: bundle lights across rooms, ~12 fixtures per circuit
        total_lights = sum(len(room_assign.get(r.get('id'), {}).get('lights', [])) for r in rooms)
        if total_lights > 0:
            per_ckt = int(os.environ.get('TARGET_LIGHTS_PER_CKT', '12'))
            num_light_ckts = max(1, (total_lights + per_ckt - 1) // per_ckt)
            for i in range(num_light_ckts):
                circuits.append({'circuit': ckt_num, 'desc': 'Lighting', 'breaker': 15})
                ckt_num += 1

        # 4) Optionally refine using Cerebras API
        try:
            use_cerebras = os.environ.get('USE_CEREBRAS', '0') == '1'
            api_key = os.environ.get('CEREBRAS_API_KEY')
            if use_cerebras and api_key:
                from .ai_schedule import refine_schedule_with_cerebras
                refined = refine_schedule_with_cerebras(api_key, rooms, room_assign, circuits, room_to_circuit)
                if refined:
                    circuits = refined.get('circuits', circuits)
                    room_to_circuit = refined.get('room_to_circuit', room_to_circuit)
        except Exception as e:
            # Fail quietly to fallback schedule
            print(f"[schedule] Cerebras refinement skipped: {e}")

        return {
            'circuits': circuits,
            'room_to_circuit': room_to_circuit,
            'counts': {
                'outlets': len(outlets),
                'lights': len(lights),
                'switches': len(switches)
            }
        }

    def _assign_devices_to_rooms(self, rooms: List[Dict], outlets: List[Dict], lights: List[Dict], switches: List[Dict]) -> Dict[int, Dict[str, List[Dict]]]:
        """Assign detected devices to the nearest room centroid (fast heuristic).
        Returns mapping room_id -> {'outlets': [...], 'lights': [...], 'switches': [...]}.
        """
        result: Dict[int, Dict[str, List[Dict]]] = {r.get('id'): {'outlets': [], 'lights': [], 'switches': []} for r in rooms}
        if not rooms:
            return result
        # Pre-extract centroids
        centroids = [(r.get('id'), r.get('centroid', [0, 0])) for r in rooms]

        def nearest_room_id(pt: Tuple[float, float]) -> int:
            px, py = float(pt[0]), float(pt[1])
            best = None
            best_d = float('inf')
            for rid, c in centroids:
                dx = float(c[0]) - px
                dy = float(c[1]) - py
                d2 = dx * dx + dy * dy
                if d2 < best_d:
                    best_d = d2
                    best = rid
            return best

        for dev in outlets or []:
            rid = nearest_room_id(dev.get('position', [0, 0]))
            if rid in result:
                result[rid]['outlets'].append(dev)
        for dev in lights or []:
            rid = nearest_room_id(dev.get('position', [0, 0]))
            if rid in result:
                result[rid]['lights'].append(dev)
        for dev in switches or []:
            rid = nearest_room_id(dev.get('position', [0, 0]))
            if rid in result:
                result[rid]['switches'].append(dev)
        return result

    def _nearest_point_on_walls(self, point: Tuple[float, float], walls: List[Dict]) -> Tuple[int, int]:
        px, py = float(point[0]), float(point[1])
        best = (px, py)
        best_d = float('inf')
        for wall in walls or []:
            try:
                x1, y1 = map(float, wall.get('start', [px, py]))
                x2, y2 = map(float, wall.get('end', [px, py]))
            except Exception:
                continue
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
        return (int(round(best[0])), int(round(best[1])))

    def _nearest_room_circuit(self, room_to_circuit: Dict, rooms: List[Dict], pt: Tuple[float, float]):
        px, py = float(pt[0]), float(pt[1])
        best_room_id = None
        best_d = float('inf')
        for r in rooms or []:
            c = r.get('centroid', [0, 0])
            try:
                dx = float(c[0]) - px
                dy = float(c[1]) - py
                d2 = dx * dx + dy * dy
            except Exception:
                continue
            if d2 < best_d:
                best_d = d2
                best_room_id = r.get('id')
        return room_to_circuit.get(best_room_id) if best_room_id is not None else None
    
    def _draw_drainage_line(self, draw: ImageDraw, start: List[float], end: List[float]):
        """Draw drainage line"""
        draw.line([tuple(start), tuple(end)], fill=self.colors.plumbing_drain, width=3)
    
    def _draw_plumbing_vent(self, draw: ImageDraw, location: List[float]):
        """Draw plumbing vent"""
        x, y = location
        # Draw vent pipe going up
        draw.line([(x, y), (x, y-20)], fill=self.colors.plumbing_drain, width=2)
        draw.ellipse([x-3, y-23, x+3, y-17], outline=self.colors.plumbing_drain, width=1)
    
    def _add_plumbing_legend(self, draw: ImageDraw, img_size: Tuple[int, int]):
        """Add legend for plumbing symbols"""
        legend_x = img_size[0] - 150
        legend_y = 20
        
        draw.text((legend_x, legend_y), "PLUMBING LEGEND:", fill='black')
        
        # Hot water
        draw.line([(legend_x, legend_y + 20), (legend_x + 30, legend_y + 20)], 
                 fill=(255, 0, 0), width=2)
        draw.text((legend_x + 35, legend_y + 15), "Hot Water", fill='black')
        
        # Cold water
        draw.line([(legend_x, legend_y + 35), (legend_x + 30, legend_y + 35)], 
                 fill=(0, 0, 255), width=2)
        draw.text((legend_x + 35, legend_y + 30), "Cold Water", fill='black')
        
        # Drain
        draw.line([(legend_x, legend_y + 50), (legend_x + 30, legend_y + 50)], 
                 fill=self.colors.plumbing_drain, width=3)
        draw.text((legend_x + 35, legend_y + 45), "Drain", fill='black')
    
    def _add_combined_legend(self, draw: ImageDraw, img_size: Tuple[int, int]):
        """Add legend for all systems"""
        legend_x = img_size[0] - 180
        legend_y = 20
        
        draw.text((legend_x, legend_y), "SYSTEMS LEGEND:", fill='black')
        
        # Electrical
        draw.ellipse([legend_x, legend_y + 20, legend_x + 10, legend_y + 30], 
                    outline=self.colors.electrical, width=2)
        draw.text((legend_x + 15, legend_y + 18), "Electrical Outlet", fill='black')
        
        # HVAC Supply
        draw.line([(legend_x, legend_y + 40), (legend_x + 30, legend_y + 40)], 
                 fill=self.colors.hvac_supply, width=6)
        draw.text((legend_x + 35, legend_y + 35), "HVAC Supply", fill='black')
        
        # HVAC Return
        draw.line([(legend_x, legend_y + 55), (legend_x + 30, legend_y + 55)], 
                 fill=self.colors.hvac_return, width=6)
        draw.text((legend_x + 35, legend_y + 50), "HVAC Return", fill='black')
        
        # Plumbing
        draw.line([(legend_x, legend_y + 70), (legend_x + 30, legend_y + 70)], 
                 fill=self.colors.plumbing_supply, width=2)
        draw.text((legend_x + 35, legend_y + 65), "Water Supply", fill='black')

    # ---- Routing helpers ----
    def _build_routing_graph(self, floor_plan_data: Dict):
        """Build a simple undirected graph from wall segments and room contours.
        Returns (nodes_list, adjacency_dict) where adjacency maps node_index -> list[(nei_index, weight)].
        """
        node_index = {}
        nodes: List[Tuple[float, float]] = []
        adj: Dict[int, List[Tuple[int, float]]] = {}

        def add_node(pt: Tuple[float, float]) -> int:
            if pt in node_index:
                return node_index[pt]
            idx = len(nodes)
            node_index[pt] = idx
            nodes.append(pt)
            adj[idx] = []
            return idx

        def add_edge(a: int, b: int):
            if a == b:
                return
            w = math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
            adj[a].append((b, w))
            adj[b].append((a, w))

        # From walls
        for wall in floor_plan_data.get('walls', []) or []:
            s = tuple(map(float, wall['start']))
            e = tuple(map(float, wall['end']))
            ia = add_node(s)
            ib = add_node(e)
            add_edge(ia, ib)

        # From room contours (subsample to control graph size)
        for room in floor_plan_data.get('rooms', []) or []:
            contour = room.get('contour')
            if not contour:
                continue
            # Handle OpenCV shapes: Nx1x2 or Nx2 converted to nested lists
            parsed_pts: List[Tuple[float, float]] = []
            for p in contour:
                try:
                    # Case: [[x, y]]
                    if isinstance(p, (list, tuple)) and len(p) == 1 and isinstance(p[0], (list, tuple)):
                        x, y = p[0][0], p[0][1]
                    # Case: [x, y]
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        x, y = p[0], p[1]
                    else:
                        continue
                    parsed_pts.append((float(x), float(y)))
                except Exception:
                    continue
            if not parsed_pts:
                continue
            if len(parsed_pts) > 200:
                step = max(1, len(parsed_pts)//200)
                parsed_pts = parsed_pts[::step]
            for i in range(len(parsed_pts)):
                ia = add_node(parsed_pts[i])
                ib = add_node(parsed_pts[(i+1) % len(parsed_pts)])
                add_edge(ia, ib)

        # From sampled edge points (dense but subsampled)
        for pt in floor_plan_data.get('edge_points', []) or []:
            try:
                ia = add_node((float(pt[0]), float(pt[1])))
            except Exception:
                continue

        # Try to connect nearby nodes to increase connectivity
        # Naive O(n^2) with limit for small graphs
        N = len(nodes)
        if N <= 1200:  # soft cap
            for i in range(N):
                xi, yi = nodes[i]
                for j in range(i+1, N):
                    xj, yj = nodes[j]
                    d = abs(xi - xj) + abs(yi - yj)
                    if d <= 12.0:  # close enough
                        add_edge(i, j)

        return nodes, adj

    def _nearest_node(self, nodes: List[Tuple[float, float]], pt: List[float]) -> int:
        bx, by = float(pt[0]), float(pt[1])
        best = None
        best_d = float('inf')
        for i, (x, y) in enumerate(nodes):
            d = (x - bx) * (x - bx) + (y - by) * (y - by)
            if d < best_d:
                best_d = d
                best = i
        return best if best is not None else 0

    def _route_path(self, nodes, adj, start: List[float], end: List[float]) -> List[List[float]]:
        """Dijkstra from nearest node to nearest node; returns list of points including start/end approximations.
        Fallback to straight line if graph empty or unreachable.
        """
        if not nodes:
            return [start, end]
        s_idx = self._nearest_node(nodes, start)
        t_idx = self._nearest_node(nodes, end)

        # Dijkstra
        dist = {s_idx: 0.0}
        prev = {}
        heap = [(0.0, s_idx)]
        visited = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == t_idx:
                break
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if t_idx not in dist:
            return [start, end]

        # Reconstruct path
        path_nodes = []
        cur = t_idx
        while cur in prev or cur == s_idx:
            path_nodes.append(cur)
            if cur == s_idx:
                break
            cur = prev[cur]
        path_nodes.reverse()

        # Convert nodes to points, prepend start and append end for nicer anchors
        path_pts = [list(start)]
        path_pts.extend([[nodes[i][0], nodes[i][1]] for i in path_nodes])
        path_pts.append(list(end))
        return path_pts
