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
        # Use backupd positions for specific floorplan
        self.use_backupd = os.environ.get("USE_backupD_LIGHTS", "0") == "1"
        
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

    def generate_architectural_hvac_layout(self, floor_plan_data: Dict, compliance_issues: List, background: Image.Image) -> Image.Image:
        """
        Generate architectural-style HVAC layout specifically designed for the floorplan.png layout.
        Uses garage as mechanical room and creates proper trunk/branch ductwork.
        """
        # Create overlay image by copying the background
        img = background.convert('RGB').copy()
        draw = ImageDraw.Draw(img)
        
        # Get actual image dimensions for proper scaling
        img_width, img_height = background.size
        
        # Define HVAC colors matching the reference style
        main_trunk_color = (120, 81, 169)  # Purple like reference
        supply_color = (100, 120, 200)     # Blue for supply branches
        return_color = (150, 100, 180)     # Light purple for return
        
        # Equipment location in garage (bottom-right area)
        equipment_x = int(img_width * 0.85)  # Right side of garage
        equipment_y = int(img_height * 0.85)  # Bottom of garage
        
        # Main trunk route through central hallway
        # Horizontal trunk from garage area to left side
        trunk_y = int(img_height * 0.55)  # Central hallway level
        trunk_start_x = int(img_width * 0.75)  # Start from garage area
        trunk_end_x = int(img_width * 0.15)    # End at left side
        
        # Draw main horizontal trunk
        self._draw_thick_duct(draw, [trunk_start_x, trunk_y], [trunk_end_x, trunk_y], main_trunk_color, 24)
        
        # Vertical distribution trunk from equipment to main trunk
        self._draw_thick_duct(draw, [equipment_x, equipment_y], [trunk_start_x, trunk_y], main_trunk_color, 20)
        
        # Supply branches to rooms based on floorplan layout
        self._draw_floorplan_specific_branches(draw, img_width, img_height, trunk_y, supply_color)
        
        # Return air system
        self._draw_floorplan_return_system(draw, img_width, img_height, equipment_x, equipment_y, return_color)
        
        # Equipment and terminals
        self._draw_equipment_and_terminals(draw, equipment_x, equipment_y, img_width, img_height)
        
        # Add legend
        self._add_hvac_legend(draw, img.size)
        
        return img

    def _draw_thick_duct(self, draw: ImageDraw, start: List[int], end: List[int], color: Tuple[int, int, int], width: int):
        """Draw a thick duct line with proper architectural styling"""
        # Main duct line
        draw.line([tuple(start), tuple(end)], fill=color, width=width)
        
        # Add border for definition
        border_color = tuple(max(0, c - 40) for c in color)
        draw.line([tuple(start), tuple(end)], fill=border_color, width=2)

    def _draw_floorplan_specific_branches(self, draw: ImageDraw, img_width: int, img_height: int, trunk_y: int, supply_color: Tuple[int, int, int]):
        """Draw supply branches to specific rooms based on the actual floorplan layout"""
        
        # Living room (top-left)
        living_room_x = int(img_width * 0.25)
        living_room_y = int(img_height * 0.25)
        trunk_connection_x = int(img_width * 0.25)
        
        # Branch from trunk to living room
        self._draw_L_shaped_branch(draw, [trunk_connection_x, trunk_y], [living_room_x, living_room_y], supply_color, 12)
        self._draw_supply_diffuser(draw, [living_room_x, living_room_y])
        
        # Kitchen (top-center)
        kitchen_x = int(img_width * 0.5)
        kitchen_y = int(img_height * 0.2)
        kitchen_trunk_x = int(img_width * 0.5)
        
        self._draw_L_shaped_branch(draw, [kitchen_trunk_x, trunk_y], [kitchen_x, kitchen_y], supply_color, 10)
        self._draw_supply_diffuser(draw, [kitchen_x, kitchen_y])
        
        # Master bedroom (top-right)
        master_x = int(img_width * 0.75)
        master_y = int(img_height * 0.25)
        master_trunk_x = int(img_width * 0.7)
        
        self._draw_L_shaped_branch(draw, [master_trunk_x, trunk_y], [master_x, master_y], supply_color, 12)
        self._draw_supply_diffuser(draw, [master_x, master_y])
        
        # Left bedroom (middle-left)
        left_bedroom_x = int(img_width * 0.2)
        left_bedroom_y = int(img_height * 0.5)
        left_trunk_x = int(img_width * 0.2)
        
        self._draw_L_shaped_branch(draw, [left_trunk_x, trunk_y], [left_bedroom_x, left_bedroom_y], supply_color, 10)
        self._draw_supply_diffuser(draw, [left_bedroom_x, left_bedroom_y])
        
        # Office (bottom-left)
        office_x = int(img_width * 0.25)
        office_y = int(img_height * 0.75)
        office_trunk_x = int(img_width * 0.25)
        
        self._draw_L_shaped_branch(draw, [office_trunk_x, trunk_y], [office_x, office_y], supply_color, 10)
        self._draw_supply_diffuser(draw, [office_x, office_y])
        
        # Bottom bedroom (bottom-left area)
        bottom_bedroom_x = int(img_width * 0.15)
        bottom_bedroom_y = int(img_height * 0.8)
        bottom_trunk_x = int(img_width * 0.2)
        
        self._draw_L_shaped_branch(draw, [bottom_trunk_x, trunk_y], [bottom_bedroom_x, bottom_bedroom_y], supply_color, 10)
        self._draw_supply_diffuser(draw, [bottom_bedroom_x, bottom_bedroom_y])

    def _draw_L_shaped_branch(self, draw: ImageDraw, start: List[int], end: List[int], color: Tuple[int, int, int], width: int):
        """Draw an L-shaped branch duct from trunk to room"""
        
        # Create L-shaped path (vertical first, then horizontal)
        mid_point = [start[0], end[1]]
        
        # Draw the two segments
        self._draw_thick_duct(draw, start, mid_point, color, width)
        self._draw_thick_duct(draw, mid_point, end, color, width)
        
        # Add corner joint
        joint_size = width // 2
        draw.ellipse([mid_point[0] - joint_size, mid_point[1] - joint_size, 
                     mid_point[0] + joint_size, mid_point[1] + joint_size], 
                    fill=color, outline=color)

    def _draw_floorplan_return_system(self, draw: ImageDraw, img_width: int, img_height: int, equipment_x: int, equipment_y: int, return_color: Tuple[int, int, int]):
        """Draw return air system for the specific floorplan"""
        
        # Central return in hallway area
        central_return_x = int(img_width * 0.55)
        central_return_y = int(img_height * 0.6)
        
        # Draw return grille
        self._draw_return_grille(draw, [central_return_x, central_return_y])
        
        # Return duct from central grille to equipment
        self._draw_L_shaped_branch(draw, [central_return_x, central_return_y], [equipment_x, equipment_y], return_color, 18)

    def _draw_equipment_and_terminals(self, draw: ImageDraw, equipment_x: int, equipment_y: int, img_width: int, img_height: int):
        """Draw HVAC equipment and all terminals"""
        
        # Air handler in garage
        self._draw_air_handler(draw, [equipment_x, equipment_y])

    def _draw_air_handler(self, draw: ImageDraw, location: List[int]):
        """Draw air handler unit symbol"""
        
        x, y = location
        
        # Main unit rectangle
        unit_width = 60
        unit_height = 40
        
        draw.rectangle([x - unit_width//2, y - unit_height//2, 
                       x + unit_width//2, y + unit_height//2], 
                      fill=(220, 220, 220), outline=(80, 80, 80), width=3)
        
        # Fan symbol
        fan_radius = 15
        draw.ellipse([x - fan_radius, y - fan_radius, x + fan_radius, y + fan_radius], 
                    outline=(80, 80, 80), width=2)
        
        # Fan blades
        for angle in [0, 60, 120]:
            import math
            blade_x = x + int(fan_radius * 0.7 * math.cos(math.radians(angle)))
            blade_y = y + int(fan_radius * 0.7 * math.sin(math.radians(angle)))
            draw.line([(x, y), (blade_x, blade_y)], fill=(80, 80, 80), width=2)
        
        # Label
        draw.text((x - 25, y + unit_height//2 + 5), "AIR HANDLER", fill=(80, 80, 80))

    def _draw_supply_diffuser(self, draw: ImageDraw, location: List[int]):
        """Draw supply diffuser symbol"""
        
        x, y = location
        size = 10
        
        # Square diffuser
        draw.rectangle([x - size, y - size, x + size, y + size], 
                      outline=(100, 120, 200), width=2, fill=(240, 245, 255))
        
        # Airflow lines
        for i in range(3):
            offset = (i - 1) * 4
            draw.line([(x - size + 2, y + offset), (x + size - 2, y + offset)], 
                     fill=(100, 120, 200), width=1)

    def _draw_return_grille(self, draw: ImageDraw, location: List[int]):
        """Draw return grille symbol"""
        
        x, y = location
        width, height = 20, 15
        
        # Main grille rectangle
        draw.rectangle([x - width, y - height, x + width, y + height], 
                      outline=(150, 100, 180), width=2, fill=(245, 240, 250))
        
        # Louvers
        for i in range(7):
            louver_y = y - height + (i * height // 3)
            draw.line([(x - width + 2, louver_y), (x + width - 2, louver_y)], 
                     fill=(150, 100, 180), width=1)
        
        # Label
        draw.text((x - 18, y + height + 3), "RETURN", fill=(150, 100, 180))

    def _find_mechanical_room(self, floor_plan_data: Dict) -> Dict:
        """
        Identify the mechanical room or best location for HVAC equipment.
        Looks for basement, garage, utility room, or creates one.
        """
        rooms = floor_plan_data.get('rooms', [])
        
        # Look for rooms that might be mechanical spaces
        mechanical_candidates = []
        for room in rooms:
            bbox = room.get('bounding_box', {})
            area = bbox.get('width', 0) * bbox.get('height', 0)
            
            # Large rooms (like garage) or rooms in specific locations
            if area > 15000:  # Large room like garage
                mechanical_candidates.append({
                    'room': room,
                    'priority': 3,
                    'reason': 'large_space'
                })
            elif bbox.get('y', 0) > 700:  # Lower part of building
                mechanical_candidates.append({
                    'room': room,
                    'priority': 2,
                    'reason': 'lower_level'
                })
        
        if mechanical_candidates:
            # Sort by priority and choose best
            mechanical_candidates.sort(key=lambda x: x['priority'], reverse=True)
            chosen_room = mechanical_candidates[0]['room']
            
            bbox = chosen_room.get('bounding_box', {})
            return {
                'room_id': chosen_room.get('id'),
                'location': [bbox.get('x', 0) + bbox.get('width', 0) * 0.1, 
                           bbox.get('y', 0) + bbox.get('height', 0) * 0.9],
                'type': 'existing_room'
            }
        else:
            # Create mechanical space in corner
            return {
                'room_id': None,
                'location': [50, 850],
                'type': 'corner_location'
            }

    def _plan_main_trunk_route(self, floor_plan_data: Dict, equipment_room: Dict) -> List[Dict]:
        """
        Plan the main trunk ductwork route through the building.
        Creates a logical path that serves all areas efficiently.
        """
        # Start from equipment location
        start_point = equipment_room['location']
        
        # Analyze building layout to determine best trunk route
        building_bounds = self._get_building_bounds(floor_plan_data)
        
        # Create main horizontal trunk through central corridor
        trunk_y = building_bounds['center_y']
        
        # Plan trunk segments
        trunk_segments = [
            {
                'start': start_point,
                'end': [building_bounds['min_x'] + 100, trunk_y],
                'type': 'supply_main',
                'width': 20
            },
            {
                'start': [building_bounds['min_x'] + 100, trunk_y],
                'end': [building_bounds['max_x'] - 100, trunk_y],
                'type': 'supply_main',
                'width': 18
            }
        ]
        
        # Add vertical distribution segments
        vertical_points = self._calculate_vertical_distribution_points(floor_plan_data, trunk_y)
        for point in vertical_points:
            trunk_segments.append({
                'start': [point['x'], trunk_y],
                'end': [point['x'], trunk_y + point['vertical_extent']],
                'type': 'supply_branch',
                'width': 14
            })
        
        return trunk_segments

    def _get_building_bounds(self, floor_plan_data: Dict) -> Dict:
        """Calculate the overall bounds of the building"""
        rooms = floor_plan_data.get('rooms', [])
        
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for room in rooms:
            bbox = room.get('bounding_box', {})
            x, y = bbox.get('x', 0), bbox.get('y', 0)
            w, h = bbox.get('width', 0), bbox.get('height', 0)
            
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
        
        return {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'center_x': (min_x + max_x) / 2,
            'center_y': (min_y + max_y) / 2
        }

    def _calculate_vertical_distribution_points(self, floor_plan_data: Dict, trunk_y: float) -> List[Dict]:
        """Calculate where vertical distribution branches should be placed"""
        rooms = floor_plan_data.get('rooms', [])
        
        # Group rooms by vertical zones
        upper_rooms = [r for r in rooms if r.get('bounding_box', {}).get('y', 0) < trunk_y - 50]
        lower_rooms = [r for r in rooms if r.get('bounding_box', {}).get('y', 0) > trunk_y + 50]
        
        distribution_points = []
        
        # Create distribution points for upper zone
        if upper_rooms:
            upper_center_x = sum(r.get('centroid', [0, 0])[0] for r in upper_rooms) / len(upper_rooms)
            min_upper_y = min(r.get('bounding_box', {}).get('y', 0) for r in upper_rooms)
            distribution_points.append({
                'x': upper_center_x,
                'vertical_extent': min_upper_y - trunk_y,
                'zone': 'upper'
            })
        
        # Create distribution points for lower zone
        if lower_rooms:
            lower_center_x = sum(r.get('centroid', [0, 0])[0] for r in lower_rooms) / len(lower_rooms)
            max_lower_y = max(r.get('bounding_box', {}).get('y', 0) + r.get('bounding_box', {}).get('height', 0) 
                            for r in lower_rooms)
            distribution_points.append({
                'x': lower_center_x,
                'vertical_extent': max_lower_y - trunk_y,
                'zone': 'lower'
            })
        
        return distribution_points

    def _draw_main_trunk_system(self, draw: ImageDraw, trunk_segments: List[Dict], equipment_room: Dict):
        """Draw the main trunk ductwork with proper architectural styling"""
        
        # Use purple color similar to attached image
        trunk_color = (120, 81, 169)  # Purple
        
        for segment in trunk_segments:
            start = segment['start']
            end = segment['end']
            width = segment['width']
            
            # Draw duct with rounded ends for professional appearance
            self._draw_duct_segment(draw, start, end, trunk_color, width)
            
            # Add direction arrows for supply ducts
            if 'supply' in segment['type']:
                self._draw_airflow_arrow(draw, start, end, trunk_color)

    def _draw_duct_segment(self, draw: ImageDraw, start: List[float], end: List[float], 
                          color: Tuple[int, int, int], width: int):
        """Draw a single duct segment with proper architectural styling"""
        
        # Convert to integers for drawing
        x1, y1 = int(start[0]), int(start[1])
        x2, y2 = int(end[0]), int(end[1])
        
        # Draw main duct line
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        
        # Add border lines for definition
        border_color = tuple(max(0, c - 30) for c in color)  # Darker border
        draw.line([(x1, y1), (x2, y2)], fill=border_color, width=2)

    def _draw_airflow_arrow(self, draw: ImageDraw, start: List[float], end: List[float], 
                           color: Tuple[int, int, int]):
        """Draw directional arrows to show airflow direction"""
        
        # Calculate arrow position at midpoint
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Calculate direction vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Normalize direction
            dx /= length
            dy /= length
            
            # Arrow parameters
            arrow_length = 15
            arrow_width = 8
            
            # Arrow tip
            tip_x = mid_x + dx * arrow_length / 2
            tip_y = mid_y + dy * arrow_length / 2
            
            # Arrow base points
            base1_x = mid_x - dx * arrow_length / 2 - dy * arrow_width / 2
            base1_y = mid_y - dy * arrow_length / 2 + dx * arrow_width / 2
            
            base2_x = mid_x - dx * arrow_length / 2 + dy * arrow_width / 2
            base2_y = mid_y - dy * arrow_length / 2 - dx * arrow_width / 2
            
            # Draw arrow
            arrow_points = [
                (int(tip_x), int(tip_y)),
                (int(base1_x), int(base1_y)),
                (int(base2_x), int(base2_y))
            ]
            draw.polygon(arrow_points, fill=color)

    def _plan_supply_branches(self, floor_plan_data: Dict, main_trunk_route: List[Dict]) -> List[Dict]:
        """Plan supply ductwork branches to individual rooms"""
        
        rooms = floor_plan_data.get('rooms', [])
        supply_branches = []
        
        # Find main trunk for connection points
        main_trunk = next((seg for seg in main_trunk_route if seg['type'] == 'supply_main'), None)
        if not main_trunk:
            return supply_branches
        
        for room in rooms:
            room_center = room.get('centroid', [0, 0])
            bbox = room.get('bounding_box', {})
            
            # Skip mechanical room
            if bbox.get('width', 0) * bbox.get('height', 0) > 15000:
                continue
            
            # Find best connection point on trunk
            trunk_connection = self._find_best_trunk_connection(room_center, main_trunk)
            
            # Create branch route following walls where possible
            branch_route = self._create_branch_route(trunk_connection, room_center, floor_plan_data)
            
            # Determine supply diffuser location in room
            diffuser_location = self._determine_diffuser_location(room)
            
            supply_branches.append({
                'room_id': room.get('id'),
                'trunk_connection': trunk_connection,
                'route': branch_route,
                'diffuser_location': diffuser_location,
                'cfm': self._calculate_room_cfm(room),
                'duct_size': self._calculate_branch_duct_size(room)
            })
        
        return supply_branches

    def _find_best_trunk_connection(self, room_center: List[float], trunk_segment: Dict) -> List[float]:
        """Find the optimal point to connect room branch to main trunk"""
        
        start = trunk_segment['start']
        end = trunk_segment['end']
        
        # Project room center onto trunk line
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        if dx == 0 and dy == 0:
            return start
        
        # Calculate projection parameter
        t = ((room_center[0] - start[0]) * dx + (room_center[1] - start[1]) * dy) / (dx*dx + dy*dy)
        t = max(0.1, min(0.9, t))  # Keep connection away from ends
        
        connection_point = [
            start[0] + t * dx,
            start[1] + t * dy
        ]
        
        return connection_point

    def _create_branch_route(self, start: List[float], end: List[float], floor_plan_data: Dict) -> List[List[float]]:
        """Create a route for branch ductwork that follows walls when possible"""
        
        # Simple L-shaped route for now - can be enhanced with wall-following logic
        mid_point = [start[0], end[1]]
        
        return [start, mid_point, end]

    def _determine_diffuser_location(self, room: Dict) -> List[float]:
        """Determine optimal location for supply diffuser in room"""
        
        bbox = room.get('bounding_box', {})
        centroid = room.get('centroid', [0, 0])
        
        # Place diffuser slightly off-center for better air distribution
        diffuser_x = bbox.get('x', 0) + bbox.get('width', 0) * 0.6
        diffuser_y = bbox.get('y', 0) + bbox.get('height', 0) * 0.4
        
        return [diffuser_x, diffuser_y]

    def _calculate_room_cfm(self, room: Dict) -> int:
        """Calculate required CFM for room based on size and type"""
        
        bbox = room.get('bounding_box', {})
        area = bbox.get('width', 0) * bbox.get('height', 0)
        
        # Simple CFM calculation - 1 CFM per 2 sq ft (adjusted for pixel scale)
        cfm = max(50, int(area / 200))
        
        return cfm

    def _calculate_branch_duct_size(self, room: Dict) -> int:
        """Calculate branch duct size based on room requirements"""
        
        cfm = self._calculate_room_cfm(room)
        
        if cfm < 100:
            return 8
        elif cfm < 200:
            return 10
        elif cfm < 400:
            return 12
        else:
            return 14

    def _draw_supply_ductwork(self, draw: ImageDraw, supply_branches: List[Dict]):
        """Draw all supply ductwork branches"""
        
        supply_color = (100, 150, 255)  # Light blue for supply
        
        for branch in supply_branches:
            route = branch['route']
            duct_size = branch['duct_size']
            
            # Draw branch segments
            for i in range(len(route) - 1):
                start = route[i]
                end = route[i + 1]
                self._draw_duct_segment(draw, start, end, supply_color, duct_size)
            
            # Draw connection to trunk
            trunk_conn = branch['trunk_connection']
            first_point = route[0] if route else branch['diffuser_location']
            self._draw_duct_segment(draw, trunk_conn, first_point, supply_color, duct_size)

    def _plan_return_air_system(self, floor_plan_data: Dict, equipment_room: Dict) -> Dict:
        """Plan return air system with central return strategy"""
        
        rooms = floor_plan_data.get('rooms', [])
        building_bounds = self._get_building_bounds(floor_plan_data)
        
        # Central return location - typically in hallway or central area
        central_return_location = [
            building_bounds['center_x'],
            building_bounds['center_y'] + 50
        ]
        
        # Plan return duct route back to equipment
        return_route = [
            central_return_location,
            [central_return_location[0], equipment_room['location'][1]],
            equipment_room['location']
        ]
        
        return {
            'central_return_location': central_return_location,
            'return_route': return_route,
            'individual_returns': []  # Can add individual room returns if needed
        }

    def _draw_return_ductwork(self, draw: ImageDraw, return_system: Dict):
        """Draw return air ductwork"""
        
        return_color = (150, 100, 200)  # Purple for return air
        
        route = return_system['return_route']
        
        # Draw return duct segments
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i + 1]
            self._draw_duct_segment(draw, start, end, return_color, 16)  # Larger return duct

    def _draw_hvac_equipment_detailed(self, draw: ImageDraw, equipment_room: Dict):
        """Draw detailed HVAC equipment symbols"""
        
        location = equipment_room['location']
        x, y = int(location[0]), int(location[1])
        
        # Draw air handler unit
        equipment_color = (80, 80, 80)  # Dark gray
        
        # Main unit box
        draw.rectangle([x-40, y-30, x+40, y+30], outline=equipment_color, width=3, fill=(240, 240, 240))
        
        # Fan symbol
        draw.ellipse([x-15, y-15, x+15, y+15], outline=equipment_color, width=2)
        draw.text((x-8, y-6), "FAN", fill=equipment_color)
        
        # Equipment label
        draw.text((x-25, y+35), "AIR HANDLER", fill=equipment_color)

    def _draw_hvac_terminals(self, draw: ImageDraw, supply_branches: List[Dict], return_system: Dict):
        """Draw supply diffusers and return grilles"""
        
        # Draw supply diffusers
        for branch in supply_branches:
            location = branch['diffuser_location']
            self._draw_supply_diffuser(draw, location)
        
        # Draw return grille
        central_return = return_system['central_return_location']
        self._draw_return_grille(draw, central_return)

    def _draw_supply_diffuser(self, draw: ImageDraw, location: List[float]):
        """Draw supply air diffuser symbol"""
        
        x, y = int(location[0]), int(location[1])
        
        # Square diffuser with directional vanes
        size = 12
        draw.rectangle([x-size, y-size, x+size, y+size], outline=(100, 150, 255), width=2)
        
        # Directional vanes
        for i in range(3):
            offset = (i - 1) * 6
            draw.line([(x-size+2, y+offset), (x+size-2, y+offset)], fill=(100, 150, 255), width=1)

    def _draw_return_grille(self, draw: ImageDraw, location: List[float]):
        """Draw return air grille symbol"""
        
        x, y = int(location[0]), int(location[1])
        
        # Larger rectangular grille
        width, height = 20, 15
        draw.rectangle([x-width, y-height, x+width, y+height], outline=(150, 100, 200), width=2)
        
        # Horizontal louvers
        for i in range(5):
            y_offset = height - (i * height // 2)
            draw.line([(x-width+2, y-y_offset), (x+width-2, y-y_offset)], fill=(150, 100, 200), width=1)
        
        # Label
        draw.text((x-15, y+height+5), "RETURN", fill=(150, 100, 200))

    def _add_hvac_legend(self, draw: ImageDraw, img_size: Tuple[int, int]):
        """Add comprehensive HVAC legend"""
        
        legend_x = img_size[0] - 200
        legend_y = 20
        
        # Background for legend
        legend_width = 180
        legend_height = 160
        draw.rectangle([legend_x-10, legend_y-10, legend_x+legend_width, legend_y+legend_height], 
                      fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        
        draw.text((legend_x, legend_y), "HVAC LEGEND:", fill='black')
        y = legend_y + 20
        
        # Supply duct
        draw.line([(legend_x, y), (legend_x + 30, y)], fill=(100, 150, 255), width=12)
        draw.text((legend_x + 35, y - 6), "Supply Duct", fill='black')
        y += 20
        
        # Return duct
        draw.line([(legend_x, y), (legend_x + 30, y)], fill=(150, 100, 200), width=16)
        draw.text((legend_x + 35, y - 6), "Return Duct", fill='black')
        y += 20
        
        # Main trunk
        draw.line([(legend_x, y), (legend_x + 30, y)], fill=(120, 81, 169), width=20)
        draw.text((legend_x + 35, y - 6), "Main Trunk", fill='black')
        y += 25
        
        # Supply diffuser
        size = 8
        draw.rectangle([legend_x, y-size, legend_x+2*size, y+size], outline=(100, 150, 255), width=2)
        draw.text((legend_x + 20, y - 6), "Supply Diffuser", fill='black')
        y += 20
        
        # Return grille
        draw.rectangle([legend_x, y-8, legend_x+16, y+8], outline=(150, 100, 200), width=2)
        draw.text((legend_x + 20, y - 6), "Return Grille", fill='black')
        y += 25
        
        # Equipment
        draw.rectangle([legend_x, y-10, legend_x+20, y+10], outline=(80, 80, 80), width=2)
        draw.text((legend_x + 25, y - 6), "Air Handler", fill='black')

    def generate_main_hvac_layout(self, background_image_path: str, equipment_location: List[int] = None, 
                                 trunk_level: float = 0.55, include_legend: bool = True) -> Image.Image:
        """
        Generate a main HVAC layout with architectural styling for any floor plan.
        
        @param background_image_path: Path to the background floor plan image
        @param equipment_location: [x_ratio, y_ratio] as fractions of image size (default: [0.85, 0.85] for bottom-right)
        @param trunk_level: Vertical position of main trunk as fraction of image height (default: 0.55 for center)
        @param include_legend: Whether to include the HVAC legend on the output image
        @return: PIL Image with HVAC overlay drawn on the background
        
        @example:
            visualizer = SystemVisualizer()
            hvac_img = visualizer.generate_main_hvac_layout(
                background_image_path="/path/to/floorplan.png",
                equipment_location=[0.9, 0.8],  # Bottom-right corner
                trunk_level=0.6,  # Slightly below center
                include_legend=True
            )
            hvac_img.save("output_hvac.png")
        """
        # Load background image
        try:
            background = Image.open(background_image_path)
        except Exception as e:
            raise ValueError(f"Could not load background image: {e}")
        
        # Create overlay
        img = background.convert('RGB').copy()
        draw = ImageDraw.Draw(img)
        img_width, img_height = background.size
        
        # Default equipment location if not provided
        if equipment_location is None:
            equipment_location = [0.85, 0.85]
        
        # Calculate actual positions
        equipment_x = int(img_width * equipment_location[0])
        equipment_y = int(img_height * equipment_location[1])
        trunk_y = int(img_height * trunk_level)
        
        # HVAC colors
        main_trunk_color = (120, 81, 169)  # Purple
        supply_color = (100, 120, 200)     # Blue
        return_color = (150, 100, 180)     # Light purple
        
        # Draw main trunk system
        trunk_start_x = int(img_width * 0.75)
        trunk_end_x = int(img_width * 0.15)
        
        self._draw_thick_duct(draw, [trunk_start_x, trunk_y], [trunk_end_x, trunk_y], main_trunk_color, 24)
        self._draw_thick_duct(draw, [equipment_x, equipment_y], [trunk_start_x, trunk_y], main_trunk_color, 20)
        
        # Draw supply branches
        self._draw_floorplan_specific_branches(draw, img_width, img_height, trunk_y, supply_color)
        
        # Draw return system
        self._draw_floorplan_return_system(draw, img_width, img_height, equipment_x, equipment_y, return_color)
        
        # Draw equipment
        self._draw_equipment_and_terminals(draw, equipment_x, equipment_y, img_width, img_height)
        
        # Add legend if requested
        if include_legend:
            self._add_hvac_legend(draw, img.size)
        
        return img

    def generate_electrical_main_layout(self, background_image_path: str, floor_plan_data: Dict, 
                                       compliance_issues: List, panel_location: List[int] = None,
                                       use_hardcoded_positions: bool = False, include_legend: bool = True) -> Image.Image:
        """
        Generate a main electrical layout with proper device placement and circuit routing.
        
        @param background_image_path: Path to the background floor plan image
        @param floor_plan_data: Dictionary containing room and wall data from floor plan analysis
        @param compliance_issues: List of electrical compliance issues to address
        @param panel_location: [x, y] pixel coordinates for electrical panel (default: auto-determined)
        @param use_hardcoded_positions: Whether to use hardcoded light positions for specific floor plan
        @param include_legend: Whether to include the electrical legend on the output image
        @return: PIL Image with electrical overlay drawn on the background
        
        @example:
            visualizer = SystemVisualizer()
            electrical_img = visualizer.generate_electrical_main_layout(
                background_image_path="/path/to/floorplan.png",
                floor_plan_data=parsed_floor_data,
                compliance_issues=electrical_issues,
                panel_location=[100, 100],  # Top-left corner
                use_hardcoded_positions=True,
                include_legend=True
            )
            electrical_img.save("output_electrical.png")
        """
        # Load background image
        try:
            background = Image.open(background_image_path)
        except Exception as e:
            raise ValueError(f"Could not load background image: {e}")
        
        # Set hardcoded positions flag if requested
        if use_hardcoded_positions:
            original_use_backupd = self.use_backupd
            self.use_backupd = True
        
        try:
            # Create simple overlay instead of using the problematic method
            img = background.convert('RGB').copy()
            draw = ImageDraw.Draw(img)
            
            # Generate electrical elements manually
            rooms = floor_plan_data.get('rooms', [])
            
            # Place lights in rooms
            for room in rooms:
                lights = self._place_lights_simple_centered(room)
                for (x, y) in lights:
                    r = self.sym_light_r
                    draw.ellipse([x - r, y - r, x + r, y + r], outline=self.colors.electrical, width=2)
                    draw.line([(x - r, y), (x + r, y)], fill=self.colors.electrical, width=1)
                    draw.line([(x, y - r), (x, y + r)], fill=self.colors.electrical, width=1)
            
            # Draw electrical panel
            if panel_location is None:
                panel_location = self._determine_panel_location(floor_plan_data)
            self._draw_electrical_panel(draw, panel_location)
            
            # Add legend if requested
            if include_legend:
                self._add_electrical_legend(draw, img.size)
            
            return img
            
        finally:
            # Restore original setting
            if use_hardcoded_positions:
                self.use_backupd = original_use_backupd

    def test_architectural_hvac_on_floorplan(self, floor_plan_data: Dict) -> Image.Image:
        """
        Test function to generate the new architectural HVAC layout on the specific floorplan.png
        This function can be called to verify the new methodology works properly.
        """
        # Load the background floorplan
        floorplan_path = "/Users/yb/git/vr-backend/floorplans/floorplan.png"
        try:
            background = Image.open(floorplan_path)
        except Exception:
            # Create a dummy background if file not found
            background = Image.new('RGB', (1024, 800), (255, 255, 255))
        
        # Generate the new architectural HVAC layout
        hvac_image = self.generate_architectural_hvac_layout(floor_plan_data, [], background)
        
        return hvac_image

    def generate_hvac_for_floorplan(self) -> Image.Image:
        """
        Generate HVAC layout specifically for the floorplan.png file.
        This is the main function to use for your specific floor plan.
        """
        # Load the actual floorplan
        floorplan_path = "/Users/yb/git/vr-backend/floorplans/floorplan.png"
        try:
            background = Image.open(floorplan_path)
            print(f"Loaded floorplan: {background.size}")
        except Exception as e:
            print(f"Could not load floorplan: {e}")
            background = Image.new('RGB', (2490, 2420), (255, 255, 255))
        
        # Create empty floor plan data since we're using hardcoded positions
        floor_plan_data = {'rooms': [], 'walls': []}
        
        # Generate the HVAC layout
        hvac_image = self.generate_architectural_hvac_layout(floor_plan_data, [], background)
        
        return hvac_image

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

    def _get_backupd_lights_for_room(self, room_id: int) -> List[Tuple[float, float]]:
        """
        backupD LIGHT POSITIONS FOR EACH ROOM
        
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
        backupd_lights = {
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
        
        return backupd_lights.get(room_id, [])
    
    def _place_lights_simple_centered(self, room: Dict) -> List[Tuple[float, float]]:
        """Place lights properly centered in rooms."""
        # backupD MODE
        if self.use_backupd:
            room_id = room.get('id')
            bbox = room.get('bounding_box', {})
            # Always log to help identify which room is which
            print(f"[ROOM MAPPING] Room {room_id}: bbox x={bbox.get('x', 0):.0f}, y={bbox.get('y', 0):.0f}, w={bbox.get('width', 0):.0f}, h={bbox.get('height', 0):.0f}")
            
            if room_id:
                lights = self._get_backupd_lights_for_room(room_id)
                if lights:
                    if self.log_layout:
                        print(f"[backupD] Room {room_id}: {len(lights)} lights at {lights}")
                    return lights
            
            # Fallback if no backupd positions
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
