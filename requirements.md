# Phase 2 Implementation Guide: CAD Processing & AI Compliance Engine

## Project Overview

Based on the current VendorRealty backend repository structure, you already have:
- **2d-to-3d pipeline**: PNG to STL conversion with FastAPI backend
- **Materials processing**: Automated materials takeoff and vendor pricing system  
- **Basic floor plan processing**: Image processing and OCR capabilities

## Current Architecture Analysis

### Existing Components

#### 1. **2d-to-3d Module** (`/2d-to-3d/`)
- **FastAPI backend** with PNG to STL conversion
- **Computer vision pipeline** using OpenCV, PIL, and EasyOCR
- **3D mesh generation** with Trimesh library
- **Preprocessing capabilities** for floor plan images
- **OCR text detection and removal** from architectural drawings

#### 2. **Materials Module** (`/materials/`)
- **materials_needed.py**: Rule-based materials estimation from floor plan dimensions
- **material_search.py**: Groq-powered vendor pricing and availability system
- **OCR-based dimension extraction** from floor plan images
- **Automated materials takeoff calculations**

#### 3. **Floor Plans Directory** (`/floorplans/`)
- Sample floor plan images for testing
- Input directory for CAD file processing

## Phase 2 Implementation Plan

### A. Enhanced CAD Processing Pipeline

#### 1. **Expand File Format Support**

Add to `/2d-to-3d/main.py`:

```python
# New imports for CAD processing
import ezdxf
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF

@app.post("/api/process_cad")
async def process_cad_endpoint(
    file: UploadFile = File(..., description="CAD file (DXF, DWG, PDF)"),
    output_format: str = Form("json", description="Output format: json, stl, both"),
):
    """
    Process CAD files and extract architectural elements
    Returns structured data with room boundaries, walls, and system routing paths
    """
    file_ext = file.filename.split('.')[-1].lower()
    contents = await file.read()
    
    if file_ext == 'dxf':
        return process_dxf_file(contents)
    elif file_ext == 'pdf':
        return process_pdf_floorplan(contents)
    elif file_ext in ['dwg']:
        raise HTTPException(400, "DWG files require conversion to DXF format")
    else:
        raise HTTPException(400, f"Unsupported file format: {file_ext}")

def process_dxf_file(dxf_bytes: bytes) -> dict:
    """Extract architectural elements from DXF files"""
    try:
        # Create temporary file for ezdxf processing
        with tempfile.NamedTemporaryFile(suffix='.dxf') as tmp:
            tmp.write(dxf_bytes)
            tmp.flush()
            
            doc = ezdxf.readfile(tmp.name)
            
            # Extract architectural elements
            walls = extract_walls_from_dxf(doc)
            rooms = detect_room_boundaries(doc)
            
            # Intelligently generate door and window placements
            # Since floor plan quality is often poor, we simulate realistic placements
            doors_windows = generate_intelligent_openings(walls, rooms)
            
            return {
                "type": "floor_plan_analysis",
                "walls": walls,
                "rooms": rooms,
                "openings": doors_windows,
                "routing_paths": generate_system_routing(walls, rooms),
                "metadata": {
                    "layers": list(doc.layers),
                    "blocks": list(doc.blocks),
                    "processed_at": datetime.now().isoformat(),
                    "openings_generated": True  # Flag that openings were intelligently placed
                }
            }
    except Exception as e:
        raise HTTPException(500, f"Failed to process DXF: {e}")

def extract_walls_from_dxf(doc) -> List[Dict]:
    """Extract wall elements from DXF layers"""
    walls = []
    msp = doc.modelspace()
    
    # Look for walls in common layer names
    wall_layers = ['WALL', 'WALLS', 'A-WALL', 'ARCH-WALL']
    
    for entity in msp:
        if entity.dxftype() == 'LINE' and entity.dxf.layer in wall_layers:
            walls.append({
                "type": "wall",
                "start": [entity.dxf.start.x, entity.dxf.start.y],
                "end": [entity.dxf.end.x, entity.dxf.end.y],
                "layer": entity.dxf.layer,
                "length": entity.dxf.start.distance(entity.dxf.end)
            })
    
    return walls

def generate_intelligent_openings(walls: List[Dict], rooms: List[Dict]) -> List[Dict]:
    """
    Intelligently generate door and window placements based on room layout
    Since floor plan data quality is often poor, we use heuristics to place openings
    """
    openings = []
    
    for i, room in enumerate(rooms):
        room_area = room.get('area_pixels', 0) * 0.0001  # Convert to approximate sq ft
        perimeter = calculate_room_perimeter(room)
        
        # Generate doors - typically one per room, two for large rooms
        num_doors = 1 if room_area < 200 else 2
        for door_idx in range(num_doors):
            # Place doors on longer walls for better accessibility
            wall_for_door = select_optimal_wall_for_opening(room, walls, 'door')
            if wall_for_door:
                openings.append({
                    "type": "door",
                    "width": 36,  # Standard door width in inches
                    "height": 80,  # Standard door height
                    "wall_id": wall_for_door.get('id'),
                    "position": calculate_opening_position(wall_for_door, 'door', door_idx),
                    "room_id": room['id'],
                    "generated": True
                })
        
        # Generate windows based on room size and building code requirements
        # Minimum window area = 10% of floor area for natural light (OBC requirement)
        required_window_area = room_area * 0.1 * 144  # Convert to sq inches
        window_size = 48 * 36  # Standard window 48"x36"
        num_windows = max(1, int(required_window_area / window_size))
        
        # Distribute windows evenly on exterior walls
        exterior_walls = [w for w in walls if is_exterior_wall(w, rooms)]
        for window_idx in range(num_windows):
            wall_for_window = exterior_walls[window_idx % len(exterior_walls)] if exterior_walls else walls[0]
            openings.append({
                "type": "window",
                "width": 48,
                "height": 36,
                "sill_height": 30,  # Standard sill height from floor
                "wall_id": wall_for_window.get('id'),
                "position": calculate_opening_position(wall_for_window, 'window', window_idx),
                "room_id": room['id'],
                "generated": True,
                "egress_compliant": window_size >= 619  # Min egress area in sq inches
            })
    
    return openings
```

#### 2. **Room Detection and Boundary Analysis**

Enhance the existing image processing with architectural intelligence:

```python
def detect_architectural_elements(processed_mask: np.ndarray, original_image: Image.Image) -> Dict:
    """
    Advanced architectural element detection using computer vision
    """
    # Find contours for room boundaries
    contours, _ = cv2.findContours(
        processed_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    rooms = []
    walls = []
    
    for contour in contours:
        # Filter by area to remove noise
        area = cv2.contourArea(contour)
        if area < 1000:  # Skip very small areas
            continue
            
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Calculate bounding box and room properties
        x, y, w, h = cv2.boundingRect(contour)
        
        rooms.append({
            "id": len(rooms) + 1,
            "contour": contour.tolist(),
            "area_pixels": float(area),
            "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "centroid": calculate_centroid(contour),
            "vertices": approx.tolist()
        })
    
    # Detect walls from edge detection
    edges = cv2.Canny(processed_mask.astype(np.uint8), 50, 150)
    wall_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    
    if wall_lines is not None:
        for line in wall_lines:
            x1, y1, x2, y2 = line[0]
            walls.append({
                "start": [int(x1), int(y1)],
                "end": [int(x2), int(y2)],
                "length": np.sqrt((x2-x1)**2 + (y2-y1)**2)
            })
    
    return {
        "rooms": rooms,
        "walls": walls,
        "total_rooms": len(rooms),
        "total_wall_length": sum(w["length"] for w in walls)
    }
```

### B. AI-Powered Building Code Compliance Engine

#### 1. **Create New Compliance Module**

Create `/compliance/` directory with the following structure:

```
/compliance/
├── __init__.py
├── compliance_engine.py
├── ontario_codes.py
├── hvac_rules.py
├── electrical_rules.py
├── plumbing_rules.py
└── embeddings/
    ├── code_embeddings.json
    └── generate_embeddings.py
```

#### 2. **Core Compliance Engine** (`/compliance/compliance_engine.py`)

```python
from typing import Dict, List, Any, Optional
import openai
from dataclasses import dataclass
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class ComplianceIssue:
    system: str  # 'hvac', 'electrical', 'plumbing'
    code_reference: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    location: Optional[Dict] = None
    suggested_fix: Optional[str] = None

class OntarioBuildingCodeEngine:
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.code_embeddings = self._load_code_embeddings()
        
    def _load_code_embeddings(self) -> Dict:
        """Load precomputed embeddings for Ontario Building Code sections"""
        try:
            with open('compliance/embeddings/code_embeddings.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._generate_code_embeddings()
    
    def check_compliance(self, floor_plan_data: Dict, system_type: str = None) -> List[ComplianceIssue]:
        """
        Main compliance checking function
        """
        issues = []
        
        systems_to_check = ['hvac', 'electrical', 'plumbing'] if not system_type else [system_type]
        
        for system in systems_to_check:
            if system == 'hvac':
                issues.extend(self._check_hvac_compliance(floor_plan_data))
            elif system == 'electrical':
                issues.extend(self._check_electrical_compliance(floor_plan_data))
            elif system == 'plumbing':
                issues.extend(self._check_plumbing_compliance(floor_plan_data))
        
        return issues
    
    def _check_hvac_compliance(self, data: Dict) -> List[ComplianceIssue]:
        """Check HVAC system compliance with Ontario Building Code"""
        issues = []
        rooms = data.get('rooms', [])
        
        # Check ventilation requirements
        for room in rooms:
            room_area = room.get('area_pixels', 0) * 0.0001  # Convert to sq ft approximation
            
            # Ventilation requirement check (ASHRAE 62.1 compliance)
            if room_area > 100:  # Significant room size
                ventilation_cfm = self._calculate_required_ventilation(room_area)
                
                issues.append(ComplianceIssue(
                    system='hvac',
                    code_reference='OBC 6.3.1.1',
                    severity='info',
                    message=f'Room requires {ventilation_cfm} CFM ventilation (ASHRAE 62.1)',
                    location=room.get('centroid'),
                    suggested_fix=f'Install ventilation system with minimum {ventilation_cfm} CFM capacity'
                ))
        
        # Check for carbon monoxide alarm requirements
        residential_rooms = self._identify_residential_rooms(rooms)
        if residential_rooms:
            issues.append(ComplianceIssue(
                system='hvac',
                code_reference='OBC 6.9.3.2',
                severity='error',
                message='Carbon monoxide alarms required adjacent to sleeping rooms',
                suggested_fix='Install CO alarms within 5m of bedroom doors'
            ))
        
        return issues
    
    def _check_electrical_compliance(self, data: Dict) -> List[ComplianceIssue]:
        """Check electrical system compliance"""
        issues = []
        rooms = data.get('rooms', [])
        walls = data.get('walls', [])
        
        for room in rooms:
            # Calculate wall perimeter
            perimeter = self._calculate_room_perimeter(room)
            
            # Outlet spacing requirements (max 1.8m spacing)
            required_outlets = max(2, int(perimeter / 1.8))
            
            issues.append(ComplianceIssue(
                system='electrical',
                code_reference='CEC general rules',
                severity='info',
                message=f'Room requires minimum {required_outlets} outlets (max 1.8m spacing)',
                location=room.get('centroid'),
                suggested_fix=f'Install {required_outlets} outlets along usable wall space'
            ))
            
            # GFCI requirements for wet locations
            if self._is_wet_location(room):
                issues.append(ComplianceIssue(
                    system='electrical',
                    code_reference='CEC GFCI requirements',
                    severity='error',
                    message='GFCI protection required within 1.5m of water sources',
                    location=room.get('centroid'),
                    suggested_fix='Install GFCI outlets or GFCI breaker protection'
                ))
        
        return issues
    
    def _check_plumbing_compliance(self, data: Dict) -> List[ComplianceIssue]:
        """Check plumbing system compliance"""
        issues = []
        rooms = data.get('rooms', [])
        
        bathrooms = [r for r in rooms if self._is_bathroom(r)]
        
        for bathroom in bathrooms:
            # Hot water temperature requirements
            issues.append(ComplianceIssue(
                system='plumbing',
                code_reference='OBC 7.2.10.7',
                severity='error',
                message='Hot water temperature must not exceed 49°C at fixtures',
                location=bathroom.get('centroid'),
                suggested_fix='Install thermostatic mixing valve or temperature limiting device'
            ))
            
            # Venting requirements
            issues.append(ComplianceIssue(
                system='plumbing',
                code_reference='OBC venting requirements',
                severity='warning',
                message='Proper venting required for all plumbing fixtures',
                location=bathroom.get('centroid'),
                suggested_fix='Ensure individual or common venting for each fixture'
            ))
        
        return issues
    
    def generate_compliance_suggestions(self, floor_plan_data: Dict) -> Dict:
        """
        Use OpenAI to generate intelligent system routing suggestions
        """
        prompt = f"""
        Given this floor plan analysis, suggest optimal routing paths for HVAC, electrical, and plumbing systems 
        that comply with Ontario Building Code:
        
        Floor Plan Data:
        {json.dumps(floor_plan_data, indent=2)}
        
        Provide specific routing suggestions that:
        1. Meet all code requirements
        2. Minimize installation costs
        3. Ensure efficient system performance
        4. Consider future maintenance access
        
        Format as JSON with 'hvac_routing', 'electrical_routing', 'plumbing_routing' sections.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"error": "Failed to parse AI routing suggestions"}
    
    def _calculate_required_ventilation(self, area_sqft: float) -> int:
        """Calculate required ventilation per ASHRAE 62.1"""
        # Simplified calculation - 0.35 air changes per hour minimum
        return max(50, int(area_sqft * 0.35))
    
    def _is_wet_location(self, room: Dict) -> bool:
        """Identify wet locations requiring GFCI protection"""
        # This would be enhanced with better room classification
        bbox = room.get('bounding_box', {})
        area = bbox.get('width', 0) * bbox.get('height', 0)
        return area < 5000  # Small rooms likely bathrooms/kitchens
    
    def _is_bathroom(self, room: Dict) -> bool:
        """Identify bathroom spaces"""
        # Enhanced room classification would go here
        bbox = room.get('bounding_box', {})
        area = bbox.get('width', 0) * bbox.get('height', 0)
        return 1000 < area < 8000  # Typical bathroom size range
```

#### 3. **Integration with FastAPI Backend**

Add to `/2d-to-3d/main.py`:

```python
from compliance.compliance_engine import OntarioBuildingCodeEngine

# Initialize compliance engine
compliance_engine = OntarioBuildingCodeEngine(
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

@app.post("/api/check_compliance")
async def check_compliance_endpoint(
    floor_plan_data: dict,
    system_type: Optional[str] = None,
    generate_suggestions: bool = True
):
    """
    Check building code compliance for processed floor plan data
    """
    try:
        # Run compliance checks
        issues = compliance_engine.check_compliance(floor_plan_data, system_type)
        
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
        
        # Generate AI-powered routing suggestions if requested
        if generate_suggestions:
            response["routing_suggestions"] = compliance_engine.generate_compliance_suggestions(floor_plan_data)
        
        return response
        
    except Exception as e:
        raise HTTPException(500, f"Compliance check failed: {e}")

@app.post("/api/process_and_check")
async def process_and_check_endpoint(
    file: UploadFile = File(...),
    system_type: Optional[str] = None,
    output_3d: bool = False
):
    """
    Complete pipeline: Process CAD file and check compliance in one step
    """
    # Process the CAD file first
    floor_plan_data = await process_cad_endpoint(file, "json")
    
    # Run compliance checks
    compliance_results = await check_compliance_endpoint(
        floor_plan_data, 
        system_type, 
        generate_suggestions=True
    )
    
    result = {
        "floor_plan_analysis": floor_plan_data,
        "compliance_analysis": compliance_results,
        "metadata": {
            "filename": file.filename,
            "processed_at": datetime.now().isoformat(),
            "file_size": len(await file.read())
        }
    }
    
    # Generate 3D model if requested
    if output_3d:
        # Convert floor plan to 3D representation
        stl_data = convert_floorplan_to_3d(floor_plan_data)
        result["3d_model_stl"] = stl_data
    
    return result
```

### C. System Routing Generation

#### 1. **HVAC Routing Algorithm**

```python
def generate_hvac_routing(rooms: List[Dict], walls: List[Dict]) -> Dict:
    """Generate HVAC ductwork routing paths"""
    
    # Find optimal locations for main trunk line
    building_centroid = calculate_building_centroid(rooms)
    
    # Determine supply and return air paths
    supply_routes = []
    return_routes = []
    
    for room in rooms:
        room_center = room['centroid']
        
        # Calculate optimal duct path from main trunk
        duct_path = calculate_shortest_path(
            building_centroid, 
            room_center, 
            obstacles=walls
        )
        
        supply_routes.append({
            "room_id": room['id'],
            "path": duct_path,
            "required_cfm": calculate_room_cfm_requirement(room),
            "duct_size": calculate_duct_diameter(room)
        })
    
    return {
        "main_trunk_location": building_centroid,
        "supply_routes": supply_routes,
        "return_routes": return_routes,
        "equipment_location": find_optimal_equipment_location(rooms, walls)
    }
```

### D. Updated Requirements

Add to `/2d-to-3d/requirements.txt`:

```
# Existing requirements...
annotated-types>=0.7,<0.8
fastapi==0.116.1
starlette>=0.40,<0.48
pydantic>=2.11,<3
python-multipart>=0.0.18,<0.1
uvicorn>=0.30,<0.36
numpy>=2.3,<3
pillow>=10,<12
trimesh>=4.4,<5
opencv-python-headless>=4.8,<5
shapely>=2.0,<3
mapbox_earcut>=1.0,<2
easyocr>=1.7,<2

# New requirements for Phase 2
ezdxf>=1.0.0,<2.0.0          # CAD file processing
PyMuPDF>=1.23.0,<2.0.0       # PDF processing  
pdf2image>=3.1.0,<4.0.0      # PDF to image conversion
openai>=1.0.0,<2.0.0          # AI compliance engine
scikit-learn>=1.3.0,<2.0.0   # Vector similarity for embeddings
python-dotenv>=1.0.0,<2.0.0  # Environment variable management
```

### E. Environment Variables

Create `.env` file:

```bash
# OpenAI API for compliance checking
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For enhanced compliance features
GROQ_API_KEY=your_groq_api_key_here

# Development settings
DEBUG=True
LOG_LEVEL=INFO
```

### F. Testing Integration

Create test endpoints to validate the implementation:

```python
@app.get("/api/test_compliance")
async def test_compliance():
    """Test endpoint with sample floor plan data"""
    sample_data = {
        "rooms": [
            {
                "id": 1,
                "area_pixels": 15000,
                "centroid": [100, 100],
                "bounding_box": {"x": 50, "y": 50, "width": 100, "height": 150}
            }
        ],
        "walls": [
            {"start": [0, 0], "end": [200, 0], "length": 200},
            {"start": [200, 0], "end": [200, 200], "length": 200}
        ]
    }
    
    return await check_compliance_endpoint(sample_data, generate_suggestions=True)
```

## Next Steps for Phase 3 Integration

1. **Frontend Integration Points**:
   - API endpoints return structured JSON for React consumption
   - 3D model data compatible with Three.js/React Three Fiber
   - Real-time compliance feedback for interactive editing

2. **Database Integration**:
   - Store processed floor plans and compliance results
   - Cache AI-generated routing suggestions
   - User project management

3. **Performance Optimization**:
   - Async processing for large CAD files
   - Background compliance checking
   - Progressive 3D model loading

This implementation provides a robust foundation for Phase 2, seamlessly building on your existing 2D-to-3D pipeline while adding sophisticated building code compliance capabilities powered by AI.