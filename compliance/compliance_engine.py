"""
Ontario Building Code Compliance Engine
Main engine for checking building code compliance for HVAC, electrical, and plumbing systems
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os
import numpy as np

@dataclass
class ComplianceIssue:
    system: str  # 'hvac', 'electrical', 'plumbing'
    code_reference: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    location: Optional[Dict] = None
    suggested_fix: Optional[str] = None

class OntarioBuildingCodeEngine:
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        # Import OpenAI only if we have a key
        if self.openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            except ImportError:
                self.openai_client = None
        else:
            self.openai_client = None
            
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
        
        for room in rooms:
            # Convert pixels to approximate square feet (assuming 1px = 0.1ft for now)
            room_area = room.get('area_pixels', 0) * 0.01
            
            if room_area > 100:  # Significant room size
                ventilation_cfm = self._calculate_required_ventilation(room_area)
                
                issues.append(ComplianceIssue(
                    system='hvac',
                    code_reference='OBC 6.3.1.1',
                    severity='info',
                    message=f'Room {room.get("id", "?")} requires {ventilation_cfm} CFM ventilation',
                    location=room.get('centroid'),
                    suggested_fix=f'Install ventilation system with minimum {ventilation_cfm} CFM capacity'
                ))
        
        # Check for carbon monoxide alarm requirements
        if len(rooms) > 0:
            issues.append(ComplianceIssue(
                system='hvac',
                code_reference='OBC 6.9.3.2',
                severity='warning',
                message='Carbon monoxide alarms required if fuel-burning appliances present',
                suggested_fix='Install CO alarms within 5m of sleeping areas if applicable'
            ))
        
        return issues
    
    def _check_electrical_compliance(self, data: Dict) -> List[ComplianceIssue]:
        """Check electrical system compliance"""
        issues = []
        rooms = data.get('rooms', [])
        
        for room in rooms:
            # Calculate approximate perimeter from bounding box
            bbox = room.get('bounding_box', {})
            perimeter = 2 * (bbox.get('width', 0) + bbox.get('height', 0)) * 0.1  # Convert to feet
            
            if perimeter > 0:
                # Outlet spacing requirements (max 12 feet spacing per CEC)
                required_outlets = max(2, int(perimeter / 12))
                
                issues.append(ComplianceIssue(
                    system='electrical',
                    code_reference='CEC 26-722',
                    severity='info',
                    message=f'Room {room.get("id", "?")} requires minimum {required_outlets} outlets',
                    location=room.get('centroid'),
                    suggested_fix=f'Install {required_outlets} outlets with max 12ft spacing'
                ))
            
            # Check for GFCI requirements in wet locations
            if self._is_wet_location(room):
                issues.append(ComplianceIssue(
                    system='electrical',
                    code_reference='CEC 26-700',
                    severity='error',
                    message='GFCI protection required within 1.5m of sinks/water sources',
                    location=room.get('centroid'),
                    suggested_fix='Install GFCI outlets or GFCI breaker protection'
                ))
        
        return issues
    
    def _check_plumbing_compliance(self, data: Dict) -> List[ComplianceIssue]:
        """Check plumbing system compliance"""
        issues = []
        rooms = data.get('rooms', [])
        
        for room in rooms:
            if self._is_bathroom(room) or self._is_kitchen(room):
                # Hot water temperature requirements
                issues.append(ComplianceIssue(
                    system='plumbing',
                    code_reference='OBC 7.2.10.7',
                    severity='error',
                    message='Hot water temperature must not exceed 49°C at fixtures',
                    location=room.get('centroid'),
                    suggested_fix='Install thermostatic mixing valve or temperature limiting device'
                ))
                
                # Venting requirements
                issues.append(ComplianceIssue(
                    system='plumbing',
                    code_reference='OBC 7.9.3',
                    severity='warning',
                    message='Proper venting required for all plumbing fixtures',
                    location=room.get('centroid'),
                    suggested_fix='Ensure individual or common venting for each fixture'
                ))
        
        return issues
    
    def generate_compliance_suggestions(self, floor_plan_data: Dict) -> Dict:
        """
        Generate intelligent system routing suggestions
        If OpenAI is not available, use rule-based suggestions
        """
        if self.openai_client:
            return self._generate_ai_suggestions(floor_plan_data)
        else:
            return self._generate_rule_based_suggestions(floor_plan_data)
    
    def _generate_ai_suggestions(self, floor_plan_data: Dict) -> Dict:
        """Use OpenAI to generate intelligent routing suggestions"""
        try:
            prompt = f"""
            Given this floor plan analysis, suggest optimal routing paths for HVAC, electrical, and plumbing systems 
            that comply with Ontario Building Code:
            
            Floor Plan Data:
            - Number of rooms: {len(floor_plan_data.get('rooms', []))}
            - Number of walls: {len(floor_plan_data.get('walls', []))}
            
            Provide specific routing suggestions that:
            1. Meet all code requirements
            2. Minimize installation costs
            3. Ensure efficient system performance
            4. Consider future maintenance access
            
            Format as JSON with 'hvac_routing', 'electrical_routing', 'plumbing_routing' sections.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"Failed to generate AI suggestions: {e}"}
    
    def _generate_rule_based_suggestions(self, floor_plan_data: Dict) -> Dict:
        """Generate rule-based routing suggestions without AI"""
        rooms = floor_plan_data.get('rooms', [])
        
        # Calculate building centroid for main trunk placement
        if rooms:
            avg_x = np.mean([r.get('centroid', [0, 0])[0] for r in rooms])
            avg_y = np.mean([r.get('centroid', [0, 0])[1] for r in rooms])
            building_centroid = [avg_x, avg_y]
        else:
            building_centroid = [0, 0]
        
        return {
            "hvac_routing": {
                "main_trunk_location": building_centroid,
                "supply_strategy": "Central trunk with branch ducts to each room",
                "return_strategy": "Central return with transfer grilles",
                "equipment_location": "Mechanical room or basement"
            },
            "electrical_routing": {
                "panel_location": "Near main entrance or garage",
                "circuit_strategy": "Separate circuits for each room plus dedicated appliance circuits",
                "wire_routing": "Through walls and ceiling/floor joists"
            },
            "plumbing_routing": {
                "main_stack_location": "Central location near wet rooms",
                "supply_strategy": "Home run system from manifold",
                "drain_strategy": "Gravity drainage to main stack"
            }
        }
    
    def _calculate_required_ventilation(self, area_sqft: float) -> int:
        """Calculate required ventilation per ASHRAE 62.1"""
        # Simplified: 0.35 air changes per hour minimum
        cfm = max(50, int(area_sqft * 8.5 * 0.35 / 60))  # Assuming 8.5ft ceiling
        return cfm
    
    def _is_wet_location(self, room: Dict) -> bool:
        """Identify wet locations requiring GFCI protection"""
        bbox = room.get('bounding_box', {})
        area = bbox.get('width', 0) * bbox.get('height', 0)
        # Small rooms (< 100 sq ft) are likely bathrooms/powder rooms
        return area > 0 and area < 10000  # 10000 px^2 ≈ 100 sq ft
    
    def _is_bathroom(self, room: Dict) -> bool:
        """Identify bathroom spaces"""
        bbox = room.get('bounding_box', {})
        area = bbox.get('width', 0) * bbox.get('height', 0)
        # Typical bathroom: 40-100 sq ft
        return 4000 < area < 10000
    
    def _is_kitchen(self, room: Dict) -> bool:
        """Identify kitchen spaces"""
        bbox = room.get('bounding_box', {})
        area = bbox.get('width', 0) * bbox.get('height', 0)
        # Typical kitchen: 100-200 sq ft
        return 10000 < area < 20000
