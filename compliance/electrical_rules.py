"""
Electrical Compliance Rules based on Canadian Electrical Code (CEC)
"""

from typing import Dict, List
from .compliance_engine import ComplianceIssue

class ElectricalComplianceChecker:
    """Detailed electrical compliance checking based on CEC"""
    
    @staticmethod
    def check_outlet_spacing(room_data: Dict) -> List[ComplianceIssue]:
        """Check outlet spacing requirements per CEC"""
        issues = []
        
        # Calculate room perimeter from bounding box
        bbox = room_data.get('bounding_box', {})
        perimeter_ft = 2 * (bbox.get('width', 0) + bbox.get('height', 0)) * 0.1
        
        if perimeter_ft > 0:
            # CEC requires outlets every 12 feet along walls
            required_outlets = max(2, int(perimeter_ft / 12) + 1)
            
            issues.append(ComplianceIssue(
                system='electrical',
                code_reference='CEC 26-722',
                severity='error',
                message=f'Minimum {required_outlets} outlets required (12ft max spacing)',
                location=room_data.get('centroid'),
                suggested_fix=f'Install {required_outlets} duplex receptacles along wall space'
            ))
        
        return issues
    
    @staticmethod
    def check_gfci_requirements(room_type: str, room_data: Dict) -> List[ComplianceIssue]:
        """Check GFCI protection requirements"""
        issues = []
        
        gfci_required_rooms = ['bathroom', 'kitchen', 'laundry', 'garage', 'outdoor']
        
        if room_type in gfci_required_rooms:
            issues.append(ComplianceIssue(
                system='electrical',
                code_reference='CEC 26-700(11)',
                severity='error',
                message=f'GFCI protection required for {room_type} receptacles',
                location=room_data.get('centroid'),
                suggested_fix='Install GFCI receptacles or GFCI circuit breaker'
            ))
        
        return issues
    
    @staticmethod
    def check_lighting_requirements(room_data: Dict) -> List[ComplianceIssue]:
        """Check lighting requirements per code"""
        issues = []
        area_sqft = room_data.get('area_pixels', 0) * 0.01
        
        if area_sqft > 0:
            # Minimum lighting: 10 watts per sq meter (approx 1 watt per sq ft)
            min_watts = int(area_sqft * 1.0)
            
            issues.append(ComplianceIssue(
                system='electrical',
                code_reference='CEC lighting requirements',
                severity='info',
                message=f'Minimum lighting capacity: {min_watts} watts',
                location=room_data.get('centroid'),
                suggested_fix='Install switched lighting fixtures with adequate capacity'
            ))
        
        return issues
    
    @staticmethod
    def check_circuit_requirements(room_type: str, room_data: Dict) -> List[ComplianceIssue]:
        """Check dedicated circuit requirements"""
        issues = []
        
        dedicated_circuits = {
            'kitchen': ['refrigerator', 'dishwasher', 'microwave', 'countertop (2 circuits min)'],
            'laundry': ['washer', 'dryer (240V)'],
            'bathroom': ['bathroom circuit (20A)'],
            'garage': ['garage door opener', 'EV charger ready']
        }
        
        if room_type in dedicated_circuits:
            circuits = dedicated_circuits[room_type]
            for circuit in circuits:
                issues.append(ComplianceIssue(
                    system='electrical',
                    code_reference='CEC dedicated circuit requirements',
                    severity='warning',
                    message=f'Dedicated circuit required for {circuit}',
                    location=room_data.get('centroid'),
                    suggested_fix=f'Install dedicated {circuit} circuit from panel'
                ))
        
        return issues
