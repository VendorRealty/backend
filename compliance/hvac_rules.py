"""
HVAC Compliance Rules for Ontario Building Code
"""

from typing import Dict, List
from .compliance_engine import ComplianceIssue

class HVACComplianceChecker:
    """Detailed HVAC compliance checking based on Ontario Building Code"""
    
    @staticmethod
    def check_ventilation_requirements(room_data: Dict) -> List[ComplianceIssue]:
        """Check ventilation requirements per OBC Part 6"""
        issues = []
        area_sqft = room_data.get('area_pixels', 0) * 0.01
        
        # Minimum ventilation rates based on room type
        if area_sqft > 0:
            # Residential: 0.35 ACH or 15 CFM per person (whichever is greater)
            min_cfm = max(15, int(area_sqft * 8.5 * 0.35 / 60))
            
            issues.append(ComplianceIssue(
                system='hvac',
                code_reference='OBC 6.3.1.1',
                severity='info',
                message=f'Minimum ventilation required: {min_cfm} CFM',
                location=room_data.get('centroid'),
                suggested_fix=f'Provide mechanical or natural ventilation of {min_cfm} CFM'
            ))
        
        return issues
    
    @staticmethod
    def check_exhaust_requirements(room_type: str, room_data: Dict) -> List[ComplianceIssue]:
        """Check exhaust fan requirements for specific rooms"""
        issues = []
        
        exhaust_requirements = {
            'bathroom': 50,  # CFM
            'kitchen': 100,
            'laundry': 50
        }
        
        if room_type in exhaust_requirements:
            required_cfm = exhaust_requirements[room_type]
            issues.append(ComplianceIssue(
                system='hvac',
                code_reference='OBC 6.3.3.3',
                severity='error',
                message=f'{room_type.capitalize()} requires {required_cfm} CFM exhaust',
                location=room_data.get('centroid'),
                suggested_fix=f'Install exhaust fan with minimum {required_cfm} CFM capacity'
            ))
        
        return issues
    
    @staticmethod
    def check_heating_requirements(room_data: Dict) -> List[ComplianceIssue]:
        """Check heating requirements per OBC"""
        issues = []
        area_sqft = room_data.get('area_pixels', 0) * 0.01
        
        if area_sqft > 0:
            # Approximate heat load: 30-40 BTU/sq ft for Ontario climate
            heat_load_btu = int(area_sqft * 35)
            
            issues.append(ComplianceIssue(
                system='hvac',
                code_reference='OBC 6.2.1.1',
                severity='info',
                message=f'Estimated heating load: {heat_load_btu} BTU/hr',
                location=room_data.get('centroid'),
                suggested_fix='Size heating equipment based on proper heat load calculation'
            ))
        
        return issues
