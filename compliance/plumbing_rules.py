"""
Plumbing Compliance Rules based on Ontario Building Code Part 7
"""

from typing import Dict, List
from .compliance_engine import ComplianceIssue

class PlumbingComplianceChecker:
    """Detailed plumbing compliance checking based on OBC Part 7"""
    
    @staticmethod
    def check_fixture_requirements(room_type: str, room_data: Dict) -> List[ComplianceIssue]:
        """Check minimum fixture requirements"""
        issues = []
        
        fixture_requirements = {
            'bathroom': {
                'water_closet': 1,
                'lavatory': 1,
                'bathtub_or_shower': 1
            },
            'powder_room': {
                'water_closet': 1,
                'lavatory': 1
            },
            'kitchen': {
                'kitchen_sink': 1,
                'dishwasher_connection': 1
            },
            'laundry': {
                'laundry_tub': 1,
                'washer_connection': 1
            }
        }
        
        if room_type in fixture_requirements:
            fixtures = fixture_requirements[room_type]
            for fixture, count in fixtures.items():
                issues.append(ComplianceIssue(
                    system='plumbing',
                    code_reference='OBC 7.1.3.1',
                    severity='info',
                    message=f'{room_type.capitalize()} requires {count} {fixture.replace("_", " ")}',
                    location=room_data.get('centroid'),
                    suggested_fix=f'Install {fixture.replace("_", " ")} with proper supply and drainage'
                ))
        
        return issues
    
    @staticmethod
    def check_water_temperature_limits(room_type: str, room_data: Dict) -> List[ComplianceIssue]:
        """Check hot water temperature limits"""
        issues = []
        
        if room_type in ['bathroom', 'powder_room']:
            issues.append(ComplianceIssue(
                system='plumbing',
                code_reference='OBC 7.2.10.7',
                severity='error',
                message='Hot water temperature must not exceed 49°C (120°F)',
                location=room_data.get('centroid'),
                suggested_fix='Install thermostatic mixing valve at water heater or point of use'
            ))
        
        return issues
    
    @staticmethod
    def check_venting_requirements(room_type: str, room_data: Dict) -> List[ComplianceIssue]:
        """Check drain venting requirements"""
        issues = []
        
        if room_type in ['bathroom', 'kitchen', 'laundry']:
            issues.append(ComplianceIssue(
                system='plumbing',
                code_reference='OBC 7.9.3',
                severity='error',
                message='All fixtures require proper venting to prevent trap siphonage',
                location=room_data.get('centroid'),
                suggested_fix='Install individual or common vent within 6 feet of trap'
            ))
        
        return issues
    
    @staticmethod
    def check_backflow_prevention(room_type: str, room_data: Dict) -> List[ComplianceIssue]:
        """Check backflow prevention requirements"""
        issues = []
        
        backflow_required = {
            'bathroom': 'Handheld shower requires vacuum breaker',
            'kitchen': 'Dishwasher requires air gap or high loop',
            'laundry': 'Laundry tub faucet with hose thread requires vacuum breaker',
            'mechanical': 'Boiler fill requires reduced pressure backflow preventer'
        }
        
        if room_type in backflow_required:
            issues.append(ComplianceIssue(
                system='plumbing',
                code_reference='OBC 7.2.11',
                severity='warning',
                message=backflow_required[room_type],
                location=room_data.get('centroid'),
                suggested_fix='Install appropriate backflow prevention device'
            ))
        
        return issues
    
    @staticmethod
    def check_pipe_sizing(fixture_units: int) -> List[ComplianceIssue]:
        """Check water supply pipe sizing based on fixture units"""
        issues = []
        
        # Simplified pipe sizing based on fixture units
        if fixture_units <= 3:
            pipe_size = '1/2"'
        elif fixture_units <= 7:
            pipe_size = '3/4"'
        elif fixture_units <= 20:
            pipe_size = '1"'
        else:
            pipe_size = '1-1/4" or larger'
        
        issues.append(ComplianceIssue(
            system='plumbing',
            code_reference='OBC 7.2.4',
            severity='info',
            message=f'Main supply line requires minimum {pipe_size} pipe for {fixture_units} fixture units',
            suggested_fix=f'Size supply piping at {pipe_size} minimum'
        ))
        
        return issues
