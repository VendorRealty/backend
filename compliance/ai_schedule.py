import os
import json
from typing import Dict, List, Optional
import requests

CEREBRAS_API_URL = os.environ.get("CEREBRAS_API_URL", "https://api.cerebras.ai/v1/chat/completions")
CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "qwen-3-235b-a22b-thinking-2507")

def refine_schedule_with_cerebras(api_key: str,
                                  rooms: List[Dict],
                                  room_assign: Dict[int, Dict[str, List[Dict]]],
                                  circuits: List[Dict],
                                  room_to_circuit: Dict[int, int]) -> Optional[Dict]:
    """Call Cerebras Chat Completions API to refine circuits.
    Returns dict with possibly updated 'circuits' and 'room_to_circuit'.
    If the call fails or response invalid, return None.
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Build a concise prompt
        prompt = {
            "role": "user",
            "content": (
                "You are an electrical designer. Group residential rooms into reasonable 15A general receptacle circuits,\n"
                "combining small rooms where appropriate. Avoid one circuit per room unless necessary.\n"
                "Use 12 outlets per circuit max when possible. Keep lighting on 1-2 circuits.\n"
                "Return JSON with keys: circuits (list of {circuit:int, desc:str, breaker:int}),\n"
                "room_to_circuit (map room_id->circuit for receptacles).\n"
                f"Rooms: {json.dumps([{'id': r.get('id'), 'name': r.get('name', f'Room {r.get('id')}')} for r in rooms])}\n"
                f"RoomAssignments: {json.dumps({rid: {'outlets': len(v.get('outlets', [])), 'lights': len(v.get('lights', []))} for rid, v in room_assign.items()})}\n"
                f"CurrentCircuits: {json.dumps(circuits)}\n"
            ),
        }
        body = {
            "model": CEREBRAS_MODEL,
            "messages": [prompt],
            "temperature": 0.2,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(CEREBRAS_API_URL, headers=headers, json=body, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Extract text
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            return None
        try:
            refined = json.loads(content)
        except Exception:
            return None
        if not isinstance(refined, dict):
            return None
        # Basic sanity check
        if "circuits" not in refined:
            return None
        return refined
    except Exception:
        return None
