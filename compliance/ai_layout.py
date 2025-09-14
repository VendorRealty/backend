import os
import json
from typing import Dict, List, Optional
import requests

CEREBRAS_API_URL = os.environ.get("CEREBRAS_API_URL", "https://api.cerebras.ai/v1/chat/completions")
CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")

def refine_layout_with_cerebras(api_key: str,
                                rooms: List[Dict],
                                devices: Dict,
                                compliance_issues: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Ask Cerebras to propose non-overlapping light coordinates per room, given bbox/contours and compliance context.
    RAG-like: we build a compact context payload: rooms (bbox/centroid/contour sample), detected devices, and top compliance messages.
    Returns {'rooms': {room_id: {'lights': [[x,y], ...], 'wires': [[[x,y],...], ...]}}} or None on failure.
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        minimal_rooms = []
        for r in rooms:
            bbox = r.get('bounding_box', {})
            contour = r.get('contour', [])
            minimal_rooms.append({
                'id': r.get('id'),
                'bbox': bbox,
                'centroid': r.get('centroid', [0,0]),
                'contour': contour[:200],  # cap for size
            })
        # Compress devices to essentials
        dev_summary = {
            'outlets': [(d.get('position', [0,0])[0], d.get('position', [0,0])[1]) for d in (devices.get('outlets') or [])][:200],
            'switches': [(d.get('position', [0,0])[0], d.get('position', [0,0])[1]) for d in (devices.get('switches') or [])][:200],
            'lights': [(d.get('position', [0,0])[0], d.get('position', [0,0])[1]) for d in (devices.get('lights') or [])][:200],
        }
        # Pull top compliance messages (first 20)
        comp_msgs = []
        if compliance_issues:
            for c in compliance_issues[:20]:
                try:
                    comp_msgs.append(getattr(c, 'message', None) or c.get('message'))
                except Exception:
                    continue
        instruction = (
            "You are arranging ceiling lights and proposing wire paths for a residential plan.\n"
            "Requirements:\n"
            "- Place lights near room incenter (equidistant from walls) and distribute by area (~1 per 120 sq ft).\n"
            "- Keep coordinates inside the room bbox; clip to bbox if necessary.\n"
            "- Provide optional dashed wiring polylines per room connecting lights (no touching walls).\n"
            "- Do not change outlet/switch coordinates; focus only on lights and wires.\n"
            "Return ONLY JSON (no prose) with schema: {\"rooms\":{\"<id>\":{\"lights\":[[x,y],...],\"wires\":[[[x,y],...],...]}}}."
        )
        ctx = {
            "rooms": minimal_rooms,
            "devices": dev_summary,
            "compliance": comp_msgs,
        }
        prompt = {"role": "user", "content": instruction + "\nCTX=" + json.dumps(ctx, separators=(",", ":"))}
        body = {
            "model": CEREBRAS_MODEL,
            "messages": [prompt],
            "temperature": 0.2,
            "max_tokens": 1200,
        }
        resp = requests.post(CEREBRAS_API_URL, headers=headers, json=body, timeout=25)
        if resp.status_code != 200:
            try:
                err_text = resp.text[:500]
                print(f"[CerebrasAPI] status={resp.status_code} body={err_text}")
            except Exception:
                print(f"[CerebrasAPI] status={resp.status_code} (no body)")
            return None
        try:
            data = resp.json()
        except Exception as e:
            print(f"[CerebrasAPI] JSON decode error: {e} :: {resp.text[:500]}")
            return None
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            return None
        try:
            # Some models may include extra text. Extract JSON substring if needed.
            s = content
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                s = s[start:end+1]
            refined = json.loads(s)
        except Exception as e:
            print(f"[CerebrasAPI] inner JSON parse error: {e} :: {content[:200]}")
            return None
        if not isinstance(refined, dict) or 'rooms' not in refined:
            return None
        return refined
    except Exception:
        return None
