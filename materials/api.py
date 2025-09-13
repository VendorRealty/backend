#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException

# Reuse the existing logic from material_search
from .material_search import (
    call_groq,
    sanitize_and_parse_json,
    ensure_totals_and_metadata,
)

app = FastAPI(title="Materials Estimate API", version="0.1.0")


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/api/estimate")
async def estimate(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Accept either {"materials": [...]} or just a list of items
    materials: List[Dict[str, Any]]
    if isinstance(payload, list):
        materials = [i for i in payload if isinstance(i, dict)]
    else:
        materials = payload.get("materials") or []
        if not isinstance(materials, list):
            raise HTTPException(status_code=400, detail="Payload must be a list or an object with 'materials': [...]")

    if not materials:
        raise HTTPException(status_code=400, detail="No materials provided")

    # Optional controls
    max_vendors = int(payload.get("max_vendors", 3))
    model = str(payload.get("model", "openai/gpt-oss-20b"))
    currency = str(payload.get("currency", "CAD"))

    # Require API key presence for Groq
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("grok_api_key")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set on server")

    try:
        raw = call_groq(materials, currency=currency, max_vendors=max_vendors, model=model)
    except Exception as e:  # surfacing any upstream error
        raise HTTPException(status_code=502, detail=f"Upstream Groq error: {e}")

    try:
        doc = sanitize_and_parse_json(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse model output as JSON: {e}")

    doc = ensure_totals_and_metadata(doc, currency=currency)
    return doc
