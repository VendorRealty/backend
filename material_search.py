#!/usr/bin/env python3
import argparse
import os
import sys
import re
import time
from datetime import datetime, timezone

# Fast JSON if available, otherwise fallback
try:
    import orjson as _json
    def jloads(s: str):
        return _json.loads(s)
    def jdumps(obj) -> str:
        return _json.dumps(obj, option=_json.OPT_INDENT_2).decode("utf-8")
except Exception:
    import json as _json
    def jloads(s: str):
        return _json.loads(s)
    def jdumps(obj) -> str:
        return _json.dumps(obj, indent=2, ensure_ascii=False)

# Load .env if present
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    # dotenv is optional; we can still proceed using OS env
    pass

# Groq client
try:
    from groq import Groq, RateLimitError
except Exception as e:
    print("ERROR: Missing dependency 'groq'. Install with: pip install groq python-dotenv", file=sys.stderr)
    sys.exit(1)


def _get_api_key() -> str:
    # Accept various env var names for convenience
    candidates = [
        "GROQ_API_KEY",
        "groq_api_key",
        "GROQ_api_key",
        "GROQKEY",
        "grok_api_key",  # observed in your .env
    ]
    for name in candidates:
        val = os.environ.get(name)
        if val:
            # normalize for Groq SDK
            os.environ["GROQ_API_KEY"] = val
            return val
    return ""


def _read_stdin() -> str:
    if sys.stdin.isatty():
        return ""
    try:
        return sys.stdin.read()
    except Exception:
        return ""


def load_materials(input_path: str | None):
    raw = ""
    if input_path:
        with open(input_path, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = _read_stdin()
        if not raw:
            print("ERROR: No input provided. Pass --input PATH or pipe JSON via stdin.", file=sys.stderr)
            sys.exit(2)
    try:
        data = jloads(raw)
    except Exception as e:
        print(f"ERROR: Failed to parse input JSON: {e}", file=sys.stderr)
        sys.exit(2)

    # Accept either a list of materials or {"materials":[...]}
    if isinstance(data, dict) and "materials" in data and isinstance(data["materials"], list):
        materials = data["materials"]
    elif isinstance(data, list):
        materials = data
    else:
        print("ERROR: Input must be a list of items or an object with 'materials': [...]", file=sys.stderr)
        sys.exit(2)

    # Basic normalization
    normalized = []
    for item in materials:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            qty = float(item.get("quantity", 1))
        except Exception:
            qty = 1.0
        unit = str(item.get("unit", "")).strip() or "unit"
        entry = {
            "name": name,
            "quantity": qty,
            "unit": unit,
        }
        # carry-through optional fields if present
        for k in ("spec", "preferred_brands", "vendor_preferences"):
            if k in item:
                entry[k] = item[k]
        normalized.append(entry)

    if not normalized:
        print("ERROR: No valid material items were found.", file=sys.stderr)
        sys.exit(2)

    return normalized


def build_system_prompt(max_vendors: int, currency: str) -> str:
    return (
        "You are a procurement assistant. Use the browser_search tool to find CURRENT retail prices in Canada "
        f"for each material, prioritizing Canadian vendors (e.g., Home Depot Canada, Lowe's Canada, RONA, Home Hardware, Canadian Tire). "
        f"Prices and totals must be in {currency}. For EACH item:\n"
        f"- Provide up to {max_vendors} vendors with fields: vendor_name, price_per_unit (number), unit, url, availability, notes.\n"
        "- Select best_vendor_index based on availability, spec match, and price.\n"
        "- Compute selected_price_per_unit and item_subtotal = quantity * selected_price_per_unit.\n"
        "Return STRICT JSON ONLY, no markdown/code fences, matching this schema:\n"
        "{\n"
        '  "materials": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "quantity": number,\n'
        '      "unit": "string",\n'
        '      "vendors": [\n'
        "        {\n"
        '          "vendor_name": "string",\n'
        '          "price_per_unit": number,\n'
        '          "unit": "string",\n'
        '          "url": "string",\n'
        '          "availability": "in_stock|backorder|unknown",\n'
        '          "notes": "string"\n'
        "        }\n"
        "      ],\n"
        '      "best_vendor_index": number,\n'
        '      "selected_price_per_unit": number,\n'
        '      "item_subtotal": number\n'
        "    }\n"
        "  ],\n"
        '  "totals": {\n'
        '    "materials_subtotal": number,\n'
        '    "estimated_tax": number,\n'
        '    "grand_total": number,\n'
        f'    "currency": "{currency}"\n'
        "  },\n"
        '  "metadata": {\n'
        '    "model": "openai/gpt-oss-20b",\n'
        '    "search_tool": "browser_search",\n'
        '    "zip": "",\n'
        "    \"tax_rate\": 0.0,\n"
        '    "generated_at": "ISO-8601",\n'
        '    "assumptions": "No tax included; CAD currency; Canadian vendors prioritized."\n'
        "  }\n"
        "}\n"
        "Output only valid JSON."
    )


def build_user_prompt(materials: list[dict], currency: str) -> str:
    # Provide instructions and attach input JSON
    import json as json_std
    materials_json = json_std.dumps({"materials": materials}, ensure_ascii=False)
    return (
        "Find vendors and prices for the following materials. Use only CAD pricing and prefer Canadian vendors. "
        "Include exact product links. Output must be STRICT JSON per schema. Do not include any text outside of the JSON.\n\n"
        f"Currency: {currency}\n"
        "Tax: 0.0 (no tax)\n"
        "ZIP/Postal: not provided; do not localize to a specific postal code.\n\n"
        f"Materials JSON:\n{materials_json}"
    )


def call_groq(materials: list[dict], currency: str, max_vendors: int, model: str) -> str:
    api_key = _get_api_key()
    if not api_key:
        print("ERROR: GROQ_API_KEY not set (supports GROQ_API_KEY or grok_api_key in .env).", file=sys.stderr)
        sys.exit(3)

    client = Groq()

    messages = [
        {"role": "system", "content": build_system_prompt(max_vendors=max_vendors, currency=currency)},
        {"role": "user", "content": build_user_prompt(materials, currency=currency)},
    ]

    # Use tool-assisted browsing
    resp = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.0,
        max_completion_tokens=2048,
        top_p=1,
        stream=False,
        stop=None,
        tool_choice="required",
        tools=[{"type": "browser_search"}],
    )

    try:
        content = resp.choices[0].message.content
    except Exception:
        content = None

    if not content:
        # If no content is present, return an explicit error stub
        return jdumps({
            "error": "No content returned from Groq completion.",
            "metadata": {
                "model": model,
                "search_tool": "browser_search",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    return content


def sanitize_and_parse_json(text: str):
    # Remove common code fences if any slipped in
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE)

    # Try direct parse
    try:
        return jloads(cleaned)
    except Exception:
        pass

    # Fallback: extract the largest {...} block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start:end + 1]
        try:
            return jloads(snippet)
        except Exception:
            pass

    raise ValueError("Model did not return valid JSON.")


def parse_retry_after_seconds(msg: str) -> float | None:
    """Extract retry-after seconds from Groq rate limit error message if present.

    Example pattern: "Please try again in 7m29.257s"
    Returns total seconds as float if parsed, otherwise None.
    """
    try:
        m = re.search(r"Please try again in\s+(?:(\d+)m)?([0-9]+(?:\.[0-9]+)?)s", msg)
        if not m:
            return None
        minutes = int(m.group(1)) if m.group(1) else 0
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    except Exception:
        return None


def ensure_totals_and_metadata(doc: dict, currency: str):
    # Ensure materials array
    materials = doc.get("materials")
    if not isinstance(materials, list):
        doc["materials"] = []
        materials = doc["materials"]

    # Compute per-item subtotal if missing
    for item in materials:
        try:
            qty = float(item.get("quantity", 0))
        except Exception:
            qty = 0.0

        sel_price = item.get("selected_price_per_unit")
        if sel_price is None:
            # try derive from best vendor
            vendors = item.get("vendors") or []
            bvi = item.get("best_vendor_index", 0)
            try:
                sel_price = float(vendors[bvi].get("price_per_unit"))
            except Exception:
                sel_price = 0.0
            item["selected_price_per_unit"] = sel_price

        try:
            sel_price = float(sel_price)
        except Exception:
            sel_price = 0.0
            item["selected_price_per_unit"] = 0.0

        item_subtotal = qty * sel_price
        item["item_subtotal"] = round(float(item_subtotal), 2)

    # Compute totals (no tax; CAD)
    materials_subtotal = round(sum(float(i.get("item_subtotal", 0.0)) for i in materials), 2)
    totals = {
        "materials_subtotal": materials_subtotal,
        "estimated_tax": 0.0,
        "grand_total": materials_subtotal,
        "currency": currency,
    }
    doc["totals"] = totals

    # Metadata
    md = doc.get("metadata") or {}
    md.update({
        "model": "openai/gpt-oss-20b",
        "search_tool": "browser_search",
        "zip": "",
        "tax_rate": 0.0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assumptions": "No tax included; CAD currency; Canadian vendors prioritized."
    })
    doc["metadata"] = md

    return doc


def write_output(doc: dict, output_path: str):
    # Ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(jdumps(doc))


def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch vendor pricing via Groq browser_search and produce a CAD estimate JSON (no tax)."
    )
    p.add_argument("--input", "-i", help="Path to input materials JSON. If omitted, reads from stdin.")
    p.add_argument("--output", "-o", required=True, help="Path to write the output JSON (required).")
    p.add_argument("--max-vendors", type=int, default=3, help="Max vendors per item (default: 3).")
    p.add_argument("--model", default="openai/gpt-oss-20b", help="Groq model to use.")
    return p.parse_args()


def main():
    args = parse_args()
    currency = "CAD"  # per user instruction
    materials = load_materials(args.input)

    try:
        raw = call_groq(materials, currency=currency, max_vendors=args.max_vendors, model=args.model)
    except RateLimitError as e:
        retry_after = parse_retry_after_seconds(str(e))
        err = {
            "error": "rate_limit_exceeded",
            "message": str(e),
            "retry_after_seconds": retry_after,
            "suggestions": [
                "Wait the indicated time and re-run",
                "Reduce input size or --max-vendors to lower token usage",
                "Try a different model via --model to use a separate quota",
            ],
            "materials": materials,
            "totals": {
                "materials_subtotal": 0.0,
                "estimated_tax": 0.0,
                "grand_total": 0.0,
                "currency": currency,
            },
            "metadata": {
                "model": args.model,
                "search_tool": "browser_search",
                "zip": "",
                "tax_rate": 0.0,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "assumptions": "No tax included; CAD currency; Canadian vendors prioritized.",
            },
        }
        write_output(err, args.output)
        print(f"Wrote rate-limit error JSON to {args.output}", file=sys.stderr)
        sys.exit(5)
    except Exception as e:
        err = {
            "error": "groq_request_failed",
            "message": str(e),
            "materials": materials,
            "totals": {
                "materials_subtotal": 0.0,
                "estimated_tax": 0.0,
                "grand_total": 0.0,
                "currency": currency,
            },
            "metadata": {
                "model": args.model,
                "search_tool": "browser_search",
                "zip": "",
                "tax_rate": 0.0,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "assumptions": "No tax included; CAD currency; Canadian vendors prioritized.",
            },
        }
        write_output(err, args.output)
        print(f"Wrote error JSON to {args.output}", file=sys.stderr)
        sys.exit(5)

    try:
        doc = sanitize_and_parse_json(raw)
    except Exception as e:
        # Produce an error JSON file for visibility
        err = {
            "error": f"Failed to parse Groq response as JSON: {e}",
            "raw_response_sample": raw[:1000],
            "materials": materials,
            "totals": {
                "materials_subtotal": 0.0,
                "estimated_tax": 0.0,
                "grand_total": 0.0,
                "currency": currency
            },
            "metadata": {
                "model": args.model,
                "search_tool": "browser_search",
                "zip": "",
                "tax_rate": 0.0,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "assumptions": "No tax included; CAD currency; Canadian vendors prioritized."
            }
        }
        write_output(err, args.output)
        print(f"Wrote error JSON to {args.output}", file=sys.stderr)
        sys.exit(4)

    doc = ensure_totals_and_metadata(doc, currency=currency)
    write_output(doc, args.output)
    print(f"Wrote estimate JSON to {args.output}")


if __name__ == "__main__":
    main()