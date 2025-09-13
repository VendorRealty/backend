#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError


def to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_decimal(v) for v in obj]
    return obj


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_contractors(args) -> Dict[str, List[str]]:
    contractors = {"plumbers": [], "electricians": [], "general_contractors": []}
    if args.contractors_json:
        try:
            data = load_json(args.contractors_json)
            for k in ("plumbers", "electricians", "general_contractors"):
                v = data.get(k)
                if isinstance(v, list):
                    contractors[k] = [str(x) for x in v if str(x).strip()]
        except Exception as e:
            raise SystemExit(f"Failed to read contractors JSON: {e}")
    # CLI overrides/extends
    if args.plumbers:
        contractors["plumbers"] = [s for s in args.plumbers if str(s).strip()]
    if args.electricians:
        contractors["electricians"] = [s for s in args.electricians if str(s).strip()]
    if args.contractors:
        contractors["general_contractors"] = [s for s in args.contractors if str(s).strip()]
    return contractors


def build_item(floorplan_id: str, materials_doc: Dict[str, Any], input_path: str, contractors: Dict[str, List[str]]):
    materials_in = materials_doc.get("materials")
    if not isinstance(materials_in, list):
        raise SystemExit("Input JSON must contain a 'materials' array (use material_search.py output).")

    # Normalize materials and select best vendor for link + price
    materials_out = []
    for m in materials_in:
        name = str(m.get("name", "")).strip()
        if not name:
            continue
        try:
            qty = float(m.get("quantity", 0))
        except Exception:
            qty = 0.0
        unit = str(m.get("unit", "")).strip() or "unit"
        vendors = m.get("vendors") or []
        bvi = m.get("best_vendor_index", 0)
        try:
            bvi = int(bvi)
        except Exception:
            bvi = 0
        sel_vendor = None
        if isinstance(vendors, list) and vendors:
            if 0 <= bvi < len(vendors):
                sel_vendor = vendors[bvi]
            else:
                sel_vendor = vendors[0]

        selected_price_per_unit = m.get("selected_price_per_unit")
        item_subtotal = m.get("item_subtotal")

        item_out = {
            "name": name,
            "quantity": qty,
            "unit": unit,
            "vendors": vendors,
        }
        # Only include optional fields when present to avoid None values in DynamoDB
        if sel_vendor:
            item_out["selected_vendor"] = sel_vendor
        if selected_price_per_unit is not None:
            item_out["selected_price_per_unit"] = selected_price_per_unit
        if item_subtotal is not None:
            item_out["item_subtotal"] = item_subtotal

        materials_out.append(item_out)

    totals = materials_doc.get("totals") or {
        "materials_subtotal": sum(float(x.get("item_subtotal", 0.0) or 0.0) for x in materials_out),
        "estimated_tax": 0.0,
        "grand_total": sum(float(x.get("item_subtotal", 0.0) or 0.0) for x in materials_out),
        "currency": materials_doc.get("metadata", {}).get("currency") or "CAD",
    }

    item = {
        "floorplan_id": str(floorplan_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "materials": materials_out,
        "totals": totals,
        "contractors": contractors,
        "source": {
            "input_json_path": os.path.abspath(input_path),
            "metadata": materials_doc.get("metadata") or {},
        },
    }
    return to_decimal(item)


def save_item(table_name: str, region: Optional[str], item: Dict[str, Any], overwrite: bool = False):
    session = boto3.session.Session(region_name=region)
    dynamodb = session.resource("dynamodb")
    table = dynamodb.Table(table_name)

    try:
        if overwrite:
            table.put_item(Item=item)
        else:
            table.put_item(Item=item, ConditionExpression="attribute_not_exists(floorplan_id)")
        print(f"Saved estimate for floorplan_id={item['floorplan_id']} to table '{table_name}'.")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            raise SystemExit(
                "An item with this floorplan_id already exists. Use --overwrite to replace it."
            )
        raise


def parse_args():
    p = argparse.ArgumentParser(description="Save priced materials estimate and contractor names to DynamoDB")
    p.add_argument("--floorplan-id", required=True, help="Unique identifier for the floorplan (partition key)")
    p.add_argument("--materials-json", required=True, help="Path to priced materials JSON (output of material_search.py)")
    p.add_argument("--table-name", default="FloorplanEstimates", help="DynamoDB table name")
    p.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"), help="AWS region (fallbacks to env)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing item if present")

    # Contractors
    p.add_argument("--contractors-json", help="Path to JSON file with {plumbers:[], electricians:[], general_contractors:[]} ")
    p.add_argument("--plumbers", nargs="*", help="Plumber names (space-separated)")
    p.add_argument("--electricians", nargs="*", help="Electrician names (space-separated)")
    p.add_argument("--contractors", nargs="*", help="General contractor names (space-separated)")

    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.materials_json):
        raise SystemExit(f"Materials JSON not found: {args.materials_json}")

    try:
        materials_doc = load_json(args.materials_json)
    except Exception as e:
        raise SystemExit(f"Failed to parse materials JSON: {e}")

    contractors = parse_contractors(args)
    item = build_item(args.floorplan_id, materials_doc, args.materials_json, contractors)
    save_item(args.table_name, args.region, item, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
