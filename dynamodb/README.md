# DynamoDB: Floorplan Estimates Store

This module provides scripts to create and use a DynamoDB table that, for a given floorplan, stores:

- Materials needed
- Cost of the materials (per-item and totals)
- Product links for each material (selected vendor and all vendor options)
- Names of plumbers, electricians, and general contractors involved

## Table schema

- Table name: `FloorplanEstimates` (configurable)
- Primary key: `floorplan_id` (string)
- Example item shape:

```json
{
  "floorplan_id": "plan-123",
  "created_at": "2025-09-13T15:20:31Z",
  "materials": [
    {
      "name": "OSB sheathing 7/16 in 4x8",
      "quantity": 80,
      "unit": "sheet",
      "selected_vendor": {
        "vendor_name": "Home Depot Canada",
        "price_per_unit": 19.97,
        "unit": "sheet",
        "url": "https://www.homedepot.ca/...",
        "availability": "in_stock"
      },
      "selected_price_per_unit": 19.97,
      "item_subtotal": 1597.6,
      "vendors": [
        { "vendor_name": "Home Depot Canada", "price_per_unit": 19.97, "unit": "sheet", "url": "...", "availability": "in_stock" },
        { "vendor_name": "RONA", "price_per_unit": 21.50, "unit": "sheet", "url": "...", "availability": "in_stock" }
      ]
    }
  ],
  "totals": {
    "materials_subtotal": 3500.25,
    "estimated_tax": 0.0,
    "grand_total": 3500.25,
    "currency": "CAD"
  },
  "contractors": {
    "plumbers": ["ABC Plumbing"],
    "electricians": ["XYZ Electric"],
    "general_contractors": ["Delta Contracting Inc."]
  },
  "source": {
    "input_json_path": "backend/materials/priced.json",
    "metadata": { "model": "openai/gpt-oss-20b" }
  }
}
```

Note: Numbers are stored using DynamoDB Number type (Decimal). The scripts convert Python floats to Decimal to satisfy DynamoDB's requirements.

## Install dependencies

```bash
pip install -r backend/dynamodb/requirements.txt
```

You must have AWS credentials configured (e.g., via `aws configure`, environment variables, or an assumed role) and a default region (or pass `--region`).

## Create the table

```bash
python backend/dynamodb/create_table.py --table-name FloorplanEstimates --region ca-central-1
```

This is idempotent; it will create the table if it doesn't exist.

## Generate a priced estimate JSON

Typical flow using the existing materials tools:

1) From a floorplan image to materials list (detailed):

```bash
python backend/materials/materials_needed.py --from-floorplan --floorplan backend/floorplans/floorplan.png --detailed --print-only > /tmp/materials.json
```

2) Fetch vendor prices and totals (CAD) using Groq:

```bash
python backend/materials/material_search.py -i /tmp/materials.json -o /tmp/priced.json
```

## Save to DynamoDB

Provide a floorplan ID and optional contractor names:

```bash
python backend/dynamodb/save_estimate.py \
  --floorplan-id plan-123 \
  --materials-json /tmp/priced.json \
  --plumbers "ABC Plumbing" "BestFlow Ltd." \
  --electricians "XYZ Electric" \
  --contractors "Delta Contracting Inc." \
  --table-name FloorplanEstimates \
  --region ca-central-1
```

Alternatively, provide a contractors JSON file:

```json
{
  "plumbers": ["ABC Plumbing"],
  "electricians": ["XYZ Electric"],
  "general_contractors": ["Delta Contracting Inc."]
}
```

```bash
python backend/dynamodb/save_estimate.py \
  --floorplan-id plan-123 \
  --materials-json /tmp/priced.json \
  --contractors-json backend/dynamodb/sample_contractors.json \
  --table-name FloorplanEstimates
```

## Retrieve an estimate

```bash
python backend/dynamodb/get_estimate.py --floorplan-id plan-123 --table-name FloorplanEstimates --region ca-central-1
```

This prints the saved item as JSON.
