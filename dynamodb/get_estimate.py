#!/usr/bin/env python3
import argparse
import json
import os

import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key


def parse_args():
    p = argparse.ArgumentParser(description="Retrieve a floorplan estimate from DynamoDB by floorplan_id")
    p.add_argument("--floorplan-id", required=True)
    p.add_argument("--table-name", default="FloorplanEstimates")
    p.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    return p.parse_args()


def main():
    args = parse_args()
    session = boto3.session.Session(region_name=args.region)
    dynamodb = session.resource("dynamodb")
    table = dynamodb.Table(args.table_name)

    try:
        resp = table.get_item(Key={"floorplan_id": args.floorplan_id})
    except ClientError as e:
        raise SystemExit(f"DynamoDB error: {e}")

    item = resp.get("Item")
    if not item:
        raise SystemExit("No item found.")

    # Decimal -> float for pretty print
    def de_decimal(o):
        from decimal import Decimal
        if isinstance(o, Decimal):
            return float(o)
        raise TypeError

    print(json.dumps(item, indent=2, default=de_decimal))


if __name__ == "__main__":
    main()
