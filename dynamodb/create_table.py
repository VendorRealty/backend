#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError


def ensure_table(table_name: str, region: Optional[str] = None):
    session = boto3.session.Session(region_name=region)
    dynamodb = session.resource("dynamodb")
    client = dynamodb.meta.client

    # Check if table exists
    try:
        client.describe_table(TableName=table_name)
        print(f"Table '{table_name}' already exists.")
        return dynamodb.Table(table_name)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise

    # Create table (on-demand billing)
    print(f"Creating table '{table_name}' ...")
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "floorplan_id", "KeyType": "HASH"},  # partition key
        ],
        AttributeDefinitions=[
            {"AttributeName": "floorplan_id", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
        Tags=[
            {"Key": "Project", "Value": "RenderRealty"},
            {"Key": "Component", "Value": "FloorplanEstimates"},
        ],
    )

    # Wait until created
    waiter = client.get_waiter("table_exists")
    waiter.wait(TableName=table_name)
    print(f"Table '{table_name}' created.")
    return table


def parse_args():
    p = argparse.ArgumentParser(description="Create DynamoDB table for floorplan estimates (idempotent)")
    p.add_argument("--table-name", default="FloorplanEstimates", help="DynamoDB table name (default: FloorplanEstimates)")
    p.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"), help="AWS region (fallbacks to env)")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_table(args.table_name, region=args.region)


if __name__ == "__main__":
    main()
