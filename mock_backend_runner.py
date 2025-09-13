#!/usr/bin/env python3
"""
Mock Backend Runner

Simulates fetching the materials JSON from a backend service and pipes it to
material_search.py via stdin, producing the final estimate JSON.

Usage examples:
  python mock_backend_runner.py --output estimate_mock.json
  python mock_backend_runner.py --project-id demo-123 --simulate-latency 0.5 --max-vendors 1

Notes:
- This script "pretends" to call an HTTP backend by logging a fake GET request,
  then loads materials from the local materials_test.json file.
- It runs material_search.py using the same Python interpreter (sys.executable),
  ensuring it uses your current virtual environment.
- If you pass a relative --output path, it will be saved inside this folder
  (the material_estimate/ directory) by default.
"""
import argparse
import json
import os
import sys
import time
import subprocess
from typing import Optional


def mock_backend_fetch_materials(project_id: str, auth_token: Optional[str], source_path: str, simulate_latency: float) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_abs = source_path if os.path.isabs(source_path) else os.path.join(script_dir, source_path)

    # Fake HTTP log
    print(f"[mock-backend] GET https://mock-backend.local/api/materials?project_id={project_id}")
    if auth_token:
        print("[mock-backend] Authorization: Bearer ****** (redacted)")
    print(f"[mock-backend] Simulated 200 OK, body from {os.path.relpath(source_abs, script_dir)}")

    if simulate_latency > 0:
        time.sleep(simulate_latency)

    with open(source_abs, "r", encoding="utf-8") as f:
        body = f.read()

    # Validate JSON quickly
    try:
        json.loads(body)
    except Exception as e:
        raise SystemExit(f"[mock-backend] ERROR: Invalid JSON in {source_abs}: {e}")

    return body


def run_estimator(input_json: str, output_path: str, max_vendors: int, model: str) -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    estimator = os.path.join(script_dir, "material_search.py")

    if not os.path.exists(estimator):
        raise SystemExit(f"ERROR: material_search.py not found at {estimator}")

    cmd = [
        sys.executable,
        estimator,
        "--output", output_path,
        "--max-vendors", str(max_vendors),
        "--model", model,
    ]

    print(f"[runner] Executing: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        input=input_json.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )

    # Forward stdout/stderr from estimator for visibility
    if proc.stdout:
        print(proc.stdout.decode("utf-8"), end="")
    if proc.stderr:
        print(proc.stderr.decode("utf-8"), file=sys.stderr, end="")

    return proc.returncode


def parse_args():
    p = argparse.ArgumentParser(description="Simulate fetching materials from a backend and run the estimator.")
    p.add_argument("--project-id", default="demo-123", help="Mock project identifier to include in fake GET logs.")
    p.add_argument("--auth-token", default="demo-token", help="Mock auth token (only logged as redacted).")
    p.add_argument("--source-file", default="materials_test.json", help="Local JSON file to serve as backend response.")
    p.add_argument("--simulate-latency", type=float, default=0.4, help="Seconds to sleep to simulate network latency.")
    p.add_argument("--output", "-o", default="estimate_mock.json", help="Output JSON file path for the estimate.")
    p.add_argument("--max-vendors", type=int, default=1, help="Max vendors per item (forwarded to estimator).")
    p.add_argument("--model", default="openai/gpt-oss-20b", help="Groq model name (forwarded to estimator).")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve output path to be inside this folder when a relative path is provided
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest_abs = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)

    try:
        materials_body = mock_backend_fetch_materials(
            project_id=args.project_id,
            auth_token=args.auth_token,
            source_path=args.source_file,
            simulate_latency=args.simulate_latency,
        )
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    rc = run_estimator(
        input_json=materials_body,
        output_path=dest_abs,
        max_vendors=args.max_vendors,
        model=args.model,
    )

    if rc == 0:
        print(f"[runner] Estimate written to {dest_abs}")
    else:
        print(f"[runner] Estimator exited with code {rc}. See logs above.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
