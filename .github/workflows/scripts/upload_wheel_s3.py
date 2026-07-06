#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upload wheels to S3 and manage staging lifecycle.

Production upload (PEP 503 index):
    python upload_wheel_s3.py --bucket BUCKET --package vllm --wheel-dir dist/

Staging upload (per-run, for testing PR builds):
    python upload_wheel_s3.py --bucket BUCKET --package vllm --wheel-dir dist/ \
        --staging-run-id 28383971650

Staging cleanup (delete staging runs older than N days):
    python upload_wheel_s3.py --bucket BUCKET --cleanup-staging-days 31

Staging layout:
    s3://BUCKET/staging/<run-id>/<wheel-filename>.whl

The staging prefix is separate from the PEP 503 simple/ index so
staging wheels are never pip-installable from the main index.
The S3 bucket's CloudFront function auto-generates index pages for
any prefix, so staging works with --find-links out of the box.
"""

import argparse
import glob
import os
from datetime import datetime, timedelta, timezone

import boto3


def upload_wheels(
    s3, bucket: str, package: str, wheel_dir: str, staging_run_id: str = None
) -> None:
    wheel_names = []
    for whl in glob.glob(os.path.join(wheel_dir, "*.whl")):
        name = os.path.basename(whl)
        if staging_run_id:
            key = f"staging/{staging_run_id}/{name}"
        else:
            key = f"simple/{package}/{name}"
        print(f"Uploading {key}")
        s3.upload_file(
            whl,
            bucket,
            key,
            ExtraArgs={"ContentType": "application/zip"},
        )
        wheel_names.append(name)

    if staging_run_id and wheel_names:
        print(f"Staging run: {staging_run_id}")
        print(f"Wheel(s): {', '.join(wheel_names)}")


def cleanup_staging(s3, bucket: str, max_age_days: int) -> None:
    """Delete staging/ prefixes older than max_age_days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix="staging/")

    # Group objects by run-id prefix
    runs: dict[str, list[dict]] = {}
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parts = key.split("/")
            if len(parts) >= 3 and parts[0] == "staging":
                run_id = parts[1]
                runs.setdefault(run_id, []).append(obj)

    deleted = 0
    for run_id, objects in runs.items():
        newest = max(obj["LastModified"] for obj in objects)
        if newest < cutoff:
            keys = [{"Key": obj["Key"]} for obj in objects]
            print(
                f"Deleting staging/{run_id}/ ({len(keys)} files, "
                f"newest {newest.strftime('%Y-%m-%d')})"
            )
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            deleted += len(keys)

    print(
        f"Staging cleanup: deleted {deleted} objects "
        f"from runs older than {max_age_days} days"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--package",
        help="PEP 503 normalized package name (e.g. vllm, flash-attn)",
    )
    parser.add_argument(
        "--wheel-dir",
        help="Local directory containing .whl files",
    )
    parser.add_argument(
        "--staging-run-id",
        help="Upload to staging/<run-id>/ instead of simple/<package>/",
    )
    parser.add_argument(
        "--cleanup-staging-days",
        type=int,
        help="Delete staging runs older than this many days",
    )
    args = parser.parse_args()

    if not args.wheel_dir and args.cleanup_staging_days is None:
        parser.error("--wheel-dir or --cleanup-staging-days is required")
    if args.wheel_dir and not args.package:
        parser.error("--package is required when uploading wheels")

    s3 = boto3.client("s3")

    if args.cleanup_staging_days is not None:
        cleanup_staging(s3, args.bucket, args.cleanup_staging_days)

    if args.wheel_dir:
        upload_wheels(
            s3,
            args.bucket,
            args.package,
            args.wheel_dir,
            staging_run_id=args.staging_run_id,
        )


if __name__ == "__main__":
    main()
