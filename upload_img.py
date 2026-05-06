#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Upload a single file or a folder to Cloudflare R2 via S3-compatible API.

Install:
    pip install boto3

Required:
    R2 endpoint URL:
        https://<ACCOUNT_ID>.r2.cloudflarestorage.com

Recommended env vars:
    R2_ACCESS_KEY_ID=xxxx
    R2_SECRET_ACCESS_KEY=xxxx
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from pathlib import Path
from urllib.parse import quote

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


def normalize_key(key: str) -> str:
    """
    Normalize R2/S3 object key.
    Example:
        /uploads/test.txt -> uploads/test.txt
        uploads\\test.txt -> uploads/test.txt
    """
    return key.replace("\\", "/").strip("/")


def build_public_url(public_base_url: str | None, key: str) -> str | None:
    if not public_base_url:
        return None
    return public_base_url.rstrip("/") + "/" + quote(key, safe="/~")


def create_r2_client(endpoint_url: str, access_key_id: str, secret_access_key: str):
    return boto3.client(
        service_name="s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def guess_extra_args(file_path: Path, cache_control: str | None = None) -> dict:
    extra_args = {}

    content_type, _ = mimetypes.guess_type(str(file_path))
    if content_type:
        extra_args["ContentType"] = content_type

    if cache_control:
        extra_args["CacheControl"] = cache_control

    return extra_args


def upload_one_file(
    s3,
    bucket: str,
    local_file: Path,
    object_key: str,
    public_base_url: str | None = None,
    cache_control: str | None = None,
    dry_run: bool = False,
):
    object_key = normalize_key(object_key)
    extra_args = guess_extra_args(local_file, cache_control)

    if dry_run:
        print(f"[DRY RUN] {local_file} -> r2://{bucket}/{object_key}")
    else:
        s3.upload_file(
            Filename=str(local_file),
            Bucket=bucket,
            Key=object_key,
            ExtraArgs=extra_args,
        )
        print(f"[OK] {local_file} -> r2://{bucket}/{object_key}")

    public_url = build_public_url(public_base_url, object_key)
    if public_url:
        print(f"     URL: {public_url}")


def upload_path(
    s3,
    bucket: str,
    local_path: Path,
    remote_key: str,
    public_base_url: str | None = None,
    cache_control: str | None = None,
    dry_run: bool = False,
):
    local_path = local_path.expanduser().resolve()

    if not local_path.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    remote_key = remote_key.replace("\\", "/")

    # Single file mode
    if local_path.is_file():
        # If remote_key ends with "/", keep original filename under that folder
        if remote_key.endswith("/"):
            object_key = normalize_key(remote_key + local_path.name)
        else:
            # Otherwise remote_key is treated as the exact remote file name
            object_key = normalize_key(remote_key) or local_path.name

        upload_one_file(
            s3=s3,
            bucket=bucket,
            local_file=local_path,
            object_key=object_key,
            public_base_url=public_base_url,
            cache_control=cache_control,
            dry_run=dry_run,
        )
        return

    # Folder mode: upload folder contents recursively
    if local_path.is_dir():
        prefix = normalize_key(remote_key)

        files = [p for p in sorted(local_path.rglob("*")) if p.is_file()]
        if not files:
            print(f"No files found in folder: {local_path}")
            return

        for file_path in files:
            rel_path = file_path.relative_to(local_path).as_posix()
            object_key = f"{prefix}/{rel_path}" if prefix else rel_path

            upload_one_file(
                s3=s3,
                bucket=bucket,
                local_file=file_path,
                object_key=object_key,
                public_base_url=public_base_url,
                cache_control=cache_control,
                dry_run=dry_run,
            )
        return

    raise ValueError(f"Unsupported local path type: {local_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a file or folder to Cloudflare R2."
    )

    parser.add_argument(
        "--endpoint-url",
        required=True,
        help="R2 S3 endpoint, e.g. https://<ACCOUNT_ID>.r2.cloudflarestorage.com",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="R2 bucket name.",
    )
    parser.add_argument(
        "--local-path",
        required=True,
        help="Local file or folder path.",
    )
    parser.add_argument(
        "--remote-key",
        required=True,
        help=(
            "Destination object key or prefix. "
            "For a file: uploads/a.jpg means exact filename; uploads/ means keep original filename. "
            "For a folder: uploads/ means upload folder contents under uploads/."
        ),
    )
    parser.add_argument(
        "--access-key-id",
        default=os.getenv("R2_ACCESS_KEY_ID"),
        help="R2 Access Key ID. Can also use env var R2_ACCESS_KEY_ID.",
    )
    parser.add_argument(
        "--secret-access-key",
        default=os.getenv("R2_SECRET_ACCESS_KEY"),
        help="R2 Secret Access Key. Can also use env var R2_SECRET_ACCESS_KEY.",
    )
    parser.add_argument(
        "--public-base-url",
        default=None,
        help=(
            "Optional public base URL, e.g. https://cdn.example.com. "
            "Used only for printing the final public URL."
        ),
    )
    parser.add_argument(
        "--cache-control",
        default=None,
        help='Optional Cache-Control header, e.g. "public, max-age=31536000".',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be uploaded; do not upload.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.access_key_id or not args.secret_access_key:
        print(
            "Missing credentials. Provide --access-key-id and --secret-access-key, "
            "or set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    s3 = create_r2_client(
        endpoint_url=args.endpoint_url,
        access_key_id=args.access_key_id,
        secret_access_key=args.secret_access_key,
    )

    try:
        upload_path(
            s3=s3,
            bucket=args.bucket,
            local_path=Path(args.local_path),
            remote_key=args.remote_key,
            public_base_url=args.public_base_url,
            cache_control=args.cache_control,
            dry_run=args.dry_run,
        )
    except ClientError as e:
        print(f"R2 upload failed: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()