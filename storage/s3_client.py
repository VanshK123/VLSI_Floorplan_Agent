"""S3 client utilities."""
from __future__ import annotations

from pathlib import Path
import boto3


class S3Client:
    """Wrapper around boto3 for uploads."""

    def __init__(self) -> None:
        self.client = boto3.client("s3")

    def upload(self, file_path: str | Path, bucket: str, key: str) -> None:
        """Upload a file to S3."""
        self.client.upload_file(str(file_path), bucket, key)
