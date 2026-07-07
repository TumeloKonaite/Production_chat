from __future__ import annotations

from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError
import boto3

from app.infrastructure.storage.base import KnowledgeFileStorage, StorageError, StoredFile


class MinioKnowledgeFileStorage(KnowledgeFileStorage):
    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool,
    ) -> None:
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            use_ssl=secure,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        self._bucket_ready = False

    def upload_file(
        self,
        *,
        file_bytes: bytes,
        storage_path: str,
        content_type: str | None = None,
    ) -> StoredFile:
        self._ensure_bucket_exists()
        try:
            kwargs: dict[str, object] = {
                "Bucket": self._bucket,
                "Key": storage_path,
                "Body": file_bytes,
            }
            if content_type:
                kwargs["ContentType"] = content_type
            self._client.put_object(**kwargs)
        except (BotoCoreError, ClientError) as exc:
            raise StorageError("Unable to upload knowledge file to MinIO.") from exc

        return StoredFile(
            provider="minio",
            bucket=self._bucket,
            path=storage_path,
            size_bytes=len(file_bytes),
            content_type=content_type,
        )

    def download_file(
        self,
        *,
        storage_path: str,
    ) -> bytes:
        self._ensure_bucket_exists()
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=storage_path)
            return response["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise StorageError("Unable to download knowledge file from MinIO.") from exc

    def delete_file(
        self,
        *,
        storage_path: str,
    ) -> None:
        self._ensure_bucket_exists()
        try:
            self._client.delete_object(Bucket=self._bucket, Key=storage_path)
        except (BotoCoreError, ClientError) as exc:
            raise StorageError("Unable to delete knowledge file from MinIO.") from exc

    def _ensure_bucket_exists(self) -> None:
        if self._bucket_ready:
            return

        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code not in {"404", "NoSuchBucket", "NotFound"}:
                raise StorageError("Unable to verify MinIO bucket.") from exc
            try:
                self._client.create_bucket(Bucket=self._bucket)
            except (BotoCoreError, ClientError) as create_exc:
                raise StorageError("Unable to create MinIO bucket.") from create_exc
        except BotoCoreError as exc:
            raise StorageError("Unable to verify MinIO bucket.") from exc

        self._bucket_ready = True
