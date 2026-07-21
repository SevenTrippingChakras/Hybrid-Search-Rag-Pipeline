"""Object storage behind one interface.

``StorageBackend`` is the port; ``S3Storage`` is the adapter, selected by
``build_storage`` from the ``STORAGE_BACKEND`` env var.
"""

from typing import Protocol, runtime_checkable

from hybrid_rag.config import settings


@runtime_checkable
class StorageBackend(Protocol):
    """The storage port: put, get, delete, and existence-check objects by key."""

    def put(self, key: str, data: bytes, content_type: str | None = None) -> None: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...

    def exists(self, key: str) -> bool: ...


class S3Storage:
    """S3-compatible object storage (Supabase / R2 / AWS S3), client made lazily."""

    def __init__(self, bucket: str | None = None) -> None:
        import boto3

        self._bucket = bucket or settings.s3_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
        )

    def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, **extra)

    def get(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as err:
            if err.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise


def build_storage(backend: str | None = None) -> StorageBackend:
    """Construct the store named by ``STORAGE_BACKEND`` (default ``s3``)."""
    backend = (backend or settings.storage_backend).lower()
    if backend == "s3":
        return S3Storage()
    raise ValueError(f"Unknown storage backend: {backend!r}")
