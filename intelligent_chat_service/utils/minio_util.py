import os
import io
import logging
from typing import Optional, List, Dict, Union, BinaryIO, Tuple, Any
from urllib.parse import urlparse

# Synchronous imports
from minio import Minio
from minio.error import S3Error, InvalidResponseError
from minio.commonconfig import ComposeSource

# Asynchronous imports
import aioboto3
import asyncio
import aiohttp
import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)


class MinioUtil:
    """
    Utility class for interacting with MinIO object storage.
    Provides both synchronous and asynchronous methods.
    Asynchronous methods are prefixed with a_.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        region: str = None,
        secure: bool = True,
        http_client: Any = None,
    ):
        """
        Initialize MinIO utility with connection parameters.

        Args:
            endpoint: MinIO server endpoint (without http/https prefix)
            access_key: MinIO access key
            secret_key: MinIO secret key
            region: Optional region name
            secure: Use HTTPS if True, HTTP if False
            http_client: Optional custom HTTP client for the synchronous client
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.secure = secure
        self.http_client = http_client

        # Initialize synchronous client
        self.client = self._get_client()

        # We'll initialize async client only when needed
        self._async_client = None
        self._async_session = None

    def _get_client(self) -> Minio:
        """Create and return a synchronous MinIO client."""
        return Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.region,
            secure=self.secure,
            http_client=self.http_client,
        )

    async def _get_async_client(self):
        """Create and return an asynchronous S3 client compatible with MinIO."""
        if self._async_session is None:
            self._async_session = aioboto3.Session()

        # Format endpoint for boto3
        endpoint_url = f"{'https' if self.secure else 'http'}://{self.endpoint}"

        return self._async_session.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            config=Config(signature_version="s3v4"),
        )

    # Bucket operations

    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if a bucket exists."""
        try:
            return self.client.bucket_exists(bucket_name)
        except S3Error as e:
            logger.error(f"Error checking if bucket {bucket_name} exists: {e}")
            raise

    async def a_bucket_exists(self, bucket_name: str) -> bool:
        """Asynchronously check if a bucket exists."""
        try:
            async with await self._get_async_client() as client:
                try:
                    await client.head_bucket(Bucket=bucket_name)
                    return True
                except Exception:
                    return False
        except Exception as e:
            logger.error(
                f"Error checking if bucket {bucket_name} exists asynchronously in MinIO storage: {e}"
            )
            raise

    def create_bucket(self, bucket_name: str, location: str = None) -> bool:
        """Create a new bucket."""
        try:
            if not self.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name, location=location)
                return True
            return False
        except S3Error as e:
            logger.error(f"Error creating bucket {bucket_name}: {e}")
            raise

    async def a_create_bucket(self, bucket_name: str, region: str = None) -> bool:
        """Asynchronously create a new bucket."""
        try:
            if not await self.a_bucket_exists(bucket_name):
                async with await self._get_async_client() as client:
                    config = {"LocationConstraint": region} if region else {}
                    await client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration=config if config else None,
                    )
                    return True
            return False
        except Exception as e:
            logger.error(f"Error creating bucket {bucket_name} asynchronously: {e}")
            raise

    def list_buckets(self) -> List[Dict[str, str]]:
        """List all buckets."""
        try:
            buckets = self.client.list_buckets()
            return [
                {"name": bucket.name, "creation_date": bucket.creation_date.isoformat()}
                for bucket in buckets
            ]
        except S3Error as e:
            logger.error(f"Error listing buckets: {e}")
            raise

    async def a_list_buckets(self) -> List[Dict[str, str]]:
        """Asynchronously list all buckets."""
        try:
            async with await self._get_async_client() as client:
                response = await client.list_buckets()
                buckets = response.get("Buckets", [])
                return [
                    {
                        "name": bucket["Name"],
                        "creation_date": (
                            bucket["CreationDate"].isoformat()
                            if "CreationDate" in bucket
                            else None
                        ),
                    }
                    for bucket in buckets
                ]
        except Exception as e:
            logger.error(f"Error listing buckets asynchronously: {e}")
            raise

    def remove_bucket(self, bucket_name: str, force: bool = False) -> bool:
        """
        Remove a bucket. If force is True, all objects will be deleted first.
        """
        try:
            if force:
                self.remove_all_objects(bucket_name)
            self.client.remove_bucket(bucket_name)
            return True
        except S3Error as e:
            logger.error(f"Error removing bucket {bucket_name}: {e}")
            raise

    async def a_remove_bucket(self, bucket_name: str, force: bool = False) -> bool:
        """
        Asynchronously remove a bucket. If force is True, all objects will be deleted first.
        """
        try:
            if force:
                await self.a_remove_all_objects(bucket_name)
            async with await self._get_async_client() as client:
                await client.delete_bucket(Bucket=bucket_name)
                return True
        except Exception as e:
            logger.error(f"Error removing bucket {bucket_name} asynchronously: {e}")
            raise

    # Object operations

    def upload_object(
        self,
        bucket_name: str,
        object_name: str,
        data: Union[str, bytes, BinaryIO],
        content_type: str = None,
        metadata: Dict[str, str] = None,
        length: int = None,
    ) -> Dict[str, str]:
        """
        Upload an object to a bucket.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            data: File data as string, bytes, or file-like object
            content_type: Optional content type
            metadata: Optional metadata dictionary
            length: Optional length for file-like objects

        Returns:
            Dictionary with etag, version_id if available
        """
        try:
            # Handle different data types
            if isinstance(data, str):
                data = data.encode("utf-8")
                length = len(data)
                data = io.BytesIO(data)
            elif isinstance(data, bytes):
                length = len(data)
                data = io.BytesIO(data)
            elif length is None and hasattr(data, "seek") and hasattr(data, "tell"):
                # Get length from file-like object if possible
                current_pos = data.tell()
                data.seek(0, os.SEEK_END)
                length = data.tell() - current_pos
                data.seek(current_pos, os.SEEK_SET)

            # Ensure the bucket exists
            if not self.bucket_exists(bucket_name):
                self.create_bucket(bucket_name)

            # Upload the object
            result = self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=data,
                length=length,
                content_type=content_type,
                metadata=metadata,
            )

            return {"etag": result.etag, "version_id": result.version_id}
        except S3Error as e:
            logger.error(
                f"Error uploading object {object_name} to bucket {bucket_name}: {e}"
            )
            raise

    async def a_upload_object(
        self,
        bucket_name: str,
        object_name: str,
        data: Union[str, bytes, BinaryIO, io.BytesIO],
        content_type: str = None,
        metadata: Dict[str, str] = None,
    ) -> Dict[str, str]:
        """
        Asynchronously upload an object to a bucket.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            data: File data as string, bytes, or file-like object
            content_type: Optional content type
            metadata: Optional metadata dictionary

        Returns:
            Dictionary with etag, version_id if available
        """
        try:
            # Handle different data types
            if isinstance(data, str):
                data = data.encode("utf-8")

            # Convert to bytes if it's a file-like object
            if hasattr(data, "read"):
                data = data.read()
                if not isinstance(data, bytes):
                    data = data.encode("utf-8")

            # Ensure bucket exists
            if not await self.a_bucket_exists(bucket_name):
                await self.a_create_bucket(bucket_name)

            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type
            if metadata:
                extra_args["Metadata"] = metadata

            async with await self._get_async_client() as client:
                response = await client.put_object(
                    Bucket=bucket_name, Key=object_name, Body=data, **extra_args
                )

                return {
                    "etag": response.get("ETag", "").strip('"'),
                    "version_id": response.get("VersionId"),
                }
        except Exception as e:
            logger.error(
                f"Error uploading object {object_name} to bucket {bucket_name} asynchronously: {e}"
            )
            raise

    def download_object(
        self, bucket_name: str, object_name: str, file_path: str = None
    ) -> Union[bytes, str]:
        """
        Download an object from a bucket.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            file_path: Optional file path to save the object

        Returns:
            Object data as bytes if file_path is None, otherwise the file path
        """
        try:
            if file_path:
                self.client.fget_object(bucket_name, object_name, file_path)
                return file_path
            else:
                response = self.client.get_object(bucket_name, object_name)
                data = response.read()
                response.close()
                response.release_conn()
                return data
        except S3Error as e:
            logger.error(
                f"Error downloading object {object_name} from bucket {bucket_name}: {e}"
            )
            raise

    async def a_download_object(
        self, bucket_name: str, object_name: str, file_path: str = None
    ) -> Union[bytes, str]:
        """
        Asynchronously download an object from a bucket.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            file_path: Optional file path to save the object

        Returns:
            Object data as bytes if file_path is None, otherwise the file path
        """
        try:
            async with await self._get_async_client() as client:
                response = await client.get_object(Bucket=bucket_name, Key=object_name)

                body = await response["Body"].read()

                if file_path:
                    # Write to file
                    with open(file_path, "wb") as f:
                        f.write(body)
                    return file_path
                else:
                    return body
        except Exception as e:
            logger.error(
                f"Error downloading object {object_name} from bucket {bucket_name} asynchronously: {e}"
            )
            raise

    def list_objects(
        self, bucket_name: str, prefix: str = None, recursive: bool = False
    ) -> List[Dict]:
        """
        List objects in a bucket.

        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter objects
            recursive: If True, list objects recursively

        Returns:
            List of object metadata
        """
        try:
            objects = self.client.list_objects(
                bucket_name, prefix=prefix, recursive=recursive
            )
            return [
                {
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": (
                        obj.last_modified.isoformat() if obj.last_modified else None
                    ),
                    "etag": obj.etag,
                    "content_type": obj.content_type,
                    "is_dir": obj.is_dir,
                }
                for obj in objects
            ]
        except S3Error as e:
            logger.error(f"Error listing objects in bucket {bucket_name}: {e}")
            raise

    async def a_list_objects(
        self, bucket_name: str, prefix: str = None, delimiter: str = None
    ) -> List[Dict]:
        """
        Asynchronously list objects in a bucket.

        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter objects
            delimiter: Optional delimiter for hierarchy

        Returns:
            List of object metadata
        """
        try:
            async with await self._get_async_client() as client:
                params = {"Bucket": bucket_name}
                if prefix:
                    params["Prefix"] = prefix
                if delimiter:
                    params["Delimiter"] = delimiter

                response = await client.list_objects_v2(**params)

                result = []

                # Process regular objects
                for obj in response.get("Contents", []):
                    result.append(
                        {
                            "object_name": obj.get("Key"),
                            "size": obj.get("Size"),
                            "last_modified": (
                                obj.get("LastModified").isoformat()
                                if obj.get("LastModified")
                                else None
                            ),
                            "etag": obj.get("ETag", "").strip('"'),
                            "storage_class": obj.get("StorageClass"),
                            "is_dir": False,
                        }
                    )

                # Process prefixes (directories)
                for prefix in response.get("CommonPrefixes", []):
                    result.append({"object_name": prefix.get("Prefix"), "is_dir": True})

                return result
        except Exception as e:
            logger.error(
                f"Error listing objects in bucket {bucket_name} asynchronously: {e}"
            )
            raise

    def remove_object(self, bucket_name: str, object_name: str) -> bool:
        """
        Remove an object from a bucket.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object

        Returns:
            True if successful
        """
        try:
            self.client.remove_object(bucket_name, object_name)
            return True
        except S3Error as e:
            logger.error(
                f"Error removing object {object_name} from bucket {bucket_name}: {e}"
            )
            raise

    async def a_remove_object(self, bucket_name: str, object_name: str) -> bool:
        """
        Asynchronously remove an object from a bucket.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object

        Returns:
            True if successful
        """
        try:
            async with await self._get_async_client() as client:
                await client.delete_object(Bucket=bucket_name, Key=object_name)
                return True
        except Exception as e:
            logger.error(
                f"Error removing object {object_name} from bucket {bucket_name} asynchronously: {e}"
            )
            raise

    def remove_objects(self, bucket_name: str, object_names: List[str]) -> List[str]:
        """
        Remove multiple objects from a bucket.

        Args:
            bucket_name: Name of the bucket
            object_names: List of object names to remove

        Returns:
            List of objects that failed to delete
        """
        try:
            errors = self.client.remove_objects(bucket_name, object_names)
            return [error.object_name for error in errors]
        except S3Error as e:
            logger.error(f"Error removing objects from bucket {bucket_name}: {e}")
            raise

    async def a_remove_objects(
        self, bucket_name: str, object_names: List[str]
    ) -> List[str]:
        """
        Asynchronously remove multiple objects from a bucket.

        Args:
            bucket_name: Name of the bucket
            object_names: List of object names to remove

        Returns:
            List of objects that failed to delete
        """
        try:
            objects = [{"Key": name} for name in object_names]

            async with await self._get_async_client() as client:
                response = await client.delete_objects(
                    Bucket=bucket_name, Delete={"Objects": objects}
                )

                failed = []
                if "Errors" in response:
                    failed = [error["Key"] for error in response["Errors"]]

                return failed
        except Exception as e:
            logger.error(
                f"Error removing objects from bucket {bucket_name} asynchronously: {e}"
            )
            raise

    def remove_all_objects(self, bucket_name: str, prefix: str = None) -> bool:
        """
        Remove all objects from a bucket with an optional prefix.

        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter objects

        Returns:
            True if successful
        """
        try:
            delete_object_list = self.client.list_objects(
                bucket_name, prefix=prefix, recursive=True
            )
            objects_to_delete = [obj.object_name for obj in delete_object_list]

            if not objects_to_delete:
                return True

            # Delete in batches of 1000 (S3 limit)
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i : i + 1000]
                self.remove_objects(bucket_name, batch)

            return True
        except S3Error as e:
            logger.error(f"Error removing all objects from bucket {bucket_name}: {e}")
            raise

    async def a_remove_all_objects(self, bucket_name: str, prefix: str = None) -> bool:
        """
        Asynchronously remove all objects from a bucket with an optional prefix.

        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter objects

        Returns:
            True if successful
        """
        try:
            objects = await self.a_list_objects(bucket_name, prefix)
            objects_to_delete = [
                obj["object_name"] for obj in objects if not obj.get("is_dir", False)
            ]

            if not objects_to_delete:
                return True

            # Delete in batches of 1000 (S3 limit)
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i : i + 1000]
                await self.a_remove_objects(bucket_name, batch)

            return True
        except Exception as e:
            logger.error(
                f"Error removing all objects from bucket {bucket_name} asynchronously: {e}"
            )
            raise

    # URL operations

    def get_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        expires_in: int = 604800,  # 7 days in seconds
        method: str = "GET",
    ) -> str:
        """
        Generate a presigned URL for an object.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            expires_in: URL expiration time in seconds
            method: HTTP method (GET, PUT)

        Returns:
            Presigned URL
        """
        try:
            url = self.client.presigned_url(
                method, bucket_name, object_name, expires=expires_in
            )
            return url
        except S3Error as e:
            logger.error(f"Error generating presigned URL for {object_name}: {e}")
            raise

    async def a_get_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        expires_in: int = 604800,  # 7 days in seconds
        method: str = "get_object",
    ) -> str:
        """
        Asynchronously generate a presigned URL for an object.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            expires_in: URL expiration time in seconds
            method: S3 client method name

        Returns:
            Presigned URL
        """
        try:
            async with await self._get_async_client() as client:
                url = await client.generate_presigned_url(
                    ClientMethod=method,
                    Params={"Bucket": bucket_name, "Key": object_name},
                    ExpiresIn=expires_in,
                )
                return url
        except Exception as e:
            logger.error(
                f"Error generating presigned URL for {object_name} asynchronously: {e}"
            )
            raise

    # Utility functions

    def object_exists(self, bucket_name: str, object_name: str) -> bool:
        """
        Check if an object exists in a bucket.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object

        Returns:
            True if the object exists
        """
        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey" or e.code == "NotFound":
                return False
            logger.error(f"Error checking if object {object_name} exists: {e}")
            raise

    async def a_object_exists(self, bucket_name: str, object_name: str) -> bool:
        """
        Asynchronously check if an object exists in a bucket.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object

        Returns:
            True if the object exists
        """
        try:
            async with await self._get_async_client() as client:
                try:
                    await client.head_object(Bucket=bucket_name, Key=object_name)
                    return True
                except Exception as e:
                    if (
                        "Not Found" in str(e)
                        or "NoSuchKey" in str(e)
                        or "404" in str(e)
                    ):
                        return False
                    raise
        except Exception as e:
            logger.error(
                f"Error checking if object {object_name} exists asynchronously: {e}"
            )
            raise

    def get_object_metadata(self, bucket_name: str, object_name: str) -> Dict:
        """
        Get metadata for an object.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object

        Returns:
            Dictionary with object metadata
        """
        try:
            stat = self.client.stat_object(bucket_name, object_name)
            return {
                "size": stat.size,
                "etag": stat.etag,
                "last_modified": (
                    stat.last_modified.isoformat() if stat.last_modified else None
                ),
                "content_type": stat.content_type,
                "metadata": stat.metadata,
            }
        except S3Error as e:
            logger.error(f"Error getting metadata for object {object_name}: {e}")
            raise

    async def a_get_object_metadata(self, bucket_name: str, object_name: str) -> Dict:
        """
        Asynchronously get metadata for an object.

        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object

        Returns:
            Dictionary with object metadata
        """
        try:
            async with await self._get_async_client() as client:
                response = await client.head_object(Bucket=bucket_name, Key=object_name)

                metadata = {}
                # Extract user metadata
                for key, value in response.items():
                    if key.startswith("Metadata-"):
                        metadata[key[9:]] = value

                return {
                    "size": response.get("ContentLength"),
                    "etag": response.get("ETag", "").strip('"'),
                    "last_modified": (
                        response.get("LastModified").isoformat()
                        if response.get("LastModified")
                        else None
                    ),
                    "content_type": response.get("ContentType"),
                    "metadata": metadata,
                }
        except Exception as e:
            logger.error(
                f"Error getting metadata for object {object_name} asynchronously: {e}"
            )
            raise

    def copy_object(
        self,
        source_bucket: str,
        source_object: str,
        target_bucket: str,
        target_object: str,
        metadata: Dict[str, str] = None,
    ) -> Dict[str, str]:
        """
        Copy an object from one location to another.

        Args:
            source_bucket: Source bucket name
            source_object: Source object name
            target_bucket: Target bucket name
            target_object: Target object name
            metadata: Optional metadata for the new object

        Returns:
            Dictionary with etag, version_id of new object
        """
        try:
            result = self.client.copy_object(
                target_bucket,
                target_object,
                f"{source_bucket}/{source_object}",
                metadata=metadata,
            )

            return {"etag": result.etag, "version_id": result.version_id}
        except S3Error as e:
            logger.error(
                f"Error copying object from {source_bucket}/{source_object} to {target_bucket}/{target_object}: {e}"
            )
            raise

    async def a_copy_object(
        self,
        source_bucket: str,
        source_object: str,
        target_bucket: str,
        target_object: str,
        metadata: Dict[str, str] = None,
    ) -> Dict[str, str]:
        """
        Asynchronously copy an object from one location to another.

        Args:
            source_bucket: Source bucket name
            source_object: Source object name
            target_bucket: Target bucket name
            target_object: Target object name
            metadata: Optional metadata for the new object

        Returns:
            Dictionary with etag, version_id of new object
        """
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata
                extra_args["MetadataDirective"] = "REPLACE"

            async with await self._get_async_client() as client:
                response = await client.copy_object(
                    Bucket=target_bucket,
                    Key=target_object,
                    CopySource={"Bucket": source_bucket, "Key": source_object},
                    **extra_args,
                )

                return {
                    "etag": response.get("CopyObjectResult", {})
                    .get("ETag", "")
                    .strip('"'),
                    "version_id": response.get("VersionId"),
                }
        except Exception as e:
            logger.error(
                f"Error copying object from {source_bucket}/{source_object} to {target_bucket}/{target_object} asynchronously: {e}"
            )
            raise

    # Helper functions

    @staticmethod
    def parse_s3_uri(uri: str) -> Tuple[str, str]:
        """
        Parse an S3 URI into bucket and object names.

        Args:
            uri: S3 URI in the format s3://bucket/object

        Returns:
            Tuple of (bucket_name, object_name)
        """
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Not a valid S3 URI: {uri}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        return bucket, key
