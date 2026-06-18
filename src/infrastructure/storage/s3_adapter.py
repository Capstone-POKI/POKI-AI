import os
from io import BytesIO
from typing import Optional


class S3Adapter:
    def __init__(self) -> None:
        self.bucket = os.getenv("S3_BUCKET_NAME", "").strip()
        self.region = os.getenv("AWS_REGION", "").strip() or "ap-northeast-2"
        self.prefix = os.getenv("S3_THUMBNAIL_PREFIX", "ir-thumbnails").strip().strip("/")
        self.public_base_url = os.getenv("S3_PUBLIC_BASE_URL", "").strip().rstrip("/")
        self.use_presigned_url = os.getenv("S3_USE_PRESIGNED_URL", "0").strip() == "1"
        self.presigned_expire_sec = max(
            60, int(os.getenv("S3_PRESIGNED_EXPIRE_SEC", "604800"))
        )
        self._client = None
        self._enabled = bool(self.bucket)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _client_or_none(self):
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client
        try:
            import boto3

            self._client = boto3.client("s3", region_name=self.region)
            return self._client
        except Exception:
            self._enabled = False
            return None

    def upload_slide_thumbnail(
        self,
        *,
        deck_id: str,
        slide_number: int,
        image_bytes: bytes,
        content_type: str = "image/jpeg",
    ) -> Optional[str]:
        client = self._client_or_none()
        if client is None:
            return None
        normalized_content_type = (content_type or "image/jpeg").split(";")[0].strip().lower()
        extension = "jpg"
        if normalized_content_type == "image/png":
            extension = "png"
        elif normalized_content_type == "image/webp":
            extension = "webp"
        key = f"{self.prefix}/{deck_id}/slide-{slide_number}.{extension}"
        try:
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=BytesIO(image_bytes),
                ContentType=content_type,
                CacheControl="public, max-age=31536000, immutable",
            )
            if self.use_presigned_url:
                return client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": self.bucket, "Key": key},
                    ExpiresIn=self.presigned_expire_sec,
                )
            if self.public_base_url:
                return f"{self.public_base_url}/{key}"
            return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"
        except Exception:
            return None

    def upload_file(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """공고문 PDF, IR Deck PDF, 음성 파일 등 원본 파일을 S3에 업로드한다."""
        client = self._client_or_none()
        if client is None:
            return None
        try:
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=BytesIO(data),
                ContentType=content_type,
            )
            if self.use_presigned_url:
                return client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": self.bucket, "Key": key},
                    ExpiresIn=self.presigned_expire_sec,
                )
            if self.public_base_url:
                return f"{self.public_base_url}/{key}"
            return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"
        except Exception:
            return None
