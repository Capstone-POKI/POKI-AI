import os
from typing import List, Optional

import requests


class EmbeddingClient:
    def __init__(self, model_name: str = "text-embedding-004"):
        self.model_name = model_name
        self._model = None

    def init_vertex(self, project_id: str, location: str = "us-central1") -> None:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        vertexai.init(project=project_id, location=location)
        self._model = TextEmbeddingModel.from_pretrained(self.model_name)

    def embed(self, texts: List[str], task_type: Optional[str] = None) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError("Embedding model is not initialized")

        kwargs = {}
        if task_type:
            kwargs["task_type"] = task_type

        vectors: List[List[float]] = []
        for text in texts:
            response = self._model.get_embeddings([text], **kwargs)
            vectors.append(list(response[0].values))
        return vectors


class GeminiEmbeddingClient:
    """Gemini Embedding API client using REST (no Vertex AI dependency)."""

    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def embed(self, texts: List[str], task_type: Optional[str] = None) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set for embedding")

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:batchEmbedContents"

        vectors: List[List[float]] = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            requests_payload = []
            for text in batch:
                truncated = text[:2000]
                req = {
                    "model": f"models/{self.model_name}",
                    "content": {"parts": [{"text": truncated}]},
                }
                if task_type:
                    req["taskType"] = task_type
                requests_payload.append(req)

            payload = {"requests": requests_payload}
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                raise RuntimeError(f"Gemini embedding failed: {response.status_code} {response.text[:300]}")

            data = response.json()
            for emb in data.get("embeddings", []):
                vectors.append(emb.get("values", []))

        return vectors
