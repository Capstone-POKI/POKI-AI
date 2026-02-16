from typing import List, Optional


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
