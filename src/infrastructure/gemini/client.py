import json
import os
from typing import Any, Dict, Optional


class GeminiJSONClient:
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        model_candidates: Optional[list[str]] = None,
    ):
        self.project_id = project_id or os.getenv("PROJECT_ID", "pitchcoachai")
        self.location = location
        self.model = None
        self.model_name = None
        self.model_candidates = model_candidates or [
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-001",
        ]
        self._init_model()

    def _init_model(self) -> None:
        try:
            key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            credentials = None
            if key_path:
                from google.oauth2 import service_account

                if not os.path.isabs(key_path):
                    key_path = os.path.join(os.getcwd(), key_path)
                if os.path.exists(key_path):
                    credentials = service_account.Credentials.from_service_account_file(key_path)

            import vertexai
            from vertexai.generative_models import GenerativeModel

            if credentials:
                vertexai.init(project=self.project_id, location=self.location, credentials=credentials)
            else:
                vertexai.init(project=self.project_id, location=self.location)

            for model_name in self.model_candidates:
                try:
                    self.model = GenerativeModel(model_name)
                    self.model_name = model_name
                    break
                except Exception:
                    continue
        except Exception:
            self.model = None
            self.model_name = None

    def generate_json(self, prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Gemini model is not available")

        response = self.model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": temperature,
            },
        )
        raw = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
