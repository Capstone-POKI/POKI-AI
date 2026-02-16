import json
import os
from typing import Any, Dict
from urllib.parse import quote

import requests


class GeminiJSONClient:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model = None
        self.model_name = None
        self.model_candidates = []
        self.api_key = None
        self._init_model(os.getenv("GEMINI_MODEL", model_name))

    def _init_model(self, model_name: str) -> None:
        try:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                self.model = None
                self.model_name = None
                self.api_key = None
                return

            self.api_key = api_key
            self.model = True
            self.model_name = model_name
            self.model_candidates = [model_name]
            if "GEMINI_MODEL" not in os.environ:
                self.model_candidates.extend(
                    [
                        "gemini-2.5-flash-lite",
                        "gemini-2.0-flash",
                        "gemini-1.5-flash",
                    ]
                )
        except Exception:
            self.model = None
            self.model_name = None
            self.model_candidates = []
            self.api_key = None

    def generate_json(self, prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
        if self.model is None or self.api_key is None or not self.model_candidates:
            raise RuntimeError("Gemini model is not available")

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        last_error = ""
        for model_name in self.model_candidates:
            url = "https://generativelanguage.googleapis.com/v1/models/" f"{quote(model_name)}:generateContent"
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code == 404:
                last_error = f"{model_name}: 404"
                continue

            if response.status_code != 200:
                body_preview = (response.text or "")[:500]
                raise RuntimeError(f"Gemini request failed: {response.status_code} body={body_preview}")

            self.model_name = model_name
            data = response.json()

            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini response has no candidates")
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError("Gemini response has no parts")
            raw = (parts[0].get("text", "") or "").replace("```json", "").replace("```", "").strip()
            if not raw:
                raise RuntimeError("Gemini response text is empty")
            return json.loads(raw)

        raise RuntimeError(
            "Gemini request failed: no available model for v1 generateContent. "
            f"Tried={self.model_candidates}. Last={last_error}"
        )
