import os
from typing import Sequence

import requests

from rag_graph import LLMResult, build_extractive_answer, build_prompt


class YandexGPTClient:
    def __init__(
        self,
        folder_id: str | None = None,
        api_key: str | None = None,
        iam_token: str | None = None,
        model_name: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        self.iam_token = iam_token or os.getenv("YANDEX_IAM_TOKEN")
        self.model_name = model_name or os.getenv("YANDEX_MODEL", "yandexgpt-lite")
        self.timeout = timeout

        if not self.folder_id:
            raise RuntimeError("YANDEX_FOLDER_ID is not set")
        if not self.api_key and not self.iam_token:
            raise RuntimeError("Set YANDEX_API_KEY or YANDEX_IAM_TOKEN")

    @property
    def model_uri(self) -> str:
        if self.model_name.startswith("gpt://"):
            return self.model_name
        return f"gpt://{self.folder_id}/{self.model_name}"

    def generate(self, question: str, contexts: Sequence[str]) -> LLMResult:
        prompt = build_prompt(question, contexts)
        return self.generate_prompt(prompt, fallback_question=question, fallback_contexts=contexts)

    def generate_prompt(
        self,
        prompt: str,
        fallback_question: str = "",
        fallback_contexts: Sequence[str] = (),
    ) -> LLMResult:
        try:
            response = requests.post(
                "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                headers=self._headers(),
                json={
                    "modelUri": self.model_uri,
                    "completionOptions": {
                        "stream": False,
                        "temperature": 0.2,
                        "maxTokens": "600",
                    },
                    "messages": [
                        {
                            "role": "system",
                            "text": (
                                "Ты отвечаешь на русском языке. "
                                "Используй только предоставленный контекст. "
                                "Если данных недостаточно, прямо напиши, что данных недостаточно. "
                                "Не выдумывай факты."
                            ),
                        },
                        {"role": "user", "text": prompt},
                    ],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            answer = self._extract_answer(data) or build_extractive_answer(fallback_question, fallback_contexts)
            return LLMResult(answer=answer, prompt=prompt, used_api=True, mode="yandex-api")
        except Exception as exc:
            fallback = build_extractive_answer(fallback_question, fallback_contexts)
            if not fallback:
                fallback = f"YandexGPT API недоступен ({exc.__class__.__name__}: {exc})."
            return LLMResult(
                answer=f"YandexGPT API недоступен ({exc.__class__.__name__}: {exc}).\n\n{fallback}",
                prompt=prompt,
                used_api=False,
                mode="fallback",
            )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.iam_token:
            headers["Authorization"] = f"Bearer {self.iam_token}"
        else:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        return headers

    @staticmethod
    def _extract_answer(data: dict) -> str:
        alternatives = data.get("result", {}).get("alternatives", [])
        if not alternatives:
            alternatives = data.get("alternatives", [])
        if not alternatives:
            return ""

        message = alternatives[0].get("message", {})
        return str(message.get("text") or "").strip()
