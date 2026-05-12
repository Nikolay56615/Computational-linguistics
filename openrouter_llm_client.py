import os
from typing import Sequence

import requests

from rag_graph import LLMResult, build_extractive_answer, build_prompt, is_low_quality_generation


DEFAULT_OPENROUTER_MODELS = [
    "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "qwen/qwen3-235b-a22b:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-chat-v3.1:free",
    "z-ai/glm-4.5-air:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]


class OpenRouterLLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        models: Sequence[str] | None = None,
        timeout: int = 90,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("RAG_API_KEY")
        self.timeout = timeout
        self.models = list(models or self._models_from_env() or DEFAULT_OPENROUTER_MODELS)

        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

    @staticmethod
    def _models_from_env() -> list[str]:
        value = os.getenv("OPENROUTER_MODELS") or os.getenv("RAG_API_MODEL", "")
        return [item.strip() for item in value.split(",") if item.strip()]

    def generate(self, question: str, contexts: Sequence[str]) -> LLMResult:
        prompt = build_prompt(question, contexts)
        return self.generate_prompt(prompt, fallback_question=question, fallback_contexts=contexts)

    def generate_prompt(
        self,
        prompt: str,
        fallback_question: str = "",
        fallback_contexts: Sequence[str] = (),
    ) -> LLMResult:
        errors = []
        for model in self.models:
            try:
                answer = self._request(model, prompt)
                if is_low_quality_generation(answer):
                    errors.append(f"{model}: low-quality answer")
                    continue
                return LLMResult(
                    answer=answer,
                    prompt=prompt,
                    used_api=True,
                    mode=f"openrouter:{model}",
                )
            except Exception as exc:
                errors.append(f"{model}: {exc.__class__.__name__}: {exc}")

        fallback = build_extractive_answer(fallback_question, fallback_contexts)
        if not fallback:
            fallback = "OpenRouter API не вернул качественный ответ."
        return LLMResult(
            answer="OpenRouter API не дал качественный ответ. Ошибки моделей: "
            + " | ".join(errors[:5])
            + "\n\n"
            + fallback,
            prompt=prompt,
            used_api=False,
            mode="fallback",
        )

    def _request(self, model: str, prompt: str) -> str:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost/rag-lab",
                "X-Title": "RAG Lab",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты отвечаешь на русском языке. "
                            "Используй только предоставленный контекст. "
                            "Если данных недостаточно, прямо напиши, что данных недостаточно. "
                            "Не выдумывай факты. "
                            "Не добавляй единицы измерения к числовым значениям, если единица не указана рядом с этим значением в контексте."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", "700")),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        return str(message.get("content") or choices[0].get("text") or "").strip()
