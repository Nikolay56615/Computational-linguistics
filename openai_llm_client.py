import os
from typing import Sequence

from rag_graph import LLMResult, build_extractive_answer, build_prompt


class OpenAILLMClient:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        self.max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "600"))
        self.client = self._create_client()

    @staticmethod
    def _create_client():
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI SDK first: pip install openai") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        return OpenAI(api_key=api_key)

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
            response = self.client.responses.create(
                model=self.model,
                instructions=(
                    "Ты отвечаешь на русском языке. "
                    "Используй только предоставленный контекст. "
                    "Если данных недостаточно, прямо напиши, что данных недостаточно. "
                    "Не выдумывай факты."
                ),
                input=prompt,
                max_output_tokens=self.max_output_tokens,
            )
            answer = (response.output_text or "").strip()
            if not answer:
                answer = build_extractive_answer(fallback_question, fallback_contexts)

            return LLMResult(
                answer=answer,
                prompt=prompt,
                used_api=True,
                mode="openai-api",
            )
        except Exception as exc:
            fallback = build_extractive_answer(fallback_question, fallback_contexts)
            if not fallback:
                fallback = f"OpenAI API недоступен ({exc.__class__.__name__}: {exc})."

            return LLMResult(
                answer=f"OpenAI API недоступен ({exc.__class__.__name__}: {exc}).\n\n{fallback}",
                prompt=prompt,
                used_api=False,
                mode="fallback",
            )
