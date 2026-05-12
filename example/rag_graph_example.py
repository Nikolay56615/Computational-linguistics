from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_graph import RagPipeline, build_index


def create_pipeline(index):
    if os.getenv("OPENROUTER_API_KEY") or (
        os.getenv("RAG_API_URL", "").rstrip("/") == "https://openrouter.ai/api/v1/chat/completions"
        and os.getenv("RAG_API_KEY")
    ):
        from openrouter_llm_client import OpenRouterLLMClient

        print("Генерация: OpenRouter API")
        return RagPipeline(index, llm=OpenRouterLLMClient())

    if os.getenv("YANDEX_FOLDER_ID") and (os.getenv("YANDEX_API_KEY") or os.getenv("YANDEX_IAM_TOKEN")):
        from yandex_gpt_client import YandexGPTClient

        model = os.getenv("YANDEX_MODEL", "yandexgpt-lite")
        print(f"Генерация: YandexGPT API ({model})")
        return RagPipeline(index, llm=YandexGPTClient(model_name=model))

    if os.getenv("OPENAI_API_KEY"):
        from openai_llm_client import OpenAILLMClient

        model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        print(f"Генерация: OpenAI API ({model})")
        return RagPipeline(index, llm=OpenAILLMClient(model=model))

    print("Генерация: встроенный клиент RAG_API/fallback")
    return RagPipeline(index)


def print_results(title, results):
    print(f"\n{title}")
    print("-" * len(title))
    for idx, result in enumerate(results, start=1):
        fragment = result.fragment
        print(f"{idx}. score={result.score:.4f} | {fragment.label}")
        print(fragment.text[:700])
        print()


def main():
    graph_path = PROJECT_ROOT / "graph.json"
    question = "Какие параметры и связи есть у микросхемы 7404?"

    print("1) Загружаем онтологию и строим индекс")
    index = build_index(graph_path)
    print(f"Основных узлов graph.json: {index.graph.raw_node_count}")
    print(f"Связей graph.json: {index.graph.raw_arc_count}")
    print(f"Текстовых фрагментов в индексе: {len(index.fragments)}")
    print(f"Модель эмбеддингов: {index.backend.name}")

    print("\n2) Запускаем RAG")
    print(f"Вопрос: {question}")
    pipeline = create_pipeline(index)
    result = pipeline.answer(question, n=5, m=3)
    print(f"Режим генерации: {result.generation_mode}")

    print_results("Первая фаза поиска (N)", result.first_results)
    print("Черновой ответ:")
    print(result.draft_answer)

    print_results("Вторая фаза поиска (M)", result.second_results)

    print("\nФинальный ответ:")
    print(result.final_answer)

    if result.generation_mode == "fallback":
        print("\nAPI не дал качественный ответ, поэтому ниже показан финальный промпт:")
        print(result.final_prompt)


if __name__ == "__main__":
    main()
