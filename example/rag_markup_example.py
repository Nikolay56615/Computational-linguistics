from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_graph import LLMResult, RagPipeline, build_extractive_answer, build_index
from semantic_markup import (
    SemanticMarkupRepository,
    build_markup_prompt,
    discover_markup_files,
    format_markup_fragment,
)


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


def print_markup_results(title, fragments):
    print(f"\n{title}")
    print("-" * len(title))
    if not fragments:
        print("Фрагменты размеченных текстов для найденных сущностей не найдены.")
        return

    for idx, fragment in enumerate(fragments, start=1):
        print(f"{idx}. score={fragment.score:.4f} | {fragment.node_label} | {fragment.source_name}")
        print(fragment.text[:900])
        print()


def generate_from_ready_prompt(llm, prompt: str, question: str, contexts) -> LLMResult:
    if hasattr(llm, "generate_prompt"):
        return llm.generate_prompt(prompt, fallback_question=question, fallback_contexts=contexts)

    result = llm.generate(question, contexts)
    result.prompt = prompt
    return result


def is_uninformative_draft(answer: str) -> bool:
    normalized = answer.lower()
    markers = (
        "недостаточно информации",
        "недостаточного количества информации",
        "не содержит достаточного",
        "нет данных",
        "не могу ответить",
    )
    return len(answer.strip()) < 40 or any(marker in normalized for marker in markers)


def build_second_phase_query(question: str, draft_answer: str, first_results) -> str:
    if draft_answer.strip() and not is_uninformative_draft(draft_answer):
        return draft_answer

    first_contexts = [result.fragment.text for result in first_results]
    extractive = build_extractive_answer(question, first_contexts)
    labels = "; ".join(result.fragment.label for result in first_results[:5])
    return (
        f"{question}\n\n"
        f"Черновая выжимка по найденным узлам:\n{extractive}\n\n"
        f"Найденные сущности: {labels}"
    )


def build_markup_fallback_answer(question: str, results, markup_fragments) -> str:
    if not results:
        return f"По вопросу «{question}» релевантные сущности не найдены."

    target = results[0].fragment
    facts = []
    for line in target.text.splitlines():
        line = line.strip()
        if not line or line.startswith("Название:") or line.startswith("Описание:"):
            continue
        if line.startswith("Идентификаторы текстов разметки:"):
            continue
        facts.append(line)

    answer_lines = [f"По данным графа знаний сущность «{target.label}» имеет следующие параметры и связи:"]
    answer_lines.extend(f"- {fact}" for fact in facts[:12])

    selected_texts = []
    seen_texts = set()
    for fragment in markup_fragments:
        if fragment.node_uri != target.node_uri and fragment.node_label != target.label:
            continue
        text = fragment.text.strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            selected_texts.append(text)
        if len(selected_texts) >= 3:
            break

    if selected_texts:
        answer_lines.append("")
        answer_lines.append("Семантическая разметка текстов дополнительно уточняет:")
        answer_lines.extend(f"- {text}" for text in selected_texts)

    return "\n".join(answer_lines)


def load_markup_repository():
    markup_root = Path(os.getenv("MARKUP_DIR", str(PROJECT_ROOT)))
    recursive = os.getenv("MARKUP_RECURSIVE", "0").lower() in {"1", "true", "yes"}
    markup_files = discover_markup_files(markup_root, recursive=recursive)
    repository = SemanticMarkupRepository.from_paths(markup_files)
    return markup_files, repository


def main():
    graph_path = PROJECT_ROOT / "graph.json"
    question = os.getenv("RAG_QUESTION", "Какие параметры и связи есть у микросхемы 7404?")
    n = int(os.getenv("RAG_N", "5"))
    m = int(os.getenv("RAG_M", "3"))
    sentence_window = int(os.getenv("MARKUP_SENTENCE_WINDOW", "1"))
    top_l = int(os.getenv("MARKUP_TOP_L", "2"))
    min_score = float(os.getenv("MARKUP_MIN_SCORE", "0.15"))

    print("1) Загружаем онтологию и строим индекс")
    index = build_index(graph_path)
    print(f"Основных узлов graph.json: {index.graph.raw_node_count}")
    print(f"Связей graph.json: {index.graph.raw_arc_count}")
    print(f"Текстовых фрагментов в индексе: {len(index.fragments)}")
    print(f"Модель эмбеддингов: {index.backend.name}")

    print("\n2) Загружаем файлы семантической разметки")
    markup_files, markup_repository = load_markup_repository()
    print(f"Файлов разметки: {len(markup_files)}")
    for document in markup_repository.documents:
        print(
            f"- {document.path.name}: сущностей={len(document.entities)}, "
            f"предложений={len(document.sentence_ranges)}"
        )
    print(
        f"K={sentence_window}: берем {sentence_window} предложение до и после упоминания; "
        f"L={top_l}: оставляем до {top_l} фрагментов на сущность; порог={min_score}."
    )

    print("\n3) Запускаем RAG по графу знаний")
    print(f"Вопрос: {question}")
    pipeline = create_pipeline(index)

    first_results = index.search(question, top_k=n)
    first_contexts = [result.fragment.text for result in first_results]
    draft = pipeline.llm.generate(question, first_contexts)

    second_query = build_second_phase_query(question, draft.answer, first_results)
    second_results = index.search(second_query, top_k=m)
    all_results = RagPipeline._merge_results(first_results, second_results)

    print_results("Первая фаза поиска по графу (N)", first_results)
    print("Черновой ответ:")
    print(draft.answer)
    if second_query != draft.answer:
        print("\nЧерновой ответ признан недостаточным, поэтому для второй фазы использована выжимка из найденных N узлов.")
    print_results("Вторая фаза поиска по графу (M)", second_results)

    print("\n4) Подбираем фрагменты из семантической разметки")
    ranked_markup_fragments = markup_repository.rank_fragments(
        question,
        all_results,
        sentence_window=sentence_window,
        top_l=top_l,
        min_score=min_score,
        backend=index.backend,
    )
    print_markup_results("Топ фрагментов размеченных текстов", ranked_markup_fragments)

    entity_contexts = [result.fragment.text for result in all_results]
    markup_contexts = [format_markup_fragment(fragment) for fragment in ranked_markup_fragments]
    final_prompt = build_markup_prompt(question, entity_contexts, markup_contexts)
    final = generate_from_ready_prompt(
        pipeline.llm,
        final_prompt,
        question,
        entity_contexts + markup_contexts,
    )
    if final.mode == "fallback":
        final.answer = build_markup_fallback_answer(question, all_results, ranked_markup_fragments)

    print("\n5) Финальный ответ с учетом графа и разметки")
    print(f"Режим генерации: {final.mode}")
    print(final.answer)

    if os.getenv("SHOW_MARKUP_PROMPT", "0").lower() in {"1", "true", "yes"}:
        print("\nФинальный промпт:")
        print(final.prompt)


if __name__ == "__main__":
    main()
