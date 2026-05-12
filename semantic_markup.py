import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rag_graph import SearchResult, clean_uri, create_embedding_backend, extract_label, normalize_vectors


@dataclass
class MarkupTextFragment:
    score: float
    node_uri: str
    node_label: str
    source_name: str
    mention_text: str
    text: str
    pos_start: int
    pos_end: int
    sentence_window: int


class SemanticMarkupDocument:
    def __init__(self, path: Path, data: Dict[str, Any]) -> None:
        self.path = path
        self.data = data
        self.entities = _normalise_entities(data)
        self.text_with_ids = _normalise_text_with_ids(data.get("textWithIds"))
        self.sentence_ranges = _build_sentence_ranges(self.text_with_ids)

    @classmethod
    def load(cls, path: str | Path) -> "SemanticMarkupDocument":
        path = Path(path)
        return cls(path, _load_json_like(path))

    def fragments_for_node(self, node_uri: str, sentence_window: int = 1) -> List[MarkupTextFragment]:
        fragments: List[MarkupTextFragment] = []
        for entity in self.entities:
            if _entity_node_uri(entity) != node_uri:
                continue

            pos_start = _safe_int(entity.get("pos_start"))
            pos_end = _safe_int(entity.get("pos_end"), pos_start)
            if pos_start is None or pos_end is None:
                continue

            text = self._extract_text_window(pos_start, pos_end, sentence_window)
            if not text:
                continue

            node = entity.get("node") if isinstance(entity.get("node"), dict) else None
            label = extract_label(node) if node else ""
            mention = _join_tokens(_tokens_between(self.text_with_ids, pos_start, pos_end))

            fragments.append(
                MarkupTextFragment(
                    score=0.0,
                    node_uri=node_uri,
                    node_label=label or clean_uri(node_uri),
                    source_name=self.path.name,
                    mention_text=mention or label or clean_uri(node_uri),
                    text=text,
                    pos_start=pos_start,
                    pos_end=pos_end,
                    sentence_window=sentence_window,
                )
            )
        return fragments

    def _extract_text_window(self, pos_start: int, pos_end: int, sentence_window: int) -> str:
        if not self.text_with_ids:
            return ""

        sentence_index = _sentence_index_for_position(self.sentence_ranges, pos_start)
        if sentence_index is None:
            return _token_window(self.text_with_ids, pos_start, sentence_window)

        start_index = max(0, sentence_index - sentence_window)
        end_index = min(len(self.sentence_ranges) - 1, sentence_index + sentence_window)
        start_id = self.sentence_ranges[start_index][0]
        end_id = self.sentence_ranges[end_index][1]
        return _join_tokens(_tokens_between(self.text_with_ids, start_id, end_id))


class SemanticMarkupRepository:
    def __init__(self, documents: Sequence[SemanticMarkupDocument]) -> None:
        self.documents = list(documents)

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "SemanticMarkupRepository":
        documents = list(filter(None, map(_load_markup_document_safely, paths)))
        return cls(documents)

    def fragments_for_results(
        self,
        results: Sequence[SearchResult],
        sentence_window: int = 1,
    ) -> List[MarkupTextFragment]:
        fragments: List[MarkupTextFragment] = []
        seen = set()
        for result in results:
            node_uri = result.fragment.node_uri
            for document in self.documents:
                for fragment in document.fragments_for_node(node_uri, sentence_window=sentence_window):
                    key = (fragment.node_uri, fragment.source_name, fragment.pos_start, fragment.pos_end, fragment.text)
                    if key in seen:
                        continue
                    seen.add(key)
                    fragments.append(fragment)
        return fragments

    def rank_fragments(
        self,
        question: str,
        results: Sequence[SearchResult],
        sentence_window: int = 1,
        top_l: int = 2,
        min_score: float = 0.15,
        backend: Optional[Any] = None,
    ) -> List[MarkupTextFragment]:
        candidates = self.fragments_for_results(results, sentence_window=sentence_window)
        if not candidates:
            return []

        texts = list(map(lambda fragment: fragment.text, candidates))
        ranking_backend = _ranking_backend(question, texts, backend)
        vectors = normalize_vectors(ranking_backend.embed(texts))
        query_vec = normalize_vectors(ranking_backend.embed([question]))
        if vectors.size == 0 or query_vec.size == 0:
            return []

        scores = np.dot(vectors, query_vec[0])
        for fragment, score in zip(candidates, scores):
            fragment.score = float(score)

        order_by_node = {result.fragment.node_uri: idx for idx, result in enumerate(results)}
        grouped: Dict[str, List[MarkupTextFragment]] = {}
        for fragment in candidates:
            grouped.setdefault(fragment.node_uri, []).append(fragment)

        selected: List[MarkupTextFragment] = []
        for node_uri, node_fragments in grouped.items():
            ranked = sorted(node_fragments, key=lambda item: item.score, reverse=True)
            accepted = list(filter(lambda fragment: fragment.score >= min_score, ranked))[:top_l]
            if not accepted:
                accepted = ranked[:1]
            selected.extend(accepted)

        deduplicated: Dict[str, MarkupTextFragment] = {}
        for fragment in selected:
            key = _normalise_fragment_text(fragment.text)
            previous = deduplicated.get(key)
            if previous is None:
                deduplicated[key] = fragment
                continue

            previous_order = order_by_node.get(previous.node_uri, 10_000)
            current_order = order_by_node.get(fragment.node_uri, 10_000)
            if (fragment.score, -current_order) > (previous.score, -previous_order):
                deduplicated[key] = fragment

        return sorted(
            deduplicated.values(),
            key=lambda item: (-item.score, order_by_node.get(item.node_uri, 10_000), item.source_name),
        )


def discover_markup_files(root: str | Path, recursive: bool = False) -> List[Path]:
    root = Path(root)
    if root.is_file():
        return [root] if looks_like_markup_file(root) else []

    iterator = root.rglob("*.json") if recursive else root.glob("*.json")
    return sorted(filter(looks_like_markup_file, iterator))


def looks_like_markup_file(path: str | Path) -> bool:
    path = Path(path)
    if path.name.lower() == "graph.json":
        return False
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    except OSError:
        return False
    return "originalText" in text and "textWithIds" in text and ("entites" in text or "entities" in text)


def build_markup_prompt(
    question: str,
    entity_contexts: Sequence[str],
    markup_contexts: Sequence[str],
) -> str:
    main_text = "\n\n".join(f"[Сущность {idx + 1}]\n{text}" for idx, text in enumerate(entity_contexts))
    if not main_text:
        main_text = "Основной текст сущностей онтологии не найден."

    additional_text = "\n\n".join(f"[Фрагмент {idx + 1}]\n{text}" for idx, text in enumerate(markup_contexts))
    if not additional_text:
        additional_text = "Фрагменты семантической разметки для найденных сущностей не обнаружены."

    return (
        f"Ответь на заданный вопрос: {question}\n\n"
        f"Используя основной текст:\n{main_text}\n\n"
        f"Дополняя свой ответ данными текстами:\n{additional_text}\n\n"
        "Если данных недостаточно, укажи это явно. Не добавляй факты, которых нет в тексте. "
        "Не добавляй единицы измерения к числовым значениям, если единица не указана рядом с этим значением в тексте.\n\n"
        "Ответ:"
    )


def format_markup_fragment(fragment: MarkupTextFragment) -> str:
    return (
        f"Источник разметки: {fragment.source_name}\n"
        f"Сущность: {fragment.node_label}\n"
        f"Упоминание: {fragment.mention_text}\n"
        f"Позиции: {fragment.pos_start}-{fragment.pos_end}; окно предложений: -{fragment.sentence_window}/+{fragment.sentence_window}\n"
        f"{fragment.text}"
    )


def _load_markup_document_safely(path: str | Path) -> Optional[SemanticMarkupDocument]:
    try:
        return SemanticMarkupDocument.load(path)
    except Exception as exc:
        print(f"Разметка пропущена: {path} ({exc.__class__.__name__}: {exc})")
        return None


def _load_json_like(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.strip()
    if not stripped.startswith("{"):
        match = re.search(r"=\s*(\{.*\})\s*;?\s*$", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(1)
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("markup root must be an object")
    return data


def _normalise_entities(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_entities = data.get("entites")
    if raw_entities is None:
        raw_entities = data.get("entities")
    if not isinstance(raw_entities, list):
        return []
    return list(filter(lambda entity: isinstance(entity, dict), raw_entities))


def _normalise_text_with_ids(raw: Any) -> List[Tuple[int, str]]:
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = enumerate(raw)
    else:
        return []

    tokens = list(filter(None, map(_normalise_token_pair, items)))
    return sorted(tokens, key=lambda item: item[0])


def _normalise_token_pair(item: Tuple[Any, Any]) -> Optional[Tuple[int, str]]:
    key, value = item
    try:
        token_id = int(key)
    except (TypeError, ValueError):
        return None
    return token_id, "" if value is None else str(value)


def _build_sentence_ranges(tokens: Sequence[Tuple[int, str]]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start_id: Optional[int] = None
    last_id: Optional[int] = None

    for token_id, token in tokens:
        if not token.strip():
            if start_id is not None and last_id is not None:
                ranges.append((start_id, last_id))
                start_id = None
                last_id = None
            continue

        if start_id is None:
            start_id = token_id
        last_id = token_id

        if _ends_sentence(token):
            ranges.append((start_id, token_id))
            start_id = None
            last_id = None

    if start_id is not None and last_id is not None:
        ranges.append((start_id, last_id))
    return ranges


def _ends_sentence(token: str) -> bool:
    stripped = token.strip()
    return bool(stripped) and stripped[-1] in ".!?"


def _sentence_index_for_position(ranges: Sequence[Tuple[int, int]], position: int) -> Optional[int]:
    if not ranges:
        return None

    found = next(
        filter(lambda item: item[1][0] <= position <= item[1][1], enumerate(ranges)),
        None,
    )
    if found is not None:
        return found[0]

    return min(range(len(ranges)), key=lambda idx: abs(position - ((ranges[idx][0] + ranges[idx][1]) / 2)))


def _tokens_between(tokens: Sequence[Tuple[int, str]], start_id: int, end_id: int) -> List[str]:
    return list(
        map(
            lambda item: item[1],
            filter(lambda item: start_id <= item[0] <= end_id, tokens),
        )
    )


def _token_window(tokens: Sequence[Tuple[int, str]], position: int, sentence_window: int) -> str:
    if not tokens:
        return ""
    nearest_index = min(range(len(tokens)), key=lambda idx: abs(tokens[idx][0] - position))
    radius = max(24, 36 * max(1, sentence_window))
    start = max(0, nearest_index - radius)
    end = min(len(tokens), nearest_index + radius + 1)
    return _join_tokens([value for _, value in tokens[start:end]])


def _join_tokens(tokens: Sequence[str]) -> str:
    text = ""
    for token in tokens:
        if not token:
            continue
        if not text:
            text = token
            continue

        stripped = token.lstrip()
        if stripped and stripped[0] in ",.;:!?)»]":
            text += token
        elif text.endswith((" ", "\n", "(", "«", "[")):
            text += token
        else:
            text += " " + token

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[«])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalise_fragment_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _entity_node_uri(entity: Dict[str, Any]) -> str:
    node = entity.get("node") if isinstance(entity.get("node"), dict) else {}
    data = node.get("data", {}) if isinstance(node, dict) else {}
    return str(entity.get("node_uri") or node.get("id") or data.get("uri") or "")


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ranking_backend(question: str, texts: Sequence[str], backend: Optional[Any]) -> Any:
    if backend is not None and not str(getattr(backend, "name", "")).startswith("fallback-tfidf"):
        return backend

    ranking_backend = create_embedding_backend()
    ranking_backend.fit(list(texts) + [question])
    return ranking_backend
