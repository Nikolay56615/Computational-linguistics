import json
import math
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


RDF_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDF_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"

OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
OWL_NAMED_INDIVIDUAL = "http://www.w3.org/2002/07/owl#NamedIndividual"
OWL_DATATYPE_PROPERTY = "http://www.w3.org/2002/07/owl#DatatypeProperty"
OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"


SYSTEM_RELATION_LABELS = {
    RDF_TYPE: "Имеет тип",
    RDFS_SUBCLASS_OF: "Является подклассом",
    RDFS_DOMAIN: "Область определения",
    RDFS_RANGE: "Область значений",
}


@dataclass
class GraphArc:
    id: Any
    uri: str
    source: str
    target: str
    raw: Dict[str, Any]


@dataclass
class TextFragment:
    node_id: str
    node_uri: str
    label: str
    text: str


@dataclass
class SearchResult:
    score: float
    fragment: TextFragment


@dataclass
class LLMResult:
    answer: str
    prompt: str
    used_api: bool
    mode: str


@dataclass
class RagResult:
    question: str
    first_results: List[SearchResult]
    second_results: List[SearchResult]
    all_results: List[SearchResult]
    draft_answer: str
    final_answer: str
    draft_prompt: str
    final_prompt: str
    used_api: bool
    generation_mode: str


def clean_literal(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(clean_literal(item) for item in value if clean_literal(item))
    if isinstance(value, dict):
        return ", ".join(f"{clean_uri(k)}={clean_literal(v)}" for k, v in value.items())

    text = str(value).strip()
    if text.endswith("@ru") or text.endswith("@en"):
        text = text[:-3].strip()
    return text


def clean_uri(uri: str) -> str:
    if not uri:
        return ""
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def tokenize_for_match(text: str) -> List[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", text.lower())


def normalize_for_match(text: str) -> str:
    return " ".join(tokenize_for_match(text))


def extract_label(node: Optional[Dict[str, Any]]) -> str:
    if not node:
        return ""

    data = node.get("data", {})
    params_values = data.get("params_values", {})
    labels = params_values.get(RDF_LABEL) or data.get(RDF_LABEL) or []
    if isinstance(labels, str):
        labels = [labels]

    for label in labels:
        if str(label).endswith("@ru"):
            return clean_literal(label)
    return clean_literal(labels[0]) if labels else ""


def extract_comment(node: Optional[Dict[str, Any]]) -> str:
    if not node:
        return ""

    data = node.get("data", {})
    params_values = data.get("params_values", {})
    return clean_literal(params_values.get(RDF_COMMENT) or data.get(RDF_COMMENT))


class OntologyGraph:
    def __init__(self, main_nodes: List[Dict[str, Any]], arcs: List[GraphArc]):
        self.main_nodes = main_nodes
        self.arcs = arcs
        self.nodes_by_uri: Dict[str, Dict[str, Any]] = {}
        self.raw_node_count = len(main_nodes)
        self.raw_arc_count = len(arcs)

        for node in main_nodes:
            self.add_node(node)
        for arc in arcs:
            data = arc.raw.get("data", {})
            self.add_node(data.get("start_node"))
            self.add_node(data.get("end_node"))

        self.outgoing: Dict[str, List[GraphArc]] = {}
        self.incoming: Dict[str, List[GraphArc]] = {}
        for arc in arcs:
            self.outgoing.setdefault(arc.source, []).append(arc)
            self.incoming.setdefault(arc.target, []).append(arc)

    def add_node(self, node: Optional[Dict[str, Any]]) -> None:
        if not isinstance(node, dict):
            return
        node_id = node.get("id") or node.get("data", {}).get("uri")
        if node_id:
            self.nodes_by_uri[str(node_id)] = node

    def label_for_uri(self, uri: str) -> str:
        if not uri:
            return ""
        if uri in SYSTEM_RELATION_LABELS:
            return SYSTEM_RELATION_LABELS[uri]

        node = self.nodes_by_uri.get(uri)
        label = extract_label(node)
        return label or clean_uri(uri)

    def node_kind(self, node: Dict[str, Any]) -> str:
        labels = set(node.get("data", {}).get("labels", []))
        if OWL_CLASS in labels:
            return "Класс"
        if OWL_NAMED_INDIVIDUAL in labels:
            return "Экземпляр"
        if OWL_DATATYPE_PROPERTY in labels:
            return "Свойство данных"
        if OWL_OBJECT_PROPERTY in labels:
            return "Свойство отношения"
        return "Узел"


def load_ontology_graph(path: str | Path) -> OntologyGraph:
    with open(path, encoding="utf-8") as graph_file:
        raw_graph = json.load(graph_file)

    raw_arcs = raw_graph.get("arcs", [])
    arcs = [
        GraphArc(
            id=arc.get("id"),
            uri=arc.get("data", {}).get("uri") or arc.get("type") or "",
            source=arc.get("source") or arc.get("data", {}).get("start_node", {}).get("id") or "",
            target=arc.get("target") or arc.get("data", {}).get("end_node", {}).get("id") or "",
            raw=arc,
        )
        for arc in raw_arcs
        if arc.get("source") and arc.get("target")
    ]

    return OntologyGraph(raw_graph.get("nodes", []), arcs)


def _format_property_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(clean_literal(item) for item in value)
    return clean_literal(value)


def _is_service_param(uri: str) -> bool:
    return uri in {RDF_LABEL, RDF_COMMENT, "uri"} or uri.endswith("#label") or uri.endswith("#comment")


def _class_labels(node: Dict[str, Any], graph: OntologyGraph) -> List[str]:
    data = node.get("data", {})
    ontology_uri = data.get("ontology_uri")
    result = []
    for label_uri in data.get("labels", []):
        if label_uri.startswith("http://www.w3.org/2002/07/owl#"):
            continue
        if ontology_uri and label_uri == ontology_uri:
            continue
        label = graph.label_for_uri(label_uri)
        if label and label != clean_uri(label_uri):
            result.append(label)
    return result


def _text_mentions_summary(node: Dict[str, Any]) -> List[str]:
    mentions = node.get("data", {}).get("text_mentions") or []
    if not mentions:
        return []

    sources = []
    for mention in mentions:
        source_uri = mention.get("original_object_uri")
        if source_uri and source_uri not in sources:
            sources.append(source_uri)

    lines = [f"Упоминаний в размеченных текстах: {len(mentions)}"]
    if sources:
        short_sources = ", ".join(clean_uri(uri) for uri in sources[:3])
        if len(sources) > 3:
            short_sources += f" и еще {len(sources) - 3}"
        lines.append(f"Идентификаторы текстов разметки: {short_sources}")
    return lines


def _relation_line(arc: GraphArc, graph: OntologyGraph, outgoing: bool) -> str:
    rel_label = graph.label_for_uri(arc.uri)
    source_label = graph.label_for_uri(arc.source)
    target_label = graph.label_for_uri(arc.target)

    if arc.uri == RDF_TYPE:
        return f"Имеет тип: {target_label}" if outgoing else f"Экземпляр класса: {source_label}"
    if arc.uri == RDFS_SUBCLASS_OF:
        return f"Является подклассом: {target_label}" if outgoing else f"Имеет подкласс: {source_label}"
    if arc.uri == RDFS_DOMAIN:
        return f"Применяется к классу: {target_label}" if outgoing else f"Атрибут или отношение класса: {source_label}"
    if arc.uri == RDFS_RANGE:
        return f"Имеет область значений: {target_label}" if outgoing else f"Используется как значение свойства: {source_label}"

    if outgoing:
        return f"{rel_label}: {target_label}"
    return f"Связан через {rel_label} с: {source_label}"


def node_to_text(node: Dict[str, Any], graph: OntologyGraph, max_relations: int = 24) -> str:
    data = node.get("data", {})
    params_values = data.get("params_values", {})
    node_id = node.get("id") or data.get("uri", "")
    label = extract_label(node) or graph.label_for_uri(node_id)
    comment = extract_comment(node)
    kind = graph.node_kind(node)

    lines: List[str] = []
    if label:
        lines.append(f"Название: {label}")
    if comment:
        lines.append(f"Описание: {comment}")
    if kind:
        lines.append(f"Тип узла: {kind}")

    for class_label in _class_labels(node, graph):
        lines.append(f"Имеет тип: {class_label}")

    lines.extend(_text_mentions_summary(node))

    for param_uri, value in params_values.items():
        if _is_service_param(param_uri):
            continue
        param_label = graph.label_for_uri(param_uri)
        formatted_value = _format_property_value(value)
        if param_label and formatted_value:
            lines.append(f"{param_label}: {formatted_value}")

    relation_lines = []
    for arc in graph.outgoing.get(node_id, []):
        relation_lines.append(_relation_line(arc, graph, outgoing=True))
    for arc in graph.incoming.get(node_id, []):
        relation_lines.append(_relation_line(arc, graph, outgoing=False))

    seen = set(lines)
    for relation in relation_lines:
        if relation and relation not in seen:
            lines.append(relation)
            seen.add(relation)
        if len(lines) >= max_relations + 4:
            break

    return "\n".join(lines)


class FallbackEmbeddingBackend:
    """
    Lightweight TF-IDF backend for offline demonstration when sentence-transformers
    is not installed. The project still prefers multilingual MPNet when available.
    """

    name = "fallback-tfidf"

    def __init__(self) -> None:
        self.vocabulary: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([], dtype=np.float32)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", text.lower())

    def fit(self, texts: Sequence[str]) -> None:
        documents = [set(self.tokenize(text)) for text in texts]
        terms = sorted({term for document in documents for term in document})
        self.vocabulary = {term: idx for idx, term in enumerate(terms)}

        if not terms:
            self.idf = np.array([], dtype=np.float32)
            return

        doc_count = len(documents)
        df = np.zeros(len(terms), dtype=np.float32)
        for document in documents:
            for term in document:
                df[self.vocabulary[term]] += 1
        self.idf = np.log((1 + doc_count) / (1 + df)) + 1

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = self.tokenize(text)
            if not tokens:
                continue
            counts: Dict[int, int] = {}
            for token in tokens:
                idx = self.vocabulary.get(token)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
            for idx, count in counts.items():
                matrix[row, idx] = (1 + math.log(count)) * self.idf[idx]

        return normalize_vectors(matrix)


class SentenceTransformerBackend:
    name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def __init__(self, model_name: str = name) -> None:
        from embedding_repository import EmbeddingRepository

        self.repository = EmbeddingRepository(model_name)
        self.name = model_name

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.repository.get_embeddings(list(texts))


def create_embedding_backend(preferred_model: str = SentenceTransformerBackend.name):
    try:
        return SentenceTransformerBackend(preferred_model)
    except Exception as exc:
        backend = FallbackEmbeddingBackend()
        backend.name = f"{backend.name} ({exc.__class__.__name__}: {exc})"
        return backend


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


class OntologyRagIndex:
    def __init__(self, graph: OntologyGraph, fragments: List[TextFragment], vectors: np.ndarray, backend: Any):
        self.graph = graph
        self.fragments = fragments
        self.vectors = normalize_vectors(vectors)
        self.backend = backend

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not query.strip() or not self.fragments:
            return []

        query_vec = normalize_vectors(self.backend.embed([query]))
        if query_vec.size == 0 or self.vectors.size == 0:
            return []

        scores = np.dot(self.vectors, query_vec[0])
        scores = self._apply_label_boost(query, scores)
        ranked_indexes = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(score=float(scores[idx]), fragment=self.fragments[int(idx)])
            for idx in ranked_indexes
        ]

    def _apply_label_boost(self, query: str, scores: np.ndarray) -> np.ndarray:
        boosted = scores.copy()
        query_norm = normalize_for_match(query)
        query_tokens = set(tokenize_for_match(query))

        for idx, fragment in enumerate(self.fragments):
            label_norm = normalize_for_match(fragment.label)
            label_tokens = set(tokenize_for_match(fragment.label))
            specific_tokens = {token for token in label_tokens if token.isdigit() or len(token) >= 4}

            if label_norm and label_norm in query_norm:
                boosted[idx] += 0.25
            elif specific_tokens and specific_tokens.intersection(query_tokens):
                boosted[idx] += 0.18

        return boosted


def build_index(graph_path: str | Path = "graph.json", include_properties: bool = False) -> OntologyRagIndex:
    graph = load_ontology_graph(graph_path)
    source_nodes = list(graph.nodes_by_uri.values()) if include_properties else graph.main_nodes

    fragments = []
    seen_ids = set()
    for node in source_nodes:
        node_id = node.get("id") or node.get("data", {}).get("uri")
        if not node_id or node_id in seen_ids:
            continue
        text = node_to_text(node, graph)
        if not text.strip():
            continue
        seen_ids.add(node_id)
        fragments.append(
            TextFragment(
                node_id=str(node_id),
                node_uri=node.get("data", {}).get("uri", str(node_id)),
                label=extract_label(node) or graph.label_for_uri(str(node_id)),
                text=text,
            )
        )

    backend = create_embedding_backend()
    backend.fit([fragment.text for fragment in fragments])
    vectors = backend.embed([fragment.text for fragment in fragments])
    return OntologyRagIndex(graph, fragments, vectors, backend)


def build_prompt(question: str, contexts: Sequence[str]) -> str:
    context_text = "\n\n".join(f"[{idx + 1}]\n{text}" for idx, text in enumerate(contexts))
    return (
        "Дай ответ на данный вопрос, используя информацию из текста. "
        "Если в тексте недостаточно данных, укажи это явно.\n\n"
        f"Вопрос: {question}\n\n"
        f"Текст:\n{context_text}\n\n"
        "Ответ:"
    )


class LLMClient:
    _local_model = None
    _local_tokenizer = None
    _local_model_name = None

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        local_model: Optional[str] = None,
        use_local_llm: Optional[bool] = None,
        timeout: int = 60,
    ) -> None:
        self.api_url = api_url or os.getenv("RAG_API_URL")
        self.api_key = api_key or os.getenv("RAG_API_KEY")
        self.model = model or os.getenv("RAG_API_MODEL", "gpt-4o-mini")
        self.local_model = local_model or os.getenv("RAG_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        if use_local_llm is None:
            use_local_llm = os.getenv("RAG_USE_LOCAL_LLM", "1").lower() not in {"0", "false", "no"}
        self.use_local_llm = use_local_llm
        self.max_new_tokens = int(os.getenv("RAG_MAX_NEW_TOKENS", "140"))
        self.timeout = timeout

    def generate(self, question: str, contexts: Sequence[str]) -> LLMResult:
        prompt = build_prompt(question, contexts)

        if self.api_url and self.api_key:
            try:
                import requests

                response = requests.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "Отвечай на русском языке только на основе предоставленного контекста.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                answer = self._extract_api_answer(data)
                return LLMResult(answer=answer, prompt=prompt, used_api=True, mode="api")
            except Exception as exc:
                return LLMResult(
                    answer=f"API генерации недоступен ({exc.__class__.__name__}: {exc}).\n\n{self._fallback_answer(question, contexts)}",
                    prompt=prompt,
                    used_api=False,
                    mode="fallback",
                )

        if self.use_local_llm:
            try:
                return LLMResult(
                    answer=self._generate_local(prompt),
                    prompt=prompt,
                    used_api=False,
                    mode="local",
                )
            except Exception as exc:
                fallback = self._fallback_answer(question, contexts)
                return LLMResult(
                    answer=f"Локальная модель генерации недоступна ({exc.__class__.__name__}: {exc}).\n\n{fallback}",
                    prompt=prompt,
                    used_api=False,
                    mode="fallback",
                )

        return LLMResult(
            answer=self._fallback_answer(question, contexts),
            prompt=prompt,
            used_api=False,
            mode="fallback",
        )

    def _generate_local(self, prompt: str) -> str:
        tokenizer, model = self._get_local_model()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        max_positions = getattr(model.config, "max_position_embeddings", None) or tokenizer.model_max_length
        if max_positions is None or max_positions > 100000:
            max_positions = 1024
        input_limit = max(64, int(max_positions) - self.max_new_tokens - 8)

        model_prompt = self._format_local_prompt(tokenizer, prompt)
        inputs = tokenizer(
            model_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=input_limit,
        )
        input_length = inputs["input_ids"].shape[1]

        output_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0][input_length:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if is_low_quality_generation(answer):
            raise ValueError("local model produced low-quality text")
        return answer or "Локальная модель вернула пустой ответ."

    @staticmethod
    def _format_local_prompt(tokenizer, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты отвечаешь на русском языке. "
                    "Используй только предоставленный контекст. "
                    "Если данных недостаточно, прямо напиши, что данных недостаточно. "
                    "Не выдумывай факты."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt

    def _get_local_model(self):
        if (
            LLMClient._local_model is not None
            and LLMClient._local_tokenizer is not None
            and LLMClient._local_model_name == self.local_model
        ):
            return LLMClient._local_tokenizer, LLMClient._local_model

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.local_model)
        model = AutoModelForCausalLM.from_pretrained(self.local_model)
        LLMClient._local_tokenizer = tokenizer
        LLMClient._local_model = model
        LLMClient._local_model_name = self.local_model
        return tokenizer, model

    @staticmethod
    def _extract_api_answer(data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            if message.get("content"):
                return str(message["content"]).strip()
            if choices[0].get("text"):
                return str(choices[0]["text"]).strip()
        if data.get("text"):
            return str(data["text"]).strip()
        if data.get("answer"):
            return str(data["answer"]).strip()
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _fallback_answer(question: str, contexts: Sequence[str]) -> str:
        extractive = build_extractive_answer(question, contexts)
        if extractive:
            return extractive

        preview = []
        for text in contexts[:5]:
            first_line = next((line for line in text.splitlines() if line.strip()), "")
            if first_line:
                preview.append(first_line)
        joined_preview = "; ".join(preview)
        return (
            "Резервный режим: ответ языковой модели не был сгенерирован. "
            f"Для вопроса «{question}» найдены релевантные фрагменты: {joined_preview}. "
            "Готовый промпт сохранен в результате выполнения."
        )


def is_low_quality_generation(answer: str) -> bool:
    if not answer or len(answer.strip()) < 20:
        return True

    letters = [char for char in answer if char.isalpha()]
    if not letters:
        return True

    allowed = set(string.ascii_letters + string.digits + string.punctuation + " \n\t\r")
    allowed.update("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    odd_chars = [char for char in answer if char not in allowed]
    if len(odd_chars) / max(1, len(answer)) > 0.15:
        return True

    words = re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", answer)
    if len(words) < 4:
        return True

    repeated = re.search(r"(.{1,8})\1{8,}", answer)
    return repeated is not None


def build_extractive_answer(question: str, contexts: Sequence[str]) -> str:
    if not contexts:
        return ""

    question_tokens = set(tokenize_for_match(question))
    target_context = choose_target_context(question_tokens, contexts)
    lines = [line.strip() for line in target_context.splitlines() if line.strip()]
    if not lines:
        return ""

    title = first_value(lines, "Название") or "найденный объект"
    useful_lines = []
    for line in lines:
        if line.startswith("Название:") and useful_lines:
            continue
        if line.startswith("Описание:") and line.endswith("Описание"):
            continue
        if line.startswith("Идентификаторы текстов разметки:"):
            continue
        useful_lines.append(line)

    facts = []
    for line in useful_lines:
        if line.startswith("Название:"):
            continue
        facts.append(line)

    if not facts:
        return ""

    return (
        f"По найденному контексту объект «{title}» имеет следующие сведения:\n"
        + "\n".join(f"- {fact}" for fact in facts[:14])
    )


def choose_target_context(question_tokens: set[str], contexts: Sequence[str]) -> str:
    best_context = contexts[0]
    best_score = -1
    for context in contexts:
        first_title = first_value(context.splitlines(), "Название")
        label_tokens = set(tokenize_for_match(first_title))
        specific_tokens = {token for token in label_tokens if token.isdigit() or len(token) >= 4}
        score = len(specific_tokens.intersection(question_tokens))
        if score > best_score:
            best_context = context
            best_score = score
    return best_context


def first_value(lines: Iterable[str], prefix: str) -> str:
    marker = f"{prefix}:"
    for line in lines:
        if line.strip().startswith(marker):
            return line.split(":", 1)[1].strip()
    return ""


class RagPipeline:
    def __init__(self, index: OntologyRagIndex, llm: Optional[LLMClient] = None) -> None:
        self.index = index
        self.llm = llm or LLMClient()

    def answer(self, question: str, n: int = 5, m: int = 3) -> RagResult:
        first_results = self.index.search(question, top_k=n)
        first_contexts = [result.fragment.text for result in first_results]
        draft = self.llm.generate(question, first_contexts)

        second_results = self.index.search(draft.answer, top_k=m)
        all_results = self._merge_results(first_results, second_results)
        final_contexts = [result.fragment.text for result in all_results]
        final = self.llm.generate(question, final_contexts)

        return RagResult(
            question=question,
            first_results=first_results,
            second_results=second_results,
            all_results=all_results,
            draft_answer=draft.answer,
            final_answer=final.answer,
            draft_prompt=draft.prompt,
            final_prompt=final.prompt,
            used_api=draft.used_api or final.used_api,
            generation_mode=final.mode,
        )

    @staticmethod
    def _merge_results(first: Iterable[SearchResult], second: Iterable[SearchResult]) -> List[SearchResult]:
        merged: List[SearchResult] = []
        seen = set()
        for result in list(first) + list(second):
            node_id = result.fragment.node_id
            if node_id in seen:
                continue
            seen.add(node_id)
            merged.append(result)
        return merged
