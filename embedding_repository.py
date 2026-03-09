import re
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRepository:
    """
    Репозиторий для работы с эмбеддингами и семантическим поиском по тексту.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _merge_units(units: List[str], chunk_size: int, overlap: int) -> List[str]:
        """
        Собирает список текстовых единиц (слов, предложений, абзацев) в чанки.
        """
        if not units:
            return []

        chunks = []
        current_chunk = []

        for unit in units:
            candidate_units = current_chunk + [unit]
            candidate_text = " ".join(candidate_units).strip()

            if len(candidate_text) <= chunk_size or not current_chunk:
                current_chunk.append(unit)
            else:
                chunks.append(" ".join(current_chunk).strip())

                if overlap > 0:
                    overlap_units = []
                    overlap_len = 0

                    for prev_unit in reversed(current_chunk):
                        unit_len = len(prev_unit) + 1
                        if overlap_len + unit_len > overlap and overlap_units:
                            break
                        overlap_units.insert(0, prev_unit)
                        overlap_len += unit_len

                    current_chunk = overlap_units + [unit]
                else:
                    current_chunk = [unit]

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        return chunks

    @staticmethod
    def get_chunks(text: str, chunk_size: int = 512, overlap: int = 50, split_by: str = "word") -> List[str]:
        """
        Разбиение текста на чанки.

        Параметры:
        text (str): исходный текст
        chunk_size (int): максимальная длина чанка в символах
        overlap (int): перекрытие между чанками в символах
        split_by (str): способ разбиения:
            - "word"      -> по словам
            - "sentence"  -> по предложениям
            - "paragraph" -> по абзацам

        Возвращает:
            List[str]: список чанков
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        if split_by == "paragraph":
            units = [p.strip() for p in text.split("\n\n") if p.strip()]
            return EmbeddingRepository._merge_units(units, chunk_size, overlap)

        if split_by == "sentence":
            sentences = re.split(r'(?<=[.!?])\s+', text)
            units = [s.strip() for s in sentences if s.strip()]
            return EmbeddingRepository._merge_units(units, chunk_size, overlap)

        if split_by == "word":
            words = text.split()
            chunks = []
            current_chunk = []
            current_len = 0

            for word in words:
                word_len = len(word) + 1 # пробел

                if current_len + word_len > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))

                    overlap_words = max(0, overlap // 5)
                    current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                    current_len = sum(len(w) + 1 for w in current_chunk)

                current_chunk.append(word)
                current_len += word_len

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks
        return []

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Генерация эмбеддингов для списка текстов.
        """
        if isinstance(texts, str):
            texts = [texts]

        cleaned_texts = [text.strip() for text in texts if text and text.strip()]
        if not cleaned_texts:
            dim = int(self.model.encode(["test"], convert_to_numpy=True, normalize_embeddings=True).shape[1])
            return np.array([]).reshape(0, dim)

        embeddings = self.model.encode(
            cleaned_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings

    @staticmethod
    def cos_compare(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Вычисление косинусного сходства между двумя эмбеддингами.
        """
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)

        return float(cosine_similarity(emb1, emb2)[0][0])

    def embed_text(self, text: str) -> np.ndarray:
        """
        Получить эмбеддинг одного текста.
        """
        if not text or not text.strip():
            return np.array([])

        return self.get_embeddings([text])[0]

    def compare_texts(self, text1: str, text2: str) -> float:
        """
        Сравнить два текста напрямую.
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)

        if emb1.size == 0 or emb2.size == 0:
            return 0.0

        return self.cos_compare(emb1, emb2)

    def batch_compare(self, query_text: str, texts: List[str]) -> List[float]:
        """
        Сравнить один запрос со списком текстов.
        """
        if not query_text or not query_text.strip() or not texts:
            return []

        query_emb = self.embed_text(query_text)
        texts_emb = self.get_embeddings(texts)

        if query_emb.size == 0 or texts_emb.size == 0:
            return []

        similarities = cosine_similarity(query_emb.reshape(1, -1), texts_emb)[0]
        return [float(score) for score in similarities]

    def find_relevant_chunks(self, query_text: str, chunks: List[str], top_k: int = 3) -> List[Dict]:
        """
        Найти наиболее релевантные чанки для запроса.
        """
        scores = self.batch_compare(query_text, chunks)
        if not scores:
            return []

        results = [
            {
                "index": i,
                "chunk": chunk,
                "similarity": float(score)
            }
            for i, (chunk, score) in enumerate(zip(chunks, scores))
        ]

        results.sort(key=lambda item: item["similarity"], reverse=True)
        return results[:top_k]
