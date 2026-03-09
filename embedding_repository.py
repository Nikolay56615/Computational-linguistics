from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRepository:
    """
    Репозиторий для работы с эмбеддингами фрагментов текста.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def get_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Разбиение текста на фрагменты (chunks).

        Параметры:
            text (str): исходный текст
            chunk_size (int): максимальная длина одного чанка в символах
            overlap (int): примерный размер перекрытия между чанками

        Возвращает:
            List[str]: список фрагментов текста
        """
        if not text or not text.strip():
            return []

        words = text.split()
        chunks = []
        current_chunk = []
        current_len = 0

        for word in words:
            word_len = len(word) + 1  # пробел

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

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Генерация эмбеддингов для списка текстов.
        Возвращает матрицу эмбеддингов (numpy array).
        """
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]
        if not cleaned_texts:
            return np.array([]).reshape(0, 768)

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

        Возвращает значение от -1 до 1.
        """
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)

        return float(cosine_similarity(emb1, emb2)[0][0])

    def embed_text(self, text: str) -> np.ndarray:
        """
        Быстрый способ получить эмбеддинг одного текста.
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
        Сравнить один запрос сразу со списком текстов.
        """
        if not query_text or not query_text.strip() or not texts:
            return []

        query_emb = self.embed_text(query_text)
        texts_emb = self.get_embeddings(texts)

        if query_emb.size == 0 or texts_emb.size == 0:
            return []

        similarities = cosine_similarity(query_emb.reshape(1, -1), texts_emb)[0]
        return [float(score) for score in similarities]