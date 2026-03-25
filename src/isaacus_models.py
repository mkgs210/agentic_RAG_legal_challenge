from __future__ import annotations

import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import requests
from dotenv import load_dotenv


def _load_dotenvs() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    load_dotenv(repo_root / "env")
    load_dotenv()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _PickleVectorCache:
    def __init__(self, path: Path, as_arrays: bool = True):
        self.path = path
        self.as_arrays = as_arrays
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, np.ndarray] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("rb") as handle:
            payload = pickle.load(handle)
        if not self.as_arrays:
            return payload
        return {key: np.asarray(value, dtype=np.float32) for key, value in payload.items()}

    def save(self) -> None:
        with self.path.open("wb") as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class IsaacusEmbeddingModel:
    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        _load_dotenvs()
        self.model_name = model_name or os.getenv("ISAACUS_EMBED_MODEL", "kanon-2-embedder")
        self.base_url = (base_url or os.getenv("ISAACUS_BASE_URL", "https://api.isaacus.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("ISAACUS_API_KEY")
        if not self.api_key:
            raise ValueError("Missing ISAACUS_API_KEY")
        self.batch_size = max(1, int(batch_size))
        self.normalize_embeddings = normalize_embeddings
        cache_root = Path(cache_dir or Path("artifacts") / "isaacus_cache")
        self.cache = _PickleVectorCache(cache_root / f"{_slugify(self.model_name)}_embeddings.pkl", as_arrays=True)
        self.cache_key = f"isaacus::{self.model_name}"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.normalize_embeddings:
            return embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (embeddings / norms).astype(np.float32)

    def _request_embeddings(self, texts: List[str]) -> np.ndarray:
        response = self.session.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model_name, "texts": texts},
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("embeddings", [])
        rows = sorted(rows, key=lambda item: item.get("index", 0))
        embeddings = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        if embeddings.shape[0] != len(texts):
            raise ValueError(
                f"Isaacus embedding count mismatch: expected {len(texts)}, got {embeddings.shape[0]}"
            )
        return self._normalize(embeddings)

    def _embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts = [text or "" for text in texts]
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        missing: List[str] = []
        missing_keys: List[str] = []
        ordered_keys: List[str] = []
        for text in texts:
            key = _hash_text(text)
            ordered_keys.append(key)
            if key not in self.cache.data and key not in missing_keys:
                missing_keys.append(key)
                missing.append(text)

        if missing:
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                batch_embeddings = self._request_embeddings(batch)
                for text, vector in zip(batch, batch_embeddings):
                    self.cache.data[_hash_text(text)] = vector.astype(np.float32)
            self.cache.save()

        return np.vstack([self.cache.data[key] for key in ordered_keys]).astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_texts([text])[0]


class IsaacusReranker:
    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        batch_size: int = 32,
    ):
        _load_dotenvs()
        self.model_name = model_name or os.getenv("ISAACUS_RERANK_MODEL", "kanon-2-reranker")
        self.base_url = (base_url or os.getenv("ISAACUS_BASE_URL", "https://api.isaacus.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("ISAACUS_API_KEY")
        if not self.api_key:
            raise ValueError("Missing ISAACUS_API_KEY")
        self.batch_size = max(1, int(batch_size))
        cache_root = Path(cache_dir or Path("artifacts") / "isaacus_cache")
        self.cache = _PickleVectorCache(cache_root / f"{_slugify(self.model_name)}_pairs.pkl", as_arrays=True)
        self.cache_key = f"isaacus::{self.model_name}"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    @staticmethod
    def _pair_key(query: str, text: str) -> str:
        return f"{_hash_text(query)}::{_hash_text(text)}"

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            return scores
        if scores.min() < 0 or scores.max() > 1:
            scores = 1 / (1 + np.exp(-scores))
        if scores.size == 1:
            return np.ones_like(scores, dtype=np.float32)
        score_min = float(scores.min())
        score_max = float(scores.max())
        if score_max - score_min < 1e-8:
            return np.ones_like(scores, dtype=np.float32)
        return ((scores - score_min) / (score_max - score_min)).astype(np.float32)

    def _request_scores(self, query: str, texts: List[str]) -> np.ndarray:
        response = self.session.post(
            f"{self.base_url}/rerankings",
            json={"model": self.model_name, "query": query, "texts": texts},
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        by_index = {item["index"]: float(item["score"]) for item in payload.get("results", [])}
        return np.asarray([by_index[index] for index in range(len(texts))], dtype=np.float32)

    def rerank_documents(
        self,
        query: str,
        documents: list,
        documents_batch_size: int = 4,
        llm_weight: float = 0.7,
    ):
        texts = [doc.get("text", "") for doc in documents]
        missing_texts: List[str] = []
        missing_keys: List[str] = []
        for text in texts:
            key = self._pair_key(query, text)
            if key not in self.cache.data and key not in missing_keys:
                missing_keys.append(key)
                missing_texts.append(text)

        if missing_texts:
            for start in range(0, len(missing_texts), self.batch_size):
                batch = missing_texts[start : start + self.batch_size]
                batch_scores = self._request_scores(query, batch)
                for text, score in zip(batch, batch_scores):
                    self.cache.data[self._pair_key(query, text)] = np.asarray([score], dtype=np.float32)
            self.cache.save()

        raw_scores = np.asarray(
            [float(self.cache.data[self._pair_key(query, text)][0]) for text in texts],
            dtype=np.float32,
        )
        normalized_scores = self._normalize_scores(raw_scores)

        vector_weight = 1 - llm_weight
        reranked = []
        for doc, raw_score, normalized_score in zip(documents, raw_scores, normalized_scores):
            doc_with_score = doc.copy()
            doc_with_score["relevance_score"] = round(float(raw_score), 6)
            doc_with_score["normalized_relevance_score"] = round(float(normalized_score), 4)
            doc_with_score["combined_score"] = round(
                llm_weight * float(normalized_score) + vector_weight * float(doc.get("distance", 0.0)),
                4,
            )
            reranked.append(doc_with_score)

        reranked.sort(key=lambda item: item["combined_score"], reverse=True)
        return reranked


class IsaacusEnricher:
    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        overflow_strategy: str = "auto",
    ):
        _load_dotenvs()
        self.model_name = model_name or os.getenv("ISAACUS_ENRICH_MODEL", "kanon-2-enricher")
        self.base_url = (base_url or os.getenv("ISAACUS_BASE_URL", "https://api.isaacus.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("ISAACUS_API_KEY")
        if not self.api_key:
            raise ValueError("Missing ISAACUS_API_KEY")
        self.overflow_strategy = overflow_strategy
        cache_root = Path(cache_dir or Path("artifacts") / "isaacus_cache")
        self.cache = _PickleVectorCache(cache_root / f"{_slugify(self.model_name)}_documents.pkl", as_arrays=False)
        self.cache_key = f"isaacus::{self.model_name}"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _request_enrichments(self, texts: List[str]) -> List[Dict[str, Any]]:
        response = self.session.post(
            f"{self.base_url}/enrichments",
            json={
                "model": self.model_name,
                "texts": texts if len(texts) > 1 else texts[0],
                "overflow_strategy": self.overflow_strategy,
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        documents = [item["document"] for item in results]
        if len(documents) != len(texts):
            raise ValueError(f"Isaacus enrichment count mismatch: expected {len(texts)}, got {len(documents)}")
        return documents

    def enrich_texts(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        texts = [text or "" for text in texts]
        if not texts:
            return []

        ordered_keys: List[str] = []
        missing: List[str] = []
        missing_keys: List[str] = []
        for text in texts:
            key = _hash_text(text)
            ordered_keys.append(key)
            if key not in self.cache.data and key not in missing_keys:
                missing_keys.append(key)
                missing.append(text)

        if missing:
            documents = self._request_enrichments(missing)
            for text, document in zip(missing, documents):
                self.cache.data[_hash_text(text)] = document
            self.cache.save()

        return [self.cache.data[key] for key in ordered_keys]

    def enrich_text(self, text: str) -> Dict[str, Any]:
        return self.enrich_texts([text])[0]
