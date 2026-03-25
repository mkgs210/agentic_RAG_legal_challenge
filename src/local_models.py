import os
from functools import lru_cache
from typing import Iterable, List, Tuple
from copy import deepcopy

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
DEFAULT_LOCAL_RERANKER_MODEL = os.getenv("LOCAL_RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
DEFAULT_LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _embedding_device() -> str:
    return os.getenv("LOCAL_EMBEDDING_DEVICE", "cuda" if _cuda_available() else "cpu")


def _reranker_device() -> str:
    return os.getenv("LOCAL_RERANKER_DEVICE", "cuda" if _cuda_available() else "cpu")


def _generation_dtype():
    if not _cuda_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@lru_cache(maxsize=None)
def _load_sentence_transformer(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=None)
def _load_cross_encoder(model_name: str, device: str) -> CrossEncoder:
    automodel_args = {}
    tokenizer_args = {}
    if _cuda_available():
        automodel_args["dtype"] = _generation_dtype()
    if model_name == "jinaai/jina-reranker-v2-base-multilingual":
        tokenizer_args["fix_mistral_regex"] = True
    return CrossEncoder(
        model_name,
        device=device,
        trust_remote_code=True,
        automodel_args=automodel_args,
        tokenizer_args=tokenizer_args,
    )


@lru_cache(maxsize=None)
def _load_chat_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs = {"trust_remote_code": True}
    if _cuda_available():
        model_kwargs["device_map"] = os.getenv("LOCAL_LLM_DEVICE_MAP", "auto")
        model_kwargs["dtype"] = _generation_dtype()

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


class LocalEmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL):
        self.model_name = model_name
        self.device = _embedding_device()
        self.model = _load_sentence_transformer(self.model_name, self.device)

    def _prepare_inputs(self, texts: Iterable[str], is_query: bool) -> List[str]:
        prepared = list(texts)
        if "e5" not in self.model_name.lower():
            return prepared

        prefix = "query: " if is_query else "passage: "
        return [prefix + text for text in prepared]

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            self._prepare_inputs(texts, is_query=False),
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            self._prepare_inputs([text], is_query=True),
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)


class LocalJinaReranker:
    def __init__(self, model_name: str = DEFAULT_LOCAL_RERANKER_MODEL):
        self.model_name = model_name
        self.device = _reranker_device()
        self.model = _load_cross_encoder(self.model_name, self.device)

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        if scores.min() < 0 or scores.max() > 1:
            scores = 1 / (1 + np.exp(-scores))
        return scores

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.model.predict(
            pairs,
            batch_size=max(1, min(len(pairs), 8)),
            show_progress_bar=False,
        )
        scores = self._normalize_scores(np.asarray(scores, dtype=np.float32).reshape(-1))

        vector_weight = 1 - llm_weight
        reranked = []
        for doc, score in zip(documents, scores):
            doc_with_score = doc.copy()
            doc_with_score["relevance_score"] = round(float(score), 4)
            doc_with_score["combined_score"] = round(
                llm_weight * float(score) + vector_weight * doc["distance"],
                4,
            )
            reranked.append(doc_with_score)

        reranked.sort(key=lambda item: item["combined_score"], reverse=True)
        return reranked


class LocalChatModel:
    def __init__(self, model_name: str = DEFAULT_LOCAL_LLM_MODEL):
        self.model_name = model_name
        self.tokenizer, self.model = _load_chat_model(self.model_name)

    def generate(
        self,
        system_content: str,
        human_content: str,
        temperature: float = 0.0,
        max_new_tokens: int = 1536,
    ) -> Tuple[str, dict]:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generation_config = deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.do_sample = temperature > 0
        if temperature > 0:
            generation_config.temperature = temperature
        else:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=generation_config)

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        usage = {
            "model": self.model_name,
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(generated.shape[0]),
        }
        return text, usage
