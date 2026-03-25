import json
import logging
from typing import List, Tuple, Dict, Union, Optional, Any
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.reranking import LLMReranker
from src.local_models import LocalEmbeddingModel, LocalJinaReranker, DEFAULT_LOCAL_EMBEDDING_MODEL, DEFAULT_LOCAL_RERANKER_MODEL
from src.lexical_retrieval import select_novel_results, tokenize_for_bm25

_log = logging.getLogger(__name__)


def reciprocal_rank_fuse(
    ranked_lists: List[Tuple[str, List[Dict[str, Any]], float]],
    fusion_k: int = 60,
) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}

    max_rrf_score = sum(weight / (fusion_k + 1) for _, _, weight in ranked_lists if weight > 0)
    max_rrf_score = max_rrf_score or 1.0

    for source_name, results, weight in ranked_lists:
        for rank, item in enumerate(results, start=1):
            ref = item.get("ref") or f"{item.get('page')}::{item.get('text', '')[:80]}"
            record = fused.setdefault(ref, item.copy())
            record.setdefault("retrieval_sources", [])
            record["retrieval_sources"].append(source_name)
            record.setdefault("source_scores", {})
            record["source_scores"][source_name] = round(float(item.get("distance", 0.0)), 4)
            record.setdefault("rrf_score", 0.0)
            record["rrf_score"] += weight / (fusion_k + rank)

    fused_results = list(fused.values())
    for item in fused_results:
        item["retrieval_sources"] = sorted(set(item.get("retrieval_sources", [])))
        item["distance"] = round(float(item["rrf_score"]) / max_rrf_score, 4)

    fused_results.sort(key=lambda item: item["distance"], reverse=True)
    return fused_results

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()

    def _load_dbs(self):
        all_dbs = []
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        bm25_db_files = {db_path.stem: db_path for db_path in self.bm25_db_dir.glob('*.pkl')}

        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in bm25_db_files:
                _log.warning(f"No matching BM25 DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue

            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue

            try:
                with open(bm25_db_files[stem], 'rb') as f:
                    bm25_index = pickle.load(f)
            except Exception as e:
                _log.error(f"Error reading BM25 DB for {document_path.name}: {e}")
                continue

            all_dbs.append(
                {
                    "name": stem,
                    "bm25_index": bm25_index,
                    "document": document,
                }
            )
        return all_dbs
        
    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break

        if target_report is None:
            raise ValueError(f"No report found with '{company_name}' company name.")

        document = target_report["document"]
        bm25_index = target_report["bm25_index"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        # Get BM25 scores for the query
        tokenized_query = tokenize_for_bm25(query)
        scores = bm25_index.get_scores(tokenized_query)
        
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages = set()
        
        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        
        return retrieval_results



class VectorRetriever:
    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
        embedding_provider: str = "openai",
        embedding_model: str = None,
    ):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model or (
            "text-embedding-3-large" if embedding_provider == "openai" else DEFAULT_LOCAL_EMBEDDING_MODEL
        )
        self.all_dbs = self._load_dbs()
        self.llm = self._set_up_llm() if self.embedding_provider == "openai" else None
        self.local_embedder = LocalEmbeddingModel(self.embedding_model) if self.embedding_provider == "local" else None

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm
    
    @staticmethod
    def set_up_llm():
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    def _load_dbs(self):
        all_dbs = []
        # Get list of JSON document paths
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob('*.faiss')}
        
        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            
            # Validate that the document meets the expected schema
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue
            
            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
                
            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def _get_query_embedding(self, query: str) -> np.ndarray:
        if self.embedding_provider == "openai":
            embedding = self.llm.embeddings.create(
                input=query,
                model=self.embedding_model
            )
            return np.array(embedding.data[0].embedding, dtype=np.float32)
        if self.embedding_provider == "local":
            return np.array(self.local_embedder.embed_query(query), dtype=np.float32)
        raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Tuple[str, float]]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.error(f"Report '{report.get('name')}' is missing 'metainfo'!")
                raise ValueError(f"Report '{report.get('name')}' is missing 'metainfo'!")
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        actual_top_n = min(top_n, len(chunks))
        
        embedding_array = self._get_query_embedding(query).reshape(1, -1)
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
    
        retrieval_results = []
        seen_pages = set()
        
        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
            
        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
            
        return all_pages


class HybridRetriever:
    def __init__(
        self,
        vector_db_dir: Path = None,
        documents_dir: Path = None,
        vector_retriever: VectorRetriever = None,
        bm25_db_dir: Path = None,
        use_bm25_db: bool = False,
        bm25_auto: bool = False,
        enable_reranking: bool = True,
        embedding_provider: str = "openai",
        embedding_model: str = None,
        reranker_provider: str = "openai",
        reranker_model: str = None,
        vector_weight: float = 1.0,
        bm25_weight: float = 0.7,
        fusion_k: int = 60,
        adaptive_bm25_weight: float = 0.25,
        adaptive_bm25_max_novel: int = 4,
    ):
        self.vector_retriever = vector_retriever or VectorRetriever(
            vector_db_dir,
            documents_dir,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        self.documents_dir = documents_dir
        self.use_bm25_db = use_bm25_db
        self.bm25_auto = bm25_auto
        self.enable_reranking = enable_reranking
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.fusion_k = fusion_k
        self.adaptive_bm25_weight = adaptive_bm25_weight
        self.adaptive_bm25_max_novel = adaptive_bm25_max_novel
        self.bm25_retriever: Optional[BM25Retriever] = None
        if self.use_bm25_db and bm25_db_dir is not None and Path(bm25_db_dir).exists():
            self.bm25_retriever = BM25Retriever(Path(bm25_db_dir), Path(documents_dir))
        elif self.use_bm25_db:
            _log.warning("BM25 retrieval requested but BM25 index directory is missing; falling back to dense-only retrieval.")
        self.reranker = self._build_reranker(reranker_provider, reranker_model) if self.enable_reranking else None

    @staticmethod
    def _build_reranker(reranker_provider: str, reranker_model: str):
        if reranker_provider == "local_jina":
            return LocalJinaReranker(reranker_model or DEFAULT_LOCAL_RERANKER_MODEL)
        return LLMReranker(model_name=reranker_model)

    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: Optional[List[Dict[str, Any]]],
        bm25_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not bm25_results:
            return vector_results
        return reciprocal_rank_fuse(
            [
                ("vector", vector_results, self.vector_weight),
                ("bm25", bm25_results, self.bm25_weight if bm25_weight is None else bm25_weight),
            ],
            fusion_k=self.fusion_k,
        )
        
    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False,
        bm25_sample_size: int = None,
    ) -> List[Dict]:
        """
        Retrieve and rerank documents using hybrid approach.
        
        Args:
            company_name: Name of the company to search documents for
            query: Search query
            llm_reranking_sample_size: Number of initial results to retrieve from vector DB
            documents_batch_size: Number of documents to analyze in one LLM prompt
            top_n: Number of final results to return after reranking
            llm_weight: Weight given to LLM scores (0-1)
            return_parent_pages: Whether to return full pages instead of chunks
            
        Returns:
            List of reranked document dictionaries with scores
        """
        # Get initial results from vector retriever
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )

        bm25_results = None
        should_use_bm25 = self.bm25_retriever is not None
        if should_use_bm25 and self.bm25_retriever is not None:
            bm25_results = self.bm25_retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=bm25_sample_size or llm_reranking_sample_size,
                return_parent_pages=return_parent_pages,
            )
            if self.bm25_auto:
                bm25_results = select_novel_results(
                    primary_results=vector_results,
                    secondary_results=bm25_results,
                    max_new_results=min(
                        self.adaptive_bm25_max_novel,
                        bm25_sample_size or llm_reranking_sample_size,
                    ),
                )

        fused_results = self._fuse_results(
            vector_results,
            bm25_results,
            bm25_weight=self.adaptive_bm25_weight if self.bm25_auto else None,
        )

        if self.reranker is None:
            return fused_results[:top_n]

        # Rerank results using LLM
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=fused_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        
        return reranked_results[:top_n]
