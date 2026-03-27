from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

COLLECTION_NAME = "forecastiq_briefings"
STORE_DIR = Path("outputs/rag_store")


def _chunk_briefing(text: str) -> list[tuple[str, str]]:
    """Split briefing markdown by ## sections.

    Returns a list of (section_title, chunk_text) tuples.
    The preamble before any ## heading is titled 'intro'.
    """
    parts = text.split("\n## ")
    chunks: list[tuple[str, str]] = []
    for i, part in enumerate(parts):
        if i == 0:
            title = "intro"
            body = part.strip()
        else:
            lines = part.split("\n", 1)
            title = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
        if body:
            chunks.append((title, body))
    return chunks


class RAGStore:
    """Local ChromaDB-backed vector store for ForecastIQ operational briefings.

    Uses ChromaDB's built-in ONNX MiniLM-L6-v2 embedding function — no API key,
    no Python transformers library required.
    Storage is file-based at outputs/rag_store/ (persistent across restarts).
    All public methods degrade gracefully if dependencies are missing or the store errors.
    """

    def __init__(self) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(STORE_DIR))
        self._ef = DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_briefing(self, run_id: str, briefing_text: str, metadata: dict[str, Any]) -> int:
        """Chunk briefing_text by ## section, embed each chunk, upsert into ChromaDB.

        Args:
            run_id: Unique run identifier (e.g. '2025012500').
            briefing_text: Full markdown text of the operational briefing.
            metadata: Dict with at minimum run_id, risk_level, init_time, timestamp.

        Returns:
            Number of unique run_ids stored after this call.
        """
        chunks = _chunk_briefing(briefing_text)
        if not chunks:
            logger.warning("RAGStore.add_briefing: no chunks extracted for run_id=%s", run_id)
            return self.count_runs()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for idx, (section_title, body) in enumerate(chunks):
            doc_id = f"{run_id}_{idx}"
            chunk_meta = {
                "run_id": run_id,
                "section_title": section_title,
                "risk_level": str(metadata.get("risk_level", "")),
                "init_time": str(metadata.get("init_time", "")),
                "timestamp": str(metadata.get("timestamp", "")),
                "chunk_idx": idx,
            }
            ids.append(doc_id)
            documents.append(body)
            metadatas.append(chunk_meta)

        # Upsert is idempotent — re-running Phase 3 for same run_id won't duplicate
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        total = self.count_runs()
        logger.info("RAGStore: stored %d chunks for run_id=%s (total runs: %d)", len(chunks), run_id, total)
        return total

    def retrieve_relevant(self, query: str, n_results: int = 3) -> list[dict[str, Any]]:
        """Embed query and return top n_results similar briefing chunks.

        Returns an empty list if the store is empty or any error occurs.
        Each result dict: {"text": str, "metadata": dict, "distance": float}
        """
        try:
            total = self._collection.count()
            if total == 0:
                return []
            k = min(n_results, total)
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            output: list[dict[str, Any]] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                output.append({"text": doc, "metadata": meta, "distance": dist})
            return output
        except Exception as exc:
            logger.warning("RAGStore.retrieve_relevant failed: %s", exc)
            return []

    def count(self) -> int:
        """Return total number of chunks stored."""
        try:
            return self._collection.count()
        except Exception:
            return 0

    def count_runs(self) -> int:
        """Return number of unique run_ids in the collection."""
        try:
            total = self._collection.count()
            if total == 0:
                return 0
            result = self._collection.get(include=["metadatas"])
            metas = result.get("metadatas") or []
            unique_runs = {m.get("run_id") for m in metas if m.get("run_id")}
            return len(unique_runs)
        except Exception:
            return 0

    def get_langchain_retriever(self, k: int = 3):
        """Return a LangChain BaseRetriever backed by this RAGStore.

        Uses retrieve_relevant() directly — same ONNX embeddings, no adapter
        format issues. retrieve_relevant() is kept intact for backward compat.
        """
        from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever

        _retrieve = self.retrieve_relevant  # capture bound method

        class _RAGStoreRetriever(BaseRetriever):
            """Thin LangChain retriever wrapper around RAGStore.retrieve_relevant."""

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun,
            ) -> list[Document]:
                chunks = _retrieve(query, n_results=k)
                return [
                    Document(
                        page_content=chunk["text"],
                        metadata=chunk.get("metadata", {}),
                    )
                    for chunk in chunks
                ]

        return _RAGStoreRetriever()
