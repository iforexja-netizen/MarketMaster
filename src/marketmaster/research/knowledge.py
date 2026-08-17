"""
Knowledge Retrieval — Document indexing and similarity search.

Provides a lightweight document retrieval system for the research pipeline:
1. Index documents (news, transcripts, filings) with metadata
2. Search by keyword, entity, or semantic similarity (TF-IDF based)
3. Retrieve context-relevant documents for agent analysis

This is a simplified, dependency-free implementation using TF-IDF vectors
and cosine similarity. For production, this could be upgraded to use
embeddings + a vector database (Pinecone, Weaviate, etc.), but the
TF-IDF approach is transparent, fast, and has no external dependencies.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from collections import Counter
import math
import re


@dataclass
class Document:
    """A retrievable document in the knowledge base."""
    doc_id: str
    title: str
    content: str
    doc_type: str  # "news", "transcript", "filing", "research"
    security_id: Optional[int] = None
    symbols: list[str] = field(default_factory=list)
    date: Optional[date] = None
    source: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search result."""
    document: Document
    score: float
    snippet: str = ""


class KnowledgeBase:
    """
    In-memory knowledge base with TF-IDF search.

    Usage:
        kb = KnowledgeBase()
        kb.add_document(Document(...))
        results = kb.search("revenue growth", top_k=5)
    """

    def __init__(self):
        self._documents: dict[str, Document] = {}
        self._tf: dict[str, dict[str, int]] = {}  # doc_id -> {term: freq}
        self._df: dict[str, int] = {}  # term -> document frequency
        self._idf: dict[str, float] = {}  # term -> inverse document frequency
        self._doc_norms: dict[str, float] = {}  # doc_id -> vector norm

    def add_document(self, doc: Document) -> None:
        """Add or update a document in the knowledge base."""
        # Remove existing if updating
        if doc.doc_id in self._documents:
            self._remove_from_index(doc.doc_id)

        self._documents[doc.doc_id] = doc

        # Tokenize and compute term frequencies
        tokens = self._tokenize(doc.title + " " + doc.content)
        tf = Counter(tokens)
        self._tf[doc.doc_id] = dict(tf)

        # Update document frequencies
        for term in tf:
            self._df[term] = self._df.get(term, 0) + 1

        # Recompute IDF (cheap for small collections)
        self._recompute_idf()
        self._recompute_norm(doc.doc_id)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the knowledge base."""
        if doc_id in self._documents:
            self._remove_from_index(doc_id)
            del self._documents[doc_id]
            self._recompute_idf()

    def _remove_from_index(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id not in self._tf:
            return
        for term in self._tf[doc_id]:
            self._df[term] = max(0, self._df.get(term, 0) - 1)
            if self._df[term] == 0:
                self._df.pop(term, None)
        del self._tf[doc_id]
        self._doc_norms.pop(doc_id, None)

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_type: Optional[str] = None,
        security_id: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Search the knowledge base for documents matching the query.

        Args:
            query: Search query text
            top_k: Maximum number of results
            doc_type: Filter by document type
            security_id: Filter by security

        Returns list of SearchResult ranked by relevance.
        """
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Compute query TF-IDF vector
        query_tf = Counter(query_tokens)
        query_vec = {}
        for term, freq in query_tf.items():
            idf = self._idf.get(term, 0.0)
            if idf > 0:
                query_vec[term] = freq * idf

        query_norm = math.sqrt(sum(v ** 2 for v in query_vec.values()))
        if query_norm == 0:
            return []

        # Compute cosine similarity for each document
        results = []
        for doc_id, doc in self._documents.items():
            # Apply filters
            if doc_type and doc.doc_type != doc_type:
                continue
            if security_id and doc.security_id != security_id:
                continue

            doc_vec = self._doc_vector(doc_id)
            doc_norm = self._doc_norms.get(doc_id, 0.0)
            if doc_norm == 0:
                continue

            # Cosine similarity
            dot_product = sum(query_vec.get(t, 0) * doc_vec.get(t, 0) for t in query_vec)
            similarity = dot_product / (query_norm * doc_norm)

            if similarity > 0:
                snippet = self._extract_snippet(doc.content, query_tokens)
                results.append(SearchResult(
                    document=doc,
                    score=float(similarity),
                    snippet=snippet,
                ))

        # Sort by score descending, take top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get_by_symbol(
        self,
        symbol: str,
        top_k: int = 20,
        doc_type: Optional[str] = None,
    ) -> list[Document]:
        """Get all documents mentioning a symbol."""
        results = []
        for doc in self._documents.values():
            if symbol.upper() in [s.upper() for s in doc.symbols]:
                if doc_type and doc.doc_type != doc_type:
                    continue
                results.append(doc)

        # Sort by date descending if available
        results.sort(key=lambda d: d.date or date.min, reverse=True)
        return results[:top_k]

    def count(self) -> int:
        """Total number of documents."""
        return len(self._documents)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase terms."""
        return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

    def _recompute_idf(self) -> None:
        """Recompute IDF for all terms."""
        n_docs = len(self._documents)
        if n_docs == 0:
            self._idf = {}
            return
        self._idf = {}
        for term, df in self._df.items():
            # Standard IDF: ln(N / df) + 1
            self._idf[term] = math.log(n_docs / df) + 1.0

    def _doc_vector(self, doc_id: str) -> dict[str, float]:
        """Get the TF-IDF vector for a document."""
        if doc_id not in self._tf:
            return {}
        vec = {}
        for term, freq in self._tf[doc_id].items():
            idf = self._idf.get(term, 0.0)
            vec[term] = freq * idf
        return vec

    def _recompute_norm(self, doc_id: str) -> None:
        """Compute and cache the L2 norm of a document vector."""
        vec = self._doc_vector(doc_id)
        self._doc_norms[doc_id] = math.sqrt(sum(v ** 2 for v in vec.values()))

    def _extract_snippet(self, content: str, query_terms: list[str], max_chars: int = 200) -> str:
        """Extract a relevant snippet from the document content."""
        # Find the first occurrence of any query term
        content_lower = content.lower()
        best_pos = len(content)
        for term in query_terms:
            pos = content_lower.find(term.lower())
            if pos != -1 and pos < best_pos:
                best_pos = pos

        # Extract surrounding context
        start = max(0, best_pos - 50)
        end = min(len(content), best_pos + max_chars)
        snippet = content[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        return snippet
