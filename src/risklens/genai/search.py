"""Stage 8 (part 2) - embeddings and semantic search.

The problem this solves
-----------------------
A fraud analyst opens an alert and asks: "have we seen this before?"

Keyword search fails at that question. A case described as "overnight
transaction on an unrecognised device" will not match a query for "night-time
purchase from a new browser", even though they mean the same thing.

Semantic search matches on MEANING rather than on shared words.

How it works
------------
1. An embedding model converts each case narrative into a vector of 384
   numbers that encodes its meaning.
2. Texts with similar meaning end up close together in that 384-dimensional
   space.
3. To search, embed the query and find its nearest neighbours.

Tiny example of what "close" means
----------------------------------
    "cat"      -> [0.2, 0.9, 0.1, ...]
    "kitten"   -> [0.2, 0.8, 0.1, ...]   <- very close to "cat"
    "railway"  -> [0.9, 0.1, 0.7, ...]   <- far from both

No word is shared between "cat" and "kitten", yet the vectors are adjacent.
That is the entire value proposition.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# all-MiniLM-L6-v2: 384 dimensions, ~90 MB, runs on CPU in milliseconds.
#
# Why this model rather than a larger one:
#   * It is the standard baseline for sentence similarity - well understood,
#     easy to justify in an interview.
#   * 384 dims keeps the index small and search instant.
#   * It runs locally, so no transaction data leaves the machine. For
#     financial data that is a hard requirement, not a preference.
#
# This IS a BERT-family transformer, and it is the one place in RiskLens where
# a transformer is justified: semantic search genuinely requires learned
# sentence representations. We are not using it to classify fraud.
DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384


@dataclass
class SearchHit:
    """One retrieved case."""

    rank: int
    score: float
    transaction_id: int
    risk_band: str
    probability: float
    amount: float
    is_fraud: int | None
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "similarity": round(self.score, 4),
            "transaction_id": self.transaction_id,
            "risk_band": self.risk_band,
            "probability": round(self.probability, 4),
            "amount": round(self.amount, 2),
            "is_fraud": self.is_fraud,
            "text": self.text,
        }


class SemanticIndex:
    """A FAISS index over case narratives (or policy chunks).

    Why FAISS rather than a hosted vector database
    ----------------------------------------------
    FAISS is a local library, not a service. No account, no network, no cost,
    and - critically for financial data - nothing leaves the machine. At our
    corpus size (thousands of documents) a hosted vector DB would add
    operational complexity and latency for no benefit.

    Why IndexFlatIP specifically
    ----------------------------
    "Flat" means exhaustive search: compare the query against every vector.
    That is O(n), which sounds bad but is milliseconds for thousands of
    documents, and it returns the EXACT nearest neighbours.

    Approximate indexes (IVF, HNSW) trade accuracy for speed and only pay off
    in the millions of vectors. Choosing exact search here is the correct
    engineering call, and being able to explain WHY you did not reach for the
    fancier option is worth more than using it.

    Why inner product (IP) rather than L2 distance
    ----------------------------------------------
    We L2-NORMALISE every vector before adding it. Once vectors have unit
    length, their inner product IS their cosine similarity:

        cos(a, b) = (a . b) / (|a| |b|)  =  a . b   when |a| = |b| = 1

    Cosine similarity measures the ANGLE between vectors, ignoring magnitude.
    That matters because a long narrative and a short one can express the same
    meaning; we want direction, not length.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self.index = None
        self.metadata: list[dict[str, Any]] = []

    @property
    def model(self):
        """Lazy-load: the model is ~90 MB, so only pay for it when used."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Text -> L2-normalised float32 vectors.

        `normalize_embeddings=True` does the unit-length step described above,
        which is what turns FAISS's inner product into cosine similarity.
        """
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype("float32")

    def build(self, texts: list[str], metadata: list[dict[str, Any]]) -> "SemanticIndex":
        """Embed a corpus and build the index."""
        import faiss

        if len(texts) != len(metadata):
            raise ValueError("texts and metadata must be the same length")

        log.info("embedding %d documents ...", len(texts))
        vecs = self.embed(texts)

        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.metadata = metadata
        log.info("index built: %d vectors x %d dims", self.index.ntotal, vecs.shape[1])
        return self

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Find the k most semantically similar documents."""
        if self.index is None:
            raise RuntimeError("index not built - call build() or load() first")

        qv = self.embed([query])
        scores, idx = self.index.search(qv, min(k, self.index.ntotal))

        hits = []
        for rank, (score, i) in enumerate(zip(scores[0], idx[0]), start=1):
            if i < 0:
                continue
            m = self.metadata[i]
            hits.append(SearchHit(
                rank=rank,
                score=float(score),
                transaction_id=m.get("transaction_id", -1),
                risk_band=m.get("risk_band", "?"),
                probability=m.get("probability", float("nan")),
                amount=m.get("amount", float("nan")),
                is_fraud=m.get("is_fraud"),
                text=m.get("text", ""),
            ))
        return hits

    # ---- persistence ----------------------------------------------------
    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(
                {"model_name": self.model_name, "metadata": self.metadata},
                fh, indent=2, default=str,
            )
        log.info("saved index to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "SemanticIndex":
        import faiss

        with open(directory / "metadata.json", "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        obj = cls(model_name=payload["model_name"])
        obj.index = faiss.read_index(str(directory / "index.faiss"))
        obj.metadata = payload["metadata"]
        return obj


# =========================================================================
# Chunking, for the policy corpus
# =========================================================================
def chunk_text(
    text: str, *, chunk_words: int = 180, overlap_words: int = 40
) -> list[str]:
    """Split a long document into overlapping windows.

    Why chunk at all
    ----------------
    Embedding a whole 10-page policy into ONE 384-dimensional vector averages
    away everything specific. The vector ends up meaning "this is about fraud
    policy" and matches every query equally badly.

    Smaller chunks keep each vector focused on one idea, so retrieval can
    return the specific paragraph that answers the question.

    Why OVERLAP
    -----------
    Without overlap, a sentence that straddles a boundary is split in half and
    neither half is retrievable as a coherent statement.

    Tiny example, chunking every 5 words with no overlap:
        "...decline the transaction | and notify the cardholder..."
    A query for "notify the cardholder after declining" matches neither chunk
    well. A 2-word overlap keeps the connection intact.
    """
    words = text.split()
    if len(words) <= chunk_words:
        return [text.strip()] if text.strip() else []

    step = max(1, chunk_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_words]).strip()
        if piece:
            chunks.append(piece)
        if start + chunk_words >= len(words):
            break
    return chunks


def load_policy_corpus(policy_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read policy markdown files and chunk them for retrieval."""
    texts: list[str] = []
    meta: list[dict[str, Any]] = []

    for path in sorted(policy_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = raw.splitlines()[0].lstrip("# ").strip() if raw else path.stem
        for i, chunk in enumerate(chunk_text(raw)):
            texts.append(chunk)
            meta.append({
                "source": path.name,
                "title": title,
                "chunk": i,
                "text": chunk,
            })
    log.info("policy corpus: %d chunks from %d files",
             len(texts), len(list(policy_dir.glob("*.md"))))
    return texts, meta
