"""Stage 8 (part 3) - Retrieval-Augmented Generation over fraud policy.

What RAG is, and why it is the right tool here
----------------------------------------------
A language model knows what was in its training data. It does not know YOUR
organisation's fraud policy, and if you ask it anyway it will produce a
plausible, confident, invented answer.

RAG fixes this by changing the question. Instead of:

    "What is the policy for a HIGH risk transaction?"        -> hallucination

we do:

    1. RETRIEVE the policy passages most relevant to the question
    2. AUGMENT the prompt with those passages
    3. GENERATE an answer that must be grounded in them

The model stops being a knowledge source and becomes a reading-comprehension
engine over documents we control.

Why this matters more in finance than almost anywhere else
-----------------------------------------------------------
  * Policy changes. Retraining a model every time a threshold moves is
    absurd; swapping a document in the index is trivial.
  * Answers must be CITEABLE. An analyst acting on guidance needs to know
    which policy section it came from, and a regulator will ask.
  * Hallucinated policy is a compliance incident, not a bug.

Why a LOCAL model (Ollama)
--------------------------
Transaction data never leaves the machine. For payment data that is a hard
requirement, not a preference. It also costs nothing, which matters for a
portfolio project.

The trade-off is real and worth stating: a 3B parameter local model reasons
noticeably less well than a frontier model. We compensate by keeping its job
narrow - summarise and ground, never decide.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.2:3b"

# The system prompt is the main lever we have over a small model. Each rule
# exists because small models reliably fail in that specific way.
SYSTEM_PROMPT = """\
You are a fraud operations assistant for a payments risk team.

Rules you must follow:
1. Answer ONLY from the POLICY CONTEXT provided. It is your sole source of truth.
2. If the context does not contain the answer, say exactly:
   "The provided policy does not cover this."
   Never guess and never fill gaps from general knowledge.
3. Cite the source document for every claim, like [01_risk_scoring_and_decisions.md].
4. Be concise. Analysts are working an alert queue under time pressure.
5. Never invent thresholds, numbers, or procedure steps. If a number is not in
   the context, do not state a number.
6. You advise. You do not decide. Never instruct that a transaction be
   declined; describe what the policy requires and who owns the decision.\
"""


@dataclass
class RAGAnswer:
    """An answer plus the evidence it was built from - always together."""

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    chunks_used: int = 0
    model: str = DEFAULT_MODEL

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "chunks_used": self.chunks_used,
            "model": self.model,
        }


def ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Is a local Ollama server running with the model pulled?"""
    try:
        import ollama

        names = [m.get("model", "") for m in ollama.list().get("models", [])]
        return any(model.split(":")[0] in n for n in names)
    except Exception as exc:  # noqa: BLE001
        log.warning("ollama unavailable: %s", exc)
        return False


def generate(
    prompt: str,
    *,
    system: str = SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    num_predict: int = 400,
) -> str:
    """Call the local model.

    temperature=0.1 is deliberate. Temperature controls randomness: higher
    values produce more varied wording. For creative writing that is
    desirable; for policy guidance it is a liability. We want the same
    question to produce the same answer, so we keep it near-deterministic.
    """
    import ollama

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature, "num_predict": num_predict},
    )
    return resp["message"]["content"].strip()


def build_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    """Assemble the augmented prompt.

    Structure matters for small models. We put the CONTEXT FIRST and the
    question LAST, because a small model attends most reliably to the end of
    its prompt - putting the question there keeps it in focus. Each chunk is
    clearly delimited and labelled with its source so the model can cite it.
    """
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"--- SOURCE {i}: [{h['source']}] ---\n{h['text']}"
        )
    context = "\n\n".join(blocks)

    return textwrap.dedent(f"""\
        POLICY CONTEXT
        ==============
        {context}

        ==============
        Using ONLY the policy context above, answer this question.
        Cite the source file for each claim. If the context does not contain
        the answer, say "The provided policy does not cover this."

        QUESTION: {question}
        """)


class PolicyRAG:
    """Retrieval-augmented question answering over the policy corpus."""

    def __init__(self, index, model: str = DEFAULT_MODEL, top_k: int = 4):
        self.index = index
        self.model = model
        self.top_k = top_k

    def retrieve(self, question: str, k: int | None = None) -> list[dict[str, Any]]:
        """Fetch the most relevant policy chunks.

        We search the index directly rather than through SearchHit, because
        policy metadata has a different shape from case metadata (source and
        title rather than transaction_id and risk_band).
        """
        k = k or self.top_k
        qv = self.index.embed([question])
        scores, idx = self.index.index.search(qv, min(k, self.index.index.ntotal))
        out = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            m = dict(self.index.metadata[i])
            m["similarity"] = float(score)
            out.append(m)
        return out

    def ask(self, question: str, k: int | None = None) -> RAGAnswer:
        """Retrieve, augment, generate."""
        hits = self.retrieve(question, k)
        if not hits:
            return RAGAnswer(
                question=question,
                answer="No policy documents were retrieved.",
                model=self.model,
            )

        prompt = build_prompt(question, hits)
        answer = generate(prompt, model=self.model)
        sources = sorted({h["source"] for h in hits})
        return RAGAnswer(
            question=question,
            answer=answer,
            sources=sources,
            chunks_used=len(hits),
            model=self.model,
        )


# =========================================================================
# Evaluating the RAG system
# =========================================================================
def groundedness_check(answer: str, context_chunks: list[str]) -> dict[str, Any]:
    """A cheap, deterministic check that the answer stayed in its lane.

    This is NOT a full LLM-as-judge evaluation. It is a lexical overlap
    heuristic, and it is honest about that. It catches the obvious failure -
    an answer full of content words that appear nowhere in the retrieved
    context - which is the signature of a hallucination.

    Why bother with something so simple
    -----------------------------------
    It is free, instant, deterministic, and it runs on every answer. An
    LLM-judge is better but costs a model call per evaluation and introduces
    its own error. In production you would run this on 100 per cent of
    traffic and the LLM judge on a sample.
    """
    import re

    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for",
        "on", "with", "that", "this", "be", "must", "not", "it", "as", "by",
        "from", "at", "which", "where", "any", "may", "was", "were", "has",
    }
    ctx = " ".join(context_chunks).lower()
    ctx_words = set(re.findall(r"[a-z]{4,}", ctx))
    ans_words = [w for w in re.findall(r"[a-z]{4,}", answer.lower()) if w not in stop]

    if not ans_words:
        return {"grounded_ratio": 0.0, "verdict": "empty answer"}

    overlap = sum(1 for w in ans_words if w in ctx_words)
    ratio = overlap / len(ans_words)
    return {
        "grounded_ratio": round(ratio, 3),
        "unsupported_terms": sorted({w for w in ans_words if w not in ctx_words})[:10],
        "verdict": (
            "well grounded" if ratio >= 0.75
            else "partially grounded - review" if ratio >= 0.55
            else "POSSIBLE HALLUCINATION - most content words are absent "
                 "from the retrieved context"
        ),
        "method": "lexical overlap heuristic, not an LLM judge",
    }
