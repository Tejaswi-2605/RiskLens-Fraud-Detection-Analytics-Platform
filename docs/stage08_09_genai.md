# Stages 8 & 9 — NLP, Semantic Search, RAG and the Agentic Copilot

**Every term defined in plain language, with a tiny worked example.**

---

## First: the honesty problem, and how we solve it

**The IEEE-CIS dataset contains no text.** So how do we justify an NLP
component without bolting on a gimmick?

❌ **The dishonest answer:** invent a fake "customer comment" column.

✅ **What we do:** generate the text a real fraud operation actually produces —
the **case narrative** an analyst writes when investigating an alert.

```
structured transaction + SHAP reason codes
        ↓
   case narrative (text)
        ↓
    embeddings
        ↓
  semantic search  →  "have we seen this before?"
```

That's a genuine, non-contrived NLP corpus. And note the direction of the
architecture:

> **The GenAI layer CONSUMES the ML system. It does not replace it.**
>
> The model predicts. SHAP explains. The LLM reads, retrieves and summarises.
> Neither does the other's job.

---

# Stage 8a — Case Narratives

## What we build

```
ALERT 3512847 | risk HIGH (78.4%) | amount 425.50
Context: ProductCD=C, card4=visa, card6=credit, DeviceType=mobile.
The model raised this alert primarily because of missing payer identity data,
how common this device is and an overnight transaction.
Reducing the score: the card network.
Recommended action: hold for manual review within the hour.
[machine-generated from model output; not an analyst write-up]
```

## ⚠️ Why a deterministic template, NOT the LLM

This surprises people. Three reasons, all of which matter in a regulated
setting:

| Reason | Explanation |
|---|---|
| **Faithfulness** | A template **cannot hallucinate** a driver that SHAP did not produce |
| **Reproducibility** | The same alert always yields the same text, so two analysts see identical words |
| **Cost** | Free and instant, so we can generate hundreds of thousands to build the corpus |

**Tiny example of what an LLM might do wrong.** Given SHAP output showing
`is_night: +0.08`, an LLM asked to "write a narrative" might produce *"the
transaction occurred at 3am from an unusual IP address"* — inventing the IP
address detail entirely. In a compliance context that's a fabricated record.

**The division of labour:** templates *generate facts*; the LLM *reasons over
facts*. Keeping them separate is what keeps the system auditable.

## The glossary

Raw column names are useless to an analyst:

| Raw | Human |
|---|---|
| `id_04_isna` | missing payer identity data |
| `card1_freq` | how common this card identifier is |
| `dist1` | the distance between billing and transaction location |
| `D15` | days since a prior related transaction |
| `V257` | anonymised behavioural signal V257 |

---

# Stage 8b — Semantic Search

## The problem keyword search can't solve

An analyst searches for **"night-time purchase from a new browser"**.

A relevant case is described as **"overnight transaction on an unrecognised
device"**.

**Zero shared keywords.** Keyword search returns nothing. The analyst concludes
there's no precedent, and they're wrong.

## What an embedding is

**New term — Embedding:** a list of numbers representing the *meaning* of a
piece of text.

**Tiny example** (simplified to 3 dimensions):

```
"cat"      →  [0.2, 0.9, 0.1]
"kitten"   →  [0.2, 0.8, 0.1]   ← very close to "cat"
"railway"  →  [0.9, 0.1, 0.7]   ← far from both
```

No word is shared between "cat" and "kitten", yet their vectors are adjacent.
**That is the entire value proposition.**

Our model uses **384 dimensions**, not 3.

## The model: `all-MiniLM-L6-v2`

| Property | Value | Why it matters |
|---|---|---|
| Dimensions | 384 | Small index, instant search |
| Size | ~90 MB | Runs on CPU in milliseconds |
| Runs | **Locally** | **No transaction data leaves the machine** |

> **This is the one place in RiskLens where a transformer is justified.**
> The original brief said no BERT unless there's a clear NLP requirement.
> Semantic search genuinely requires learned sentence representations — you
> cannot do it with TF-IDF and get meaning-based matching. We are *not* using
> a transformer to classify fraud; XGBoost does that.

## Cosine similarity, and why we normalise

**New term — Cosine similarity:** measures the **angle** between two vectors,
ignoring their length.

$$\cos(a,b) = \frac{a \cdot b}{|a||b|}$$

**Why angle and not distance?** A long narrative and a short one can express
the same meaning. We want *direction*, not *magnitude*.

**The trick:** if we **L2-normalise** every vector to unit length, then
`|a| = |b| = 1`, so:

$$\cos(a,b) = a \cdot b$$

Cosine similarity **becomes** the plain inner product. That's why we use
FAISS's `IndexFlatIP` (inner product) — after normalisation, it computes
cosine similarity for free.

## FAISS, and why we chose the *simple* index

**New term — FAISS:** Facebook AI Similarity Search — a local library for fast
nearest-neighbour search over vectors.

**Why a local library rather than a hosted vector database?**
No account, no network, no cost, and nothing leaves the machine. For payment
data that's a hard requirement.

**Why `IndexFlatIP` (exhaustive) rather than an approximate index?**

| | Flat (ours) | IVF / HNSW |
|---|---|---|
| Method | Compare against **every** vector | Approximate shortcuts |
| Accuracy | **Exact** | Approximate |
| Speed at 1,500 docs | Milliseconds | Milliseconds |
| Speed at 100M docs | Too slow | Fast |

At our corpus size, exact search is **already instant**. Approximate indexes
only pay off in the millions of vectors, and they trade accuracy for that speed.

> **Being able to explain why you did NOT reach for the fancier option is worth
> more than having used it.**

## Chunking, and why overlap matters

Policy documents are long. Embedding a whole 10-page policy into **one** vector
averages away everything specific — the vector ends up meaning "this is about
fraud policy" and matches every query equally badly.

**So we chunk:** 180 words per chunk, with **40 words of overlap**.

**Tiny example of why overlap is needed.** Chunking every 5 words with no
overlap:

```
chunk 1: "...analysts must always decline the"
chunk 2: "transaction and notify the cardholder..."
```

A query for *"notify the cardholder after declining"* matches **neither chunk
well** — the idea was cut in half. A 2-word overlap keeps the connection intact.

---

# Stage 8c — RAG (Retrieval-Augmented Generation)

## The problem RAG solves

Ask an LLM: *"What's our policy for a HIGH risk transaction?"*

It **does not know your policy**. But it will produce a confident, plausible,
**completely invented** answer.

**New term — Hallucination:** an LLM generating fluent, confident text that is
factually wrong. It's not lying — it's pattern-completing.

**In a compliance context, hallucinated policy is an incident, not a bug.**

## How RAG changes the question

```
❌ WITHOUT RAG
   "What's the policy for HIGH risk?"  →  LLM invents an answer

✅ WITH RAG
   1. RETRIEVE the policy passages most relevant to the question
   2. AUGMENT the prompt with those passages
   3. GENERATE an answer that must be grounded in them
```

**The model stops being a knowledge source and becomes a reading-comprehension
engine over documents we control.**

## Why RAG matters more in finance than almost anywhere

| Reason | Explanation |
|---|---|
| **Policy changes** | Retraining a model when a threshold moves is absurd. Swapping a document in the index is trivial. |
| **Answers must be citeable** | An analyst acting on guidance needs the source. A regulator will ask. |
| **Hallucination is a compliance event** | Not a UX annoyance |

## Prompt engineering — the system prompt

For a small local model, the system prompt is the main lever we have. **Every
rule exists because small models reliably fail in that specific way:**

```
1. Answer ONLY from the POLICY CONTEXT provided.
2. If the context lacks the answer, say exactly:
   "The provided policy does not cover this."
3. Cite the source document for every claim.
4. Be concise.
5. Never invent thresholds or numbers.
6. You advise. You do not decide.
```

**Rule 2 is the most important.** Without an explicit escape hatch, a model
under instruction to answer *will* answer — from general knowledge if it must.
Giving it a permitted way to say "I don't know" is what makes refusal possible.

**Rule 6** matters because an LLM must never be the thing that declines a
customer's payment. It summarises; a human decides.

## Prompt structure

We put **context first, question last**. Small models attend most reliably to
the **end** of their prompt, so putting the question there keeps it in focus.

```
POLICY CONTEXT
--- SOURCE 1: [01_risk_scoring_and_decisions.md] ---
<chunk text>

--- SOURCE 2: [05_alert_triage_sla.md] ---
<chunk text>

Using ONLY the context above, answer this question...
QUESTION: What action is required for a HIGH risk transaction?
```

## Temperature

**New term — Temperature:** controls randomness in generation. Higher = more
varied wording.

We use **0.1** (near-deterministic). For creative writing, variety is
desirable. **For policy guidance it's a liability** — the same question must
produce the same answer.

## Evaluating RAG — the groundedness check

We check what fraction of the answer's content words appear in the retrieved
context.

```
grounded_ratio ≥ 0.75  →  well grounded
0.55 – 0.75            →  partially grounded, review
< 0.55                 →  POSSIBLE HALLUCINATION
```

**Being honest about this method:** it's a *lexical overlap heuristic*, not an
LLM judge. It catches the obvious failure — an answer full of content words
appearing nowhere in the context.

**Why bother with something so crude?** It's free, instant, deterministic, and
runs on **100% of traffic**. An LLM judge is better but costs a model call per
evaluation and introduces its own errors. In production: this on everything,
LLM judge on a sample.

## The out-of-scope test

We deliberately ask: *"What is the capital of France?"*

A correctly configured RAG system should **refuse** — *"The provided policy
does not cover this."* **That refusal is the safety property we want.** A
system that happily answers off-corpus questions will also happily invent
policy.

---

# Stage 9 — The Investigation Copilot

## What an "agent" is

**New term — Agent:** an LLM that can **call tools** (functions) to gather
information, rather than answering only from its training data.

**Tiny example.**
```
User:  "Investigate alert 3512847"
Agent: [calls score_transaction(3512847)]      → 0.784, HIGH
Agent: [calls explain_alert(3512847)]          → top drivers
Agent: [calls find_similar_cases("...")]       → 3 precedents
Agent: [calls lookup_policy("HIGH risk")]      → policy extract
Agent: "ASSESSMENT: This looks like..."
```

## Our tools ARE the rest of RiskLens

| Tool | Wraps |
|---|---|
| `score_transaction` | Stage 4 XGBoost model |
| `explain_alert` | Stage 7 SHAP reason codes |
| `find_similar_cases` | Stage 8 semantic search |
| `lookup_policy` | Stage 8 RAG |
| `get_transaction` | The underlying data |

**That's the design point worth defending.** The LLM has no fraud knowledge of
its own. It orchestrates and summarises components that do.

## Two modes, and why we default to the *less* impressive one

### Mode 2 — true tool-calling agent loop
```python
while model asks for a tool:
    execute it
    append the result
    ask the model again
```
The LLM decides what to call and when.

### Mode 1 — deterministic workflow ⭐ **our default**
```
1. get_transaction     (always)
2. score_transaction   (always)
3. explain_alert       (always)
4. find_similar_cases  (always)
5. lookup_policy       (always)
6. LLM writes the summary from what the tools returned
```

### Why default to Mode 1?

| Reason | Explanation |
|---|---|
| **Reliability** | A 3B local model calls tools unreliably — invents arguments, skips steps, sometimes answers from memory without calling anything |
| **Auditability** | Every investigation touched the same evidence, so two cases are comparable |
| **Compliance** | An investigation that *sometimes* skips the policy check is worse than one that always runs the same five steps |

> **Interview answer:** *"I implemented both a true tool-calling loop and a
> deterministic workflow, and I default to the workflow. With a 3-billion
> parameter local model, tool-calling is unreliable — it skips steps. In a
> regulated setting, an investigation that sometimes omits the policy check is
> worse than one that always runs the same five. I'd revisit that with a
> frontier model, where the agentic loop's flexibility would start to pay off."*
>
> **Being able to explain why you chose the less flashy option is the point.**

## The turn limit

```python
for turn in range(max_turns):  # capped at 6
```

**Why?** A small model can loop forever calling the same tool. **That cap is
the difference between an agent and a runaway process.**

## Output structure

We force a fixed format, because analysts scan rather than read:

```
ASSESSMENT:        what this transaction looks like
KEY DRIVERS:       why the model scored it this way
PRECEDENT:         what similar historical cases suggest
POLICY:            what policy requires, with citation
RECOMMENDED ACTION: what to do next
```

Plus: *"If evidence for a section is missing, write 'Not available'."* — again,
an explicit escape hatch prevents invention.

---

# Interview Q&A

### Q1. The dataset has no text. How is an NLP component justified?
I didn't invent a fake text column. I generate the artefact a real fraud
operation actually produces — the case narrative an analyst writes when
investigating an alert — from the model score plus its SHAP reason codes. That
gives a genuine corpus, and making it searchable is exactly what a real
case-management system does.

### Q2. Why generate narratives with a template rather than the LLM?
Faithfulness. A template cannot hallucinate a driver SHAP didn't produce. It's
also reproducible, so two analysts reading the same alert see the same words,
and it's free enough to generate a whole corpus. The LLM's job is to reason
over the facts, not to invent them — keeping generation separate from reasoning
is what keeps the system auditable.

### Q3. Explain semantic search versus keyword search.
Keyword search matches shared words. A query for "night-time purchase from a
new browser" won't match a case written as "overnight transaction on an
unrecognised device" — zero shared terms, despite identical meaning. Semantic
search embeds both into a 384-dimensional space where similar meanings sit
close together, so the match happens on meaning.

### Q4. Why cosine similarity, and why normalise the vectors?
Cosine measures the angle between vectors, ignoring length — a long narrative
and a short one can mean the same thing, so I want direction, not magnitude.
And if you L2-normalise to unit length, cosine similarity becomes the plain
inner product, which is why I use FAISS's IndexFlatIP.

### Q5. Why FAISS flat search rather than an approximate index?
At around 1,500 documents, exhaustive search is already milliseconds and
returns exact neighbours. Approximate indexes like IVF or HNSW trade accuracy
for speed and only pay off in the millions of vectors. Reaching for one here
would add complexity and lose accuracy for no gain.

### Q6. What is RAG and why use it for policy?
Retrieval-augmented generation. Instead of asking the model what the policy is
— which it doesn't know and will invent — I retrieve the relevant policy
passages, put them in the prompt, and require the answer to come from them. The
model becomes a reading-comprehension engine over documents I control. It
matters here because policy changes without retraining, answers must be
citeable, and hallucinated policy is a compliance incident.

### Q7. How do you know the RAG system isn't hallucinating?
Three layers. The system prompt forbids answering outside the retrieved context
and gives an explicit refusal phrase. Every answer carries its source
citations. And I run a lexical groundedness check — what fraction of the
answer's content words appear in the retrieved context — which flags answers
below 55%. That's a heuristic, not an LLM judge, and I'd describe it that way:
it's cheap enough to run on everything, and I'd add an LLM judge on a sample.

### Q8. Why chunk documents with overlap?
Embedding a whole document into one vector averages away everything specific,
so it matches every query equally badly. Chunking keeps each vector focused.
Overlap matters because a sentence straddling a chunk boundary gets cut in
half, and neither half is retrievable as a coherent statement.

### Q9. Describe your agent architecture.
An LLM with five tools, and the tools *are* the rest of RiskLens — the XGBoost
model, SHAP, the case index, the policy RAG, and the raw data. The LLM has no
fraud knowledge of its own; it orchestrates components that do and summarises
what they return. That inversion is deliberate: the GenAI layer consumes the ML
system rather than replacing it.

### Q10. You built a tool-calling loop but default to a fixed workflow. Why?
Reliability. With a 3B local model, tool-calling is unreliable — it invents
arguments and skips steps. In a regulated setting, an investigation that
sometimes omits the policy check is worse than one that always runs the same
five steps, and a fixed order means two cases are comparable. With a frontier
model I'd revisit it, because the flexibility would start to pay for itself.

### Q11. Why a local model instead of a hosted API?
Transaction data never leaves the machine, which for payment data is a hard
requirement rather than a preference. It also costs nothing. The trade-off is
real and I'd state it: a 3B model reasons noticeably less well than a frontier
model, so I compensate by keeping its job narrow — summarise and ground, never
decide.

---

# Common Mistakes

1. **Inventing a fake text column** to justify NLP — dishonest and obvious
2. **Letting the LLM generate the facts** — it will hallucinate drivers
3. **No refusal path in the system prompt** — a model told to answer will always answer
4. **High temperature for factual tasks** — same question, different answers
5. **Embedding whole documents** — the vector means nothing specific
6. **Chunking without overlap** — ideas cut in half become unretrievable
7. **Reaching for an approximate index at small scale** — complexity, lost accuracy, no gain
8. **No turn limit on an agent loop** — a runaway process, not an agent
9. **Letting the LLM make the decision** — it advises; a human decides
10. **Not testing out-of-scope questions** — refusal is the safety property you must verify

---

# Files

| File | Purpose |
|---|---|
| `src/risklens/genai/narratives.py` | Template narrative generation + glossary |
| `src/risklens/genai/search.py` | Embeddings, FAISS index, chunking |
| `src/risklens/genai/rag.py` | Retrieve → augment → generate, groundedness check |
| `src/risklens/genai/agent.py` | Toolbox, deterministic workflow, agent loop |
| `corpus/policies/*.md` | Six illustrative policy documents |
| `scripts/run_genai.py` | Runs Stages 6–9 end to end |
