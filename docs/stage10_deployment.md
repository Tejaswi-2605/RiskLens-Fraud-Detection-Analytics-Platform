# Stage 10 — Scale and Ship: PySpark, FastAPI, Streamlit, Docker

**Every term defined in plain language, with a tiny worked example.**

---

## The problem this stage solves

Everything before this ran in a notebook against a static file. Production
differs in one way that breaks most projects.

## ⭐ Training/serving skew — the concept that matters most here

**New term — Training/serving skew:** when the features computed at *serving*
time differ from those computed at *training* time.

**Tiny example — the classic version:**

```
TRAINING:
   median of card1 across 438,125 training rows  =  9633
   missing card1  →  filled with 9633

SERVING (naive implementation):
   median of card1 across today's 200 transactions  =  7412
   missing card1  →  filled with 7412     ← DIFFERENT VALUE
```

The model was trained expecting 9633 to mean "missing". It now receives 7412,
which it interprets as a genuine card ID. **No error is raised.** Predictions
degrade silently.

### Other ways skew creeps in

| Cause | What happens |
|---|---|
| Recomputed statistic | The imputation median, scaler mean, or frequency map differs |
| **Column order** | XGBoost matches **positionally** as well as by name |
| Missing column | The model sees a shifted feature vector |
| Different code path | Training used pandas; serving reimplemented it in Java |

### Our defence: structural, not procedural

The service loads the **same artefacts the training run produced**:

```
models/xgboost.joblib            the fitted model
models/frequency_encoder.joblib  the fitted encoder, with TRAIN-time counts
models/feature_names.joblib      the exact column list, in order
```

- The encoder is **not re-fitted**
- The feature list is **not recomputed**
- `build_features()` calls **the same `add_deterministic_features`** the
  training script called

```python
return df.reindex(columns=features)   # ← the line people forget
```

**`reindex` does three jobs at once:** adds missing columns as NaN, drops
unexpected ones, and **puts them in the exact training order.**

> If serving disagrees with training, it's because an artefact is *stale* —
> and `/health` reports which artefacts loaded, so you can tell.

---

# FastAPI

## What it is

**New term — REST API:** a way for other programs to call your service over
HTTP.

**New term — FastAPI:** a Python web framework that generates interactive
documentation and validates inputs automatically from type hints.

## Why FastAPI over Flask

| Feature | Flask | **FastAPI** |
|---|---|---|
| Input validation | Manual | **Automatic** from Pydantic types |
| API documentation | Write it yourself | **Auto-generated** at `/docs` |
| Async support | Bolt-on | Native |

**Tiny example of what Pydantic buys you.**

```python
class TransactionIn(BaseModel):
    TransactionAmt: float = Field(..., gt=0)
```

Send `{"TransactionAmt": -50}` and FastAPI returns a clear 422 error **before
your code runs**. Without it, a negative amount reaches `log1p(-50)` and
produces `NaN`, which silently propagates into the model.

## Why almost every field is optional

```python
TransactionAmt: float          # required
ProductCD: str | None = None   # optional
DeviceType: str | None = None  # optional
```

Real payment messages are **sparse**. And Stage 2 proved that **which fields
are absent is itself signal** — fraud is 4× higher when identity data is
present. So a missing field is left as `NaN`, which is *correct*, not a
fallback: the model learned how to route missing values.

## The `/health` endpoint — designed to be honest

```json
{
  "status": "degraded",
  "model_loaded": false,
  "encoder_loaded": true,
  "operating_threshold": 0.5
}
```

> **A health endpoint that returns `{"status": "ok"}` while the model failed to
> load is worse than no health endpoint at all.** It tells your orchestrator
> everything is fine while every request returns 503.

## Loading artefacts once, at startup

```python
@asynccontextmanager
async def lifespan(app): ...   # runs once
```

Loading a 30 MB model per request would add seconds of latency to every call.
`lifespan` loads once and shares.

## The endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | What's loaded and ready |
| `POST /score` | Probability + risk band + decision |
| `POST /explain` | Score **plus reason codes** — required by the governance policy |
| `POST /search/cases` | Semantic search |
| `POST /policy/ask` | RAG over policy |

---

# Streamlit

## Who it's for

A **fraud analyst**, not a data scientist. That single constraint drives every
design choice:

| Do | Don't |
|---|---|
| Show a **risk band** and a **decision** | Show a raw float |
| Show reason codes in **plain English** | Show `V257 = 3.2` |
| Show **precedent** — similar past cases | Make them search manually |
| Show **policy with citation** | Make them look it up |

## Why caching is not optional

**Streamlit reruns the entire script on every interaction** — every slider
move, every click.

```python
@st.cache_resource
def load_artifacts(): ...
```

Without this, moving a slider reloads a 30 MB model. `cache_resource` is for
unhashable objects like models and DB connections; `cache_data` is for
serialisable results like DataFrames.

---

# Docker

## What it is, and the problem it solves

**New term — Container:** your application plus every dependency, packaged so
it runs identically anywhere.

**The problem it solves:** *"it works on my machine."*

**Tiny example.** Your laptop has Python 3.11 and xgboost 2.1.3. The server has
Python 3.9 and xgboost 1.7. Your model file won't load. A container ships the
exact versions with the code.

## Multi-stage builds

```dockerfile
FROM python:3.11-slim AS builder    # has compilers, pip caches
RUN pip install ...

FROM python:3.11-slim AS runtime    # clean
COPY --from=builder /usr/local/lib/python3.11/site-packages ...
```

Two benefits:

| Benefit | Why |
|---|---|
| **Smaller image** | Compilers and caches never reach the final layer |
| **More secure** | **A compiler in a production container is an attacker's tool.** Fewer packages = fewer CVEs |

## Layer caching — the most important line in the file

```dockerfile
COPY requirements.txt pyproject.toml ./   # ← dependencies FIRST
RUN pip install -r requirements.txt
COPY src/ ./src/                          # ← code SECOND
```

**New term — Layer caching:** Docker caches each instruction and reuses it
until its inputs change.

**Tiny example of why the order matters.**

```
❌ Code copied first:
   edit one .py file  →  cache invalidated  →  reinstall ALL packages (5 min)

✅ Requirements copied first:
   edit one .py file  →  requirements layer still cached  →  rebuild in 5 sec
```

Requirements change far less often than code. Put the stable thing first.

## Running as non-root

```dockerfile
RUN useradd --create-home risklens
USER risklens
```

If the application is compromised, the attacker lands as an unprivileged user
who cannot modify system files or install packages. **Containers do not
isolate root as strongly as people assume** — a root process in a container is
much closer to root on the host than a non-root one.

## Why `libgomp1`

XGBoost uses **OpenMP** for multithreading. Omit this library and the container
fails at import with a confusing linker error rather than a clear message.

## Why models are MOUNTED, not baked in

```yaml
volumes:
  - ./models:/app/models:ro    # ro = read-only
```

| Reason | Explanation |
|---|---|
| Retraining shouldn't need an image rebuild | Swap the file, restart |
| Image layers are immutable and cached forever | A 100 MB model in a layer is there permanently |
| `:ro` | The API must never modify the artefacts it serves |

## Why `data/` is deliberately NOT mounted

The API scores payloads it is *sent*. It has no reason to read the training
dataset — and not mounting it means a compromised container **cannot
exfiltrate it**.

## `depends_on: service_healthy`

```yaml
depends_on:
  api:
    condition: service_healthy
```

Waits for the API to report a **loaded model**, not merely a running process.

---

# PySpark — with an honest verdict

## ⚠️ The honest framing

> **Our dataset is 590,540 rows and fits in 928 MB of RAM. Spark is NOT needed
> here, and using it is SLOWER than pandas** because of JVM startup and
> serialisation overhead.

So why include it? Because the interesting question isn't *"can you call
Spark"* — it's:

**"What changes when the data no longer fits in memory, and how much of your
pipeline survives?"**

For RiskLens the answer is: **the interfaces survive; the engine swaps.**

## Why that's not luck — it's a Stage 1 payoff

| Stage 1 decision | Payoff at Stage 10 |
|---|---|
| Persisted to **Parquet** | Spark reads it natively and in parallel, prunes columns, pushes down predicates. With CSV, Spark would infer schema and read everything |
| Contract expressed as **aggregations** | Row counts, key uniqueness, class balance translate to Spark almost line for line |
| Split expressed as a **rule over a column** | Not Python state, so it ports without reinterpretation |
| Features are **row-wise** | No cross-row state to distribute |

## What does NOT survive automatically

| Component | Why | What you'd do |
|---|---|---|
| `FrequencyEncoder` | Needs counts across all rows | Distributed `groupBy` + broadcast join |
| XGBoost | Single-machine library | `xgboost4j-spark`, or pull a sample to the driver |
| SHAP | No distributed implementation | Sample and explain on the driver |

## New terms

**Lazy evaluation** — Spark builds a plan and executes nothing until you call
an *action* like `.count()` or `.collect()`.

**Tiny example.**
```python
df.filter(...).select(...).groupBy(...)   # nothing happens yet
df.count()                                # NOW the whole plan runs, optimised
```
This lets Spark optimise the *whole* chain — e.g. pushing the filter down to
the file read so less data is ever loaded.

**Shuffle partitions** — how many partitions Spark creates after an operation
requiring data movement. The default of 200 is designed for clusters; locally
it creates 200 tiny tasks whose scheduling overhead exceeds the work. We set 8.

## When is Spark genuinely right?

- Data exceeds single-machine memory (~50M+ rows here)
- The source is already partitioned across a data lake
- The same job runs on a schedule over growing volumes
- Multiple teams query the same tables concurrently

---

# Interview Q&A

### Q1. What's training/serving skew and how did you prevent it?
It's when features computed at serving time differ from those at training time
— a recomputed imputation median, a different frequency map, a different column
order. Nothing errors; predictions just degrade. I prevent it structurally: the
API loads the exact artefacts the training run produced — the fitted encoder
with its train-time counts and the exact ordered feature list — and calls the
same feature-building function. Nothing is recomputed at serving time.

### Q2. Why does column order matter if the columns are named?
XGBoost matches features positionally as well as by name. A reordered or
missing column produces a confident wrong answer rather than an error, which is
the worst possible failure mode. `reindex(columns=features)` fixes order, adds
missing columns as NaN, and drops unexpected ones in one call.

### Q3. Why are most API fields optional?
Real payment messages are sparse, and Stage 2 showed that which fields are
absent is genuine signal — fraud is four times higher when identity data is
present. So a missing field becomes NaN, which XGBoost routes deliberately. It
isn't a fallback; it's information.

### Q4. Why multi-stage Docker builds?
The builder stage has compilers and pip caches; the runtime stage copies only
the installed packages. Smaller image, and more importantly a compiler in a
production container is an attacker's tool. Fewer packages also means fewer
CVEs to patch.

### Q5. Why copy requirements before source code?
Docker layer caching. Requirements change far less often than code, so putting
them first means editing a Python file rebuilds in seconds instead of
reinstalling every dependency.

### Q6. Why mount models rather than bake them into the image?
Retraining shouldn't require an image rebuild — you swap the file and restart.
And image layers are immutable and cached forever, so a large model baked into
a layer stays there permanently. I mount read-only because the API must never
modify what it serves.

### Q7. Your health endpoint returns "degraded". Why not just up/down?
Because a health check that returns OK while the model failed to load is worse
than none — it tells your orchestrator everything is fine while every request
returns 503. Mine reports which artefacts loaded, so the failure is diagnosable
from the endpoint itself.

### Q8. You said Spark is slower than pandas here. Why include it?
Because the interesting question isn't whether I can call Spark — it's what
survives when the data outgrows one machine. For RiskLens the interfaces
survive and only the engine swaps, and that's a direct payoff from choosing
Parquet in Stage 1 and expressing the data contract as aggregations rather than
row-by-row Python. I'd also say clearly which parts *don't* port: the frequency
encoder needs a distributed groupBy and broadcast, XGBoost needs
xgboost4j-spark, and SHAP has no distributed implementation.

### Q9. What is lazy evaluation and why does it help?
Spark builds a plan and executes nothing until an action like `.count()`. That
lets it optimise the whole chain rather than each step — for instance pushing a
filter down into the Parquet read so less data is ever loaded.

### Q10. Why run the container as a non-root user?
If the application is compromised, the attacker lands unprivileged and can't
modify system files or install tools. Containers don't isolate root as strongly
as people assume — a root process inside one is much closer to host root than a
non-root process is.

---

# Common Mistakes

1. **Recomputing statistics at serving time** — the canonical skew bug
2. **Not pinning column order** — XGBoost matches positionally
3. **Loading the model per request** — seconds of latency on every call
4. **Health check that always returns OK** — worse than none
5. **Copying source before requirements** in a Dockerfile — destroys layer caching
6. **Running as root** in a container
7. **Baking models into the image** — every retrain needs a rebuild
8. **Forgetting `libgomp1`** — XGBoost fails at import with a cryptic error
9. **Not caching in Streamlit** — reloads the model on every slider move
10. **Using Spark for data that fits in RAM** — slower, and it signals you don't know when to use it

---

# How to run

```powershell
# API
uvicorn risklens.api.app:app --reload
#   → http://localhost:8000/docs

# Analyst console
streamlit run app/streamlit_app.py
#   → http://localhost:8501

# Both, containerised
docker compose up --build

# PySpark demonstration (needs Java)
python scripts/run_spark.py
```

---

# Files

| File | Purpose |
|---|---|
| `src/risklens/api/app.py` | FastAPI service, skew-safe feature building |
| `app/streamlit_app.py` | Analyst console |
| `Dockerfile` | Multi-stage, non-root, healthchecked |
| `docker-compose.yml` | API + UI, artefacts mounted read-only |
| `scripts/run_spark.py` | PySpark port + honest verdict |
