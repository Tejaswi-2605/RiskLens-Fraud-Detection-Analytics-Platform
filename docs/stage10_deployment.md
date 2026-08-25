# Stage 10 — Serving: FastAPI and Streamlit

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
models/feature_schema.joblib     the exact dtypes, with TRAIN category lists
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

### ⚠️ The dtype trap — this one actually bit us

The API crashed on its first real request:

```
ValueError: DataFrame.dtypes for data must be int, float, bool or category.
Invalid columns: ProductCD: object, card4: object, ...
```

At training, low-cardinality strings were pandas `category` (set during Stage 1
ingestion). At serving, `pd.DataFrame([payload])` produces `object`.

**The raised error was the lucky outcome.** The dangerous version is silent:

> A `category` is stored as an integer **code** plus a categories list, and
> XGBoost splits on the **codes**. If serving builds its own list from one row,
> `"visa"` might be code 0 here and code 3 at training — and the model applies
> a split it learned for a *different card network*. No error. Wrong answer.

So it is not enough to cast to `category`. The **exact same categories, in the
same order** must be used. `scripts/export_feature_schema.py` captures the
`CategoricalDtype` from the **training partition** into
`models/feature_schema.joblib`, and both the API and the UI cast to it.

Categories come from train only, so levels appearing solely in the test period
cannot leak in.

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

### Q4. You had a dtype bug in production. What was it and how did you fix it?
At training, string columns were pandas `category`; at serving,
`pd.DataFrame([payload])` gives `object`, and XGBoost rejected it. The error was
the lucky outcome — a category is an integer code plus a category list, and
XGBoost splits on the codes, so if serving builds its own list from one row the
codes stop matching and the model applies a split learned for a different value,
silently. I now persist the training `CategoricalDtype` and cast to it at
serving, with the categories taken from the training partition only.

### Q5. Your health endpoint returns "degraded". Why not just up/down?
Because a health check that returns OK while the model failed to load is worse
than none — it tells your orchestrator everything is fine while every request
returns 503. Mine reports which artefacts loaded, so the failure is diagnosable
from the endpoint itself.


# Common Mistakes

1. **Recomputing statistics at serving time** — the canonical skew bug
2. **Not pinning column order** — XGBoost matches positionally
3. **Not pinning column *dtypes*** — a category built at serving gets different
   integer codes than training, so the model applies the wrong split silently
4. **Loading the model per request** — seconds of latency on every call
5. **A health check that always returns OK** — worse than none
6. **Not caching in Streamlit** — reloads the model on every slider move

---

# How to run

```powershell
# API
uvicorn risklens.api.app:app --port 8000
#   -> http://localhost:8000/docs

# Analyst console
streamlit run app\streamlit_app.py --server.port 8502
#   -> http://localhost:8502
```

---

# Files

| File | Purpose |
|---|---|
| `src/risklens/api/app.py` | FastAPI service, skew-safe feature building |
| `scripts/export_feature_schema.py` | Captures the training dtypes for serving |
| `app/streamlit_app.py` | Analyst console |
