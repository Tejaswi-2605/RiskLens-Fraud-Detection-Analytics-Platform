"""Entity-linkage features - the highest-leverage addition to the model.

The problem with the base feature set
-------------------------------------
The model sees each transaction IN ISOLATION. But fraud is a pattern over an
ENTITY: one compromised card making several purchases in quick succession. A
single row cannot express "this is the fourth transaction on this card in
twenty minutes", so the model cannot learn it.

Three steps fix that.

STEP 1 - normalise the D columns
--------------------------------
`D1`-`D15` are "days since some previous event", measured RELATIVE to each
transaction. The same real-world event therefore produces a different value
depending on when you look at it, which makes them useless for linking rows.

    day = TransactionDT / 86400
    D1n = day - D1            -> the ABSOLUTE day the card was first seen

Now two transactions from the same card yield the SAME `D1n`, which turns a
drifting offset into a stable identity signal.

STEP 2 - build a pseudo-client UID
----------------------------------
    uid = card1 + "_" + addr1 + "_" + D1n

Not a real customer identifier - an INFERRED one. Same card, same billing
region, same first-seen day is very probably the same client. This is the
"magic feature" from the IEEE-CIS competition.

STEP 3 - aggregate CAUSALLY, which is where we diverge from Kaggle
------------------------------------------------------------------
The winning solutions aggregated over train AND test combined. That is legal
in a competition, where the test features are handed to you. It is IMPOSSIBLE
in production: you cannot know a card's future transactions while scoring
today's.

So every aggregate here uses an EXPANDING, BACKWARD-ONLY window - for each
transaction, statistics over only the PRIOR transactions of that entity.

Two consequences worth stating plainly:

  * The gain will be SMALLER than the transductive version. That is the
    honest cost of a feature that would actually work in production.
  * No train/test contamination is possible, because a row can only ever see
    rows that came before it. There is no `fit` to leak.

Implementation note - why cumulative arithmetic, not `.rolling()`
-----------------------------------------------------------------
`groupby(...).expanding().mean()` is correct but extremely slow at 590k rows.
`groupby(...).cumsum()` followed by `.shift(1)` computes the same thing in one
vectorised pass. The `shift` is what makes it exclusive of the current row -
without it, each transaction would see itself, which is a subtle self-leak.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400

# D columns that represent "days since an event" and are worth normalising.
# D1 (days since card first seen) is the one used for entity identity.
D_COLUMNS = [f"D{i}" for i in range(1, 16)]

# The entity views we build aggregates over, coarse to fine.
ENTITY_KEYS = ["card1", "uid", "uid2"]


def add_normalised_d_columns(
    df: pd.DataFrame, *, time_col: str = "TransactionDT", copy: bool = True
) -> pd.DataFrame:
    """Turn each 'days since X' offset into the absolute day X occurred.

    Tiny example. Two transactions from the same card, 10 days apart:

        txn A: day 100, D1 = 30   ->  D1n = 70
        txn B: day 110, D1 = 40   ->  D1n = 70    <- SAME

    Raw `D1` differs (30 vs 40) and looks like two different situations.
    Normalised, both say "this card was first seen on day 70", which is the
    fact that actually links them.
    """
    out = df.copy() if copy else df
    day = out[time_col] / SECONDS_PER_DAY
    out["day"] = day.astype("float32")

    made = 0
    for col in D_COLUMNS:
        if col in out.columns:
            # rounded to whole days: the underlying quantity is integer days,
            # and rounding prevents float noise from splitting one entity in two
            out[f"{col}n"] = np.floor(day - out[col]).astype("float32")
            made += 1
    log.info("normalised %d D columns into absolute reference days", made)
    return out


def add_entity_keys(df: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    """Construct inferred client identifiers of increasing specificity.

    uid   card1 + addr1 + first-seen-day   -> a probable client
    uid2  uid + card2 + card3              -> a probable client-card pair

    Two levels because they trade off differently: a coarse key groups more
    transactions (better statistics, more collisions between real clients), a
    fine key is more precise but many entities appear only once and their
    aggregates are meaningless.

    NaNs are folded into the string as a literal "na". Dropping them would
    silently discard rows, and "we do not know the billing region" is itself a
    consistent, informative group.
    """
    out = df.copy() if copy else df

    def part(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series("na", index=out.index)
        s = out[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype("object")
        return s.astype("object").where(s.notna(), "na").astype(str)

    d1n = part("D1n")
    out["uid"] = (part("card1") + "_" + part("addr1") + "_" + d1n).astype("category")
    out["uid2"] = (
        out["uid"].astype("object") + "_" + part("card2") + "_" + part("card3")
    ).astype("category")

    log.info("entity keys: uid=%s distinct, uid2=%s distinct",
             f"{out['uid'].nunique():,}", f"{out['uid2'].nunique():,}")
    return out


def add_expanding_entity_features(
    df: pd.DataFrame,
    *,
    keys: list[str] | None = None,
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
) -> pd.DataFrame:
    """Backward-only aggregates per entity. Causal by construction.

    For every transaction and every entity key, we compute statistics over
    ONLY that entity's earlier transactions:

        <key>_count_prior      how many times seen before now
        <key>_amt_mean_prior   running average spend before now
        <key>_amt_std_prior    running spread before now
        <key>_amt_ratio        this amount / running mean
        <key>_secs_since_last  time since this entity last transacted
        <key>_txns_last_day    burst detector

    Why these specifically
    ----------------------
    `amt_ratio` encodes "unusual FOR THIS CUSTOMER", which a global amount
    threshold cannot. Stage 2 found raw TransactionAmt had no predictive power
    at all (Cliff's delta 0.001) - but £500 is unremarkable for one client and
    extraordinary for another, and only a per-entity view can say which.

    `secs_since_last` and `txns_last_day` are velocity. Card testing is
    defined by rapid repeats, and no single-row feature can see it.

    The `.shift(1)` inside each group is the critical line: it excludes the
    current row from its own statistics. Without it every transaction would
    contribute to the average it is being compared against - a self-leak that
    would look like signal and generalise to nothing.
    """
    keys = keys or ENTITY_KEYS

    # Sorting is only needed if the frame is not already in time order - and
    # Stage 1 ingestion sorts by TransactionDT, so in practice it never is.
    #
    # This check is not a micro-optimisation. `sort_values` copies the frame,
    # and at 590,540 rows x 460+ columns pandas must consolidate the blocks
    # into one contiguous ~937 MB array to do it. That allocation failed
    # outright on a machine with several gigabytes free:
    #
    #     numpy._core._exceptions._ArrayMemoryError: Unable to allocate
    #     937 MiB for an array with shape (416, 590540)
    #
    # Skipping a sort that would be a no-op removes the allocation entirely.
    already_sorted = df[time_col].is_monotonic_increasing
    if already_sorted:
        out = df
        log.info("frame already in time order - skipping the sort (saves ~900 MB)")
    else:
        out = df.sort_values(time_col, kind="mergesort")

    amt = out[amount_col].astype("float64")

    for key in keys:
        if key not in out.columns:
            continue
        g = out.groupby(key, observed=True, sort=False)

        # ---- count of PRIOR transactions --------------------------------
        # cumcount is already exclusive of the current row: it counts 0,1,2...
        count_prior = g.cumcount()
        out[f"{key}_count_prior"] = count_prior.astype("float32")

        # ---- running mean / std over PRIOR transactions -----------------
        # cumsum includes the current row, so shift(1) removes it.
        csum = g[amount_col].cumsum() - amt          # sum of prior rows
        csq = g[amount_col].transform(lambda s: (s.astype("float64") ** 2).cumsum())
        csq = csq - amt**2                            # sum of prior squares

        n = count_prior.replace(0, np.nan)            # no prior rows -> undefined
        mean_prior = csum / n
        var_prior = (csq / n) - mean_prior**2
        out[f"{key}_amt_mean_prior"] = mean_prior.astype("float32")
        out[f"{key}_amt_std_prior"] = np.sqrt(var_prior.clip(lower=0)).astype("float32")

        # "how unusual is this amount FOR THIS ENTITY"
        out[f"{key}_amt_ratio"] = (amt / mean_prior).astype("float32")

        # ---- velocity ----------------------------------------------------
        last_t = g[time_col].shift(1)
        out[f"{key}_secs_since_last"] = (out[time_col] - last_t).astype("float32")

        # transactions by this entity in the trailing 24h, excluding this one
        t_day_ago = out[time_col] - SECONDS_PER_DAY
        out[f"{key}_txns_last_day"] = (
            g[time_col]
            .transform(lambda s: pd.Series(
                np.searchsorted(s.to_numpy(), s.to_numpy(), side="left")
                - np.searchsorted(s.to_numpy(), s.to_numpy() - SECONDS_PER_DAY, side="left"),
                index=s.index,
            ))
            .astype("float32")
        )

        log.info("entity features for %-6s -> 6 columns", key)

    # Only restore order if we actually reordered. Calling sort_index() on an
    # unsorted-but-original frame would be another full copy for no reason.
    return out if already_sorted else out.sort_index()


def build_entity_features(
    df: pd.DataFrame,
    *,
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
) -> pd.DataFrame:
    """The full entity pipeline: normalise D, build keys, aggregate causally.

    Memory note
    -----------
    Each step used to `.copy()` the frame. At 590,540 rows x 460+ columns that
    is ~900 MB per copy, and three copies peaked high enough to fail with

        numpy._core._exceptions._ArrayMemoryError:
        Unable to allocate 937 MiB for an array with shape (416, 590540)

    on a machine with several gigabytes free. So we copy ONCE here and the
    steps mutate that copy in place. The caller's frame is still never
    modified, which is the property that actually matters.
    """
    before = df.shape[1]
    out = add_normalised_d_columns(df, time_col=time_col, copy=True)
    out = add_entity_keys(out, copy=False)
    out = add_expanding_entity_features(
        out, time_col=time_col, amount_col=amount_col
    )
    log.info("entity features: %d -> %d columns (+%d)",
             before, out.shape[1], out.shape[1] - before)
    return out
