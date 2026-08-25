"""Stage 2 - the figure set.

Every figure answers ONE question that a later decision depends on. No
decorative charts: if a plot does not change what we do next, it is not here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed, safe in Docker/CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
FRAUD_C, LEGIT_C = "#c0392b", "#2c7fb8"


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name)
    plt.close(fig)
    log.info("wrote %s", name)


def plot_class_balance(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    """Q: how imbalanced is this really? A: the reason accuracy is useless."""
    counts = df[target].value_counts().sort_index()
    rate = float(df[target].mean())
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["legitimate", "fraud"], counts.values, color=[LEGIT_C, FRAUD_C])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("transactions")
    ax.set_title(
        f"Class balance - fraud = {rate:.2%}\n"
        f"'always predict legitimate' scores {1 - rate:.1%} accuracy",
        fontsize=11,
    )
    _save(fig, out_dir, "01_class_balance.png")


def plot_fraud_over_time(ts: pd.DataFrame, out_dir: Path) -> None:
    """Q: is fraud stationary? A: no -> a random split is indefensible."""
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    a1.plot(ts["day"], ts["fraud_rate"] * 100, color=FRAUD_C, marker="o", ms=3)
    mean_rate = ts["fraud"].sum() / ts["n"].sum() * 100
    a1.axhline(mean_rate, ls="--", c="grey", label="period mean")
    a1.set_ylabel("fraud rate (%)")
    a1.set_title("Fraud rate over time (training period, weekly buckets)", fontsize=11)
    a1.legend(fontsize=9)
    a2.bar(ts["day"], ts["n"], width=5, color=LEGIT_C, alpha=0.75)
    a2.set_ylabel("volume")
    a2.set_xlabel("days since first transaction")
    _save(fig, out_dir, "02_fraud_over_time.png")


def plot_amount_distribution(
    df: pd.DataFrame, target: str, amount: str, out_dir: Path
) -> None:
    """Q: do fraudsters spend differently? Log scale - money is right-skewed."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    legit = df.loc[df[target] == 0, amount].dropna()
    fraud = df.loc[df[target] == 1, amount].dropna()
    top = float(max(legit.max(), fraud.max()))
    bins = np.logspace(0, np.log10(top), 60)
    a1.hist(legit, bins=bins, alpha=0.6, label="legitimate", color=LEGIT_C, density=True)
    a1.hist(fraud, bins=bins, alpha=0.6, label="fraud", color=FRAUD_C, density=True)
    a1.set_xscale("log")
    a1.set_xlabel(f"{amount} (log)")
    a1.set_ylabel("density")
    a1.set_title("Amount distribution by class", fontsize=11)
    a1.legend(fontsize=9)

    a2.boxplot([legit, fraud], tick_labels=["legitimate", "fraud"], showfliers=False)
    a2.set_yscale("log")
    a2.set_ylabel(f"{amount} (log)")
    a2.set_title("Median + IQR (outliers hidden)", fontsize=11)
    _save(fig, out_dir, "03_amount_distribution.png")


def plot_missingness(df: pd.DataFrame, out_dir: Path) -> None:
    """Q: how much data is absent, and is that absence structured?"""
    pct = (df.isna().mean() * 100).sort_values(ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(range(len(pct)), pct.values, color=FRAUD_C)
    a1.fill_between(range(len(pct)), pct.values, alpha=0.3, color=FRAUD_C)
    a1.set_xlabel("columns (sorted)")
    a1.set_ylabel("% missing")
    a1.set_title("Missingness profile - note the plateaus", fontsize=11)
    a2.hist(pct.values, bins=40, color=LEGIT_C)
    a2.set_xlabel("% missing")
    a2.set_ylabel("number of columns")
    a2.set_title(
        "Columns clump at shared missingness levels\n"
        "= correlated blocks, not independent features",
        fontsize=11,
    )
    _save(fig, out_dir, "04_missingness.png")


def plot_missing_as_signal(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    """Q: does ABSENCE predict fraud? The LEFT-join justification, visualised."""
    base = float(df[target].mean())
    cols = [
        c
        for c in ["id_01", "id_31", "DeviceType", "DeviceInfo", "dist1", "dist2", "D15"]
        if c in df.columns
    ]
    miss_r, pres_r = [], []
    for c in cols:
        m = df[c].isna()
        miss_r.append(float(df.loc[m, target].mean()) * 100 if m.any() else 0.0)
        pres_r.append(float(df.loc[~m, target].mean()) * 100 if (~m).any() else 0.0)
    x = np.arange(len(cols))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, miss_r, w, label="value MISSING", color=FRAUD_C)
    ax.bar(x + w / 2, pres_r, w, label="value present", color=LEGIT_C)
    ax.axhline(base * 100, ls="--", c="black", lw=1, label=f"baseline {base:.2%}")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_ylabel("fraud rate (%)")
    ax.set_title(
        "Missingness IS signal - why we LEFT-joined instead of INNER", fontsize=11
    )
    ax.legend(fontsize=9)
    _save(fig, out_dir, "05_missing_as_signal.png")


def plot_categorical_rates(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    """Q: which product / card / device segments carry elevated risk?"""
    cols = [c for c in ["ProductCD", "card4", "card6", "DeviceType"] if c in df.columns]
    base = float(df[target].mean())
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        g = (
            df.groupby(col, observed=True)[target]
            .agg(n="size", r="mean")
            .query("n >= 500")
            .sort_values("r", ascending=False)
        )
        ax.barh(
            g.index.astype(str),
            g["r"] * 100,
            color=[FRAUD_C if v > base else LEGIT_C for v in g["r"]],
        )
        ax.axvline(base * 100, ls="--", c="black", lw=1)
        ax.set_xlabel("fraud rate (%)")
        ax.set_title(col, fontsize=11)
        ax.invert_yaxis()
    fig.suptitle("Fraud rate by segment (dashed = baseline)", fontsize=12)
    _save(fig, out_dir, "06_categorical_rates.png")


def plot_hour_cycle(
    df: pd.DataFrame, target: str, time_col: str, out_dir: Path
) -> None:
    """Q: is there a daily rhythm?

    TransactionDT is seconds from an unknown origin, so `// 3600 % 24` gives a
    valid RELATIVE hour: the offset is unknown but constant, so the shape of
    the cycle is real even though the absolute clock time is not.
    """
    hour = ((df[time_col] // 3600) % 24).astype(int)
    g = df.groupby(hour)[target].agg(n="size", r="mean")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(g.index, g["r"] * 100, marker="o", color=FRAUD_C)
    ax.axhline(float(df[target].mean()) * 100, ls="--", c="grey", label="baseline")
    ax2 = ax.twinx()
    ax2.bar(g.index, g["n"], alpha=0.2, color=LEGIT_C)
    ax2.set_ylabel("volume", color=LEGIT_C)
    ax2.grid(False)
    ax.set_xlabel("hour of day (relative - origin unknown but constant)")
    ax.set_ylabel("fraud rate (%)", color=FRAUD_C)
    ax.set_title(
        "Daily cycle: fraud rate vs legitimate volume", fontsize=11
    )
    ax.legend(fontsize=9, loc="upper right")
    _save(fig, out_dir, "07_hour_of_day.png")


def make_all(train: pd.DataFrame, cfg, *, ts: pd.DataFrame, out_dir: Path) -> None:
    log.info("rendering figures ...")
    plot_class_balance(train, cfg.target, out_dir)
    plot_fraud_over_time(ts, out_dir)
    plot_amount_distribution(train, cfg.target, cfg.amount_column, out_dir)
    plot_missingness(train, out_dir)
    plot_missing_as_signal(train, cfg.target, out_dir)
    plot_categorical_rates(train, cfg.target, out_dir)
    plot_hour_cycle(train, cfg.target, cfg.time_column, out_dir)
