"""Project configuration and path resolution.

Why this exists
---------------
Every later stage (EDA, split, training, API) needs to agree on *where* data
lives and *what* the key/target/time columns are called. If each script
hard-codes its own paths, the project silently drifts apart. So there is
exactly one config file (configs/data.yaml) and one loader: this module.

Paths in the YAML are relative to the project root, so the code runs the same
way from a notebook, a script, a test, or a Docker container.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards until we find the directory that owns pyproject.toml.

    This makes the project location-independent: no os.chdir, no '../..'
    relative paths that break the moment a notebook moves.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root (no pyproject.toml found above "
        f"{here}). Are you running inside the RiskLens repo?"
    )


@dataclass(frozen=True)
class DataConfig:
    """Typed, resolved view of configs/data.yaml."""

    raw: dict[str, Any]
    root: Path

    # -- directories (absolute) --
    @property
    def raw_dir(self) -> Path:
        return self.root / self.raw["paths"]["raw"]

    @property
    def interim_dir(self) -> Path:
        return self.root / self.raw["paths"]["interim"]

    @property
    def processed_dir(self) -> Path:
        return self.root / self.raw["paths"]["processed"]

    @property
    def reports_dir(self) -> Path:
        return self.root / self.raw["paths"]["reports"]

    # -- input files (absolute) --
    @property
    def transaction_csv(self) -> Path:
        return self.raw_dir / self.raw["files"]["transaction"]

    @property
    def identity_csv(self) -> Path:
        return self.raw_dir / self.raw["files"]["identity"]

    # -- output files (absolute) --
    @property
    def joined_parquet(self) -> Path:
        return self.interim_dir / self.raw["outputs"]["joined"]

    @property
    def manifest_path(self) -> Path:
        return self.reports_dir / self.raw["outputs"]["manifest"]

    # -- schema --
    @property
    def join_key(self) -> str:
        return self.raw["schema"]["join_key"]

    @property
    def target(self) -> str:
        return self.raw["schema"]["target"]

    @property
    def time_column(self) -> str:
        return self.raw["schema"]["time_column"]

    @property
    def amount_column(self) -> str:
        return self.raw["schema"]["amount_column"]

    # -- contract / expectations --
    @property
    def contract(self) -> dict[str, Any]:
        return self.raw["contract"]

    @property
    def expected(self) -> dict[str, Any]:
        return self.raw["expected"]

    # -- temporal split (Stage 3) --
    @property
    def split(self) -> dict[str, Any]:
        return self.raw["split"]


@lru_cache(maxsize=1)
def load_data_config(path: str | Path | None = None) -> DataConfig:
    """Load configs/data.yaml once and cache it."""
    root = find_project_root()
    cfg_path = Path(path) if path else root / "configs" / "data.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return DataConfig(raw=raw, root=root)
