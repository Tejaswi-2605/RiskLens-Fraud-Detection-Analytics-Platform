"""Fetch the IEEE-CIS Fraud Detection dataset into data/raw/.

The raw data is deliberately NOT committed to git (see .gitignore): it is
~1.3 GB and it is public. Reproducibility comes from this script plus the
sha256 hashes recorded in reports/stage01_ingest_manifest.json, not from
storing the bytes in the repo.

Prerequisites
-------------
1. A Kaggle account.
2. Accept the competition rules once, in a browser:
       https://www.kaggle.com/c/ieee-fraud-detection/rules
   The API returns 403 until you do - this is a rules gate, not a bug.
3. An API token: Kaggle -> Settings -> API -> "Create New Token".
   Save the downloaded kaggle.json to:
       Windows:  %USERPROFILE%\\.kaggle\\kaggle.json
       Linux/Mac: ~/.kaggle/kaggle.json

Usage
-----
    python scripts/download_data.py
    python scripts/download_data.py --force     # re-download even if present
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risklens.config import load_data_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("download")

MANUAL_INSTRUCTIONS = """
Automatic download unavailable. Download manually instead:

  1. Open   https://www.kaggle.com/c/ieee-fraud-detection/data
  2. Accept the competition rules.
  3. Download `train_transaction.csv` and `train_identity.csv`.
  4. Place both files (uncompressed) in:
         {raw_dir}

Then re-run:  python scripts/run_ingest.py
"""


def kaggle_credentials_present() -> bool:
    return (Path.home() / ".kaggle" / "kaggle.json").is_file()


def download(competition: str, filename: str, dest_dir: Path) -> bool:
    """Download one competition file via the Kaggle CLI. Returns success."""
    log.info("downloading %s ...", filename)
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle", "competitions", "download",
            "-c", competition, "-f", filename, "-p", str(dest_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("kaggle CLI failed:\n%s\n%s", result.stdout, result.stderr)
        return False

    # Kaggle serves single files zipped when they are large.
    zipped = dest_dir / f"{filename}.zip"
    if zipped.is_file():
        log.info("extracting %s", zipped.name)
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(dest_dir)
        zipped.unlink()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    cfg = load_data_config()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    competition = cfg.raw["dataset"]["kaggle_competition"]

    targets = [cfg.transaction_csv, cfg.identity_csv]
    needed = [t for t in targets if args.force or not t.is_file()]

    if not needed:
        for t in targets:
            log.info("already present: %s (%.1f MB)", t.name, t.stat().st_size / 1024**2)
        return 0

    if not kaggle_credentials_present():
        log.warning("no ~/.kaggle/kaggle.json found")
        print(MANUAL_INSTRUCTIONS.format(raw_dir=cfg.raw_dir))
        return 1

    ok = all(download(competition, t.name, cfg.raw_dir) for t in needed)
    if not ok:
        print(MANUAL_INSTRUCTIONS.format(raw_dir=cfg.raw_dir))
        return 1

    for t in targets:
        if t.is_file():
            log.info("ready: %s (%.1f MB)", t.name, t.stat().st_size / 1024**2)
        else:
            log.error("missing after download: %s", t)
            return 1

    log.info("done. Next:  python scripts/run_ingest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
