"""Stage 1 entry point: build data/interim/transactions_joined.parquet.

Usage
-----
    python scripts/run_ingest.py            # full run, writes parquet + manifest
    python scripts/run_ingest.py --dry-run  # validate only, write nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risklens.config import load_data_config  # noqa: E402
from risklens.data.ingest import ingest  # noqa: E402
from risklens.data.validate import DataContractError  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage01")


def summarise(manifest) -> None:
    m = asdict(manifest)
    line = "-" * 66
    print(f"\n{line}\nSTAGE 1 - INGESTION SUMMARY\n{line}")
    print(f"  transaction  : {m['shapes']['transaction'][0]:>9,} rows x {m['shapes']['transaction'][1]:>3} cols")
    print(f"  identity     : {m['shapes']['identity'][0]:>9,} rows x {m['shapes']['identity'][1]:>3} cols")
    print(f"  joined       : {m['shapes']['joined'][0]:>9,} rows x {m['shapes']['joined'][1]:>3} cols")
    print()
    print(f"  fraud        : {m['target']['positives']:>9,} of {m['shapes']['joined'][0]:,}"
          f"  ({m['target']['fraud_rate']:.3%})")
    print(f"  imbalance    : 1 fraud per {m['target']['imbalance_ratio']:.1f} legitimate")
    print()
    print(f"  identity cov : {m['join']['identity_coverage']:.2%}"
          f"  ({m['join']['identity_rows_matched']:,} matched)")
    print(f"  time span    : {m['time']['min']:,} .. {m['time']['max']:,} sec"
          f"  ({m['time']['span_days']:.1f} days)")
    print()
    print(f"  in memory    : {m['memory']['joined_mb']:.1f} MB")
    if "parquet_mb" in m["memory"]:
        print(f"  on disk      : {m['memory']['parquet_mb']:.1f} MB (parquet+snappy)")
    print(f"  timings      : {json.dumps(m['timings_sec'])}")
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate without writing")
    args = parser.parse_args()

    cfg = load_data_config()
    try:
        _df, manifest = ingest(cfg, write=not args.dry_run)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except DataContractError as exc:
        log.error("DATA CONTRACT VIOLATED: %s", exc)
        return 2

    summarise(manifest)
    if args.dry_run:
        print("dry run - nothing written")
    else:
        print(f"wrote {cfg.joined_parquet.relative_to(cfg.root).as_posix()}")
        print(f"wrote {cfg.manifest_path.relative_to(cfg.root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
