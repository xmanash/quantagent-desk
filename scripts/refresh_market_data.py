#!/usr/bin/env python3
"""Download factor history and rewrite data/market/factors.csv.

One successful run is what CI reads. QUANTAGENT_MARKET_OFFLINE=1 refuses
the network and leaves the cache untouched.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent_desk.marketdata.feed import (  # noqa: E402
    PINNED_AS_OF,
    MarketDataUnavailable,
    MarketOfflineError,
    download_factor_levels,
    write_factor_cache,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the committed factor cache")
    parser.add_argument(
        "--as-of",
        default=PINNED_AS_OF.isoformat(),
        help="Keep the last close on or before this date (YYYY-MM-DD)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Cache CSV path")
    args = parser.parse_args(argv)
    as_of = date.fromisoformat(args.as_of)
    try:
        levels = download_factor_levels(as_of)
    except MarketOfflineError as exc:
        print(exc, file=sys.stderr)
        return 2
    except MarketDataUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    path = write_factor_cache(levels, args.output)
    last = levels.index.max().date().isoformat()
    print(f"Wrote {len(levels)} sessions through {last} to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
