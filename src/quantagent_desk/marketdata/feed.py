"""Factor levels through the pinned as-of date.

The desk reads ``data/market/factors.csv`` first. A refresh script is the only
writer. ``QUANTAGENT_MARKET_OFFLINE=1`` refuses Yahoo and FRED and stays on
that cache.

Equity, FX, and WTI are stored as prices. USD 10Y and the corporate OAS are
stored as decimals (FRED publishes both in percent). VIX is a strip level
only; it is not a P&L factor.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional
from urllib.parse import quote, urlencode

import pandas as pd

PINNED_AS_OF = date(2026, 10, 27)
LOOKBACK_SESSIONS = 252
# One extra level row so daily changes cover about 252 sessions.
LEVEL_ROWS = LOOKBACK_SESSIONS + 1
MIN_LEVEL_ROWS = 11
CALENDAR_PAD_DAYS = 500

LEVEL_COLUMNS = ["SPX", "VIX", "EURUSD", "WTI", "USD_10Y", "IG_OAS"]
PRICE_FACTORS = ("SPX", "EURUSD", "WTI")
DIFF_FACTORS = ("USD_10Y", "IG_OAS")

YAHOO_TICKERS = {
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "EURUSD": "EURUSD=X",
    "WTI": "CL=F",
}
FRED_SERIES = {
    "USD_10Y": "DGS10",
    "IG_OAS": "BAMLC0A0CM",
}

LABELS = {
    "SPX": "SPX",
    "VIX": "VIX",
    "USD_10Y": "USD 10Y",
    "EURUSD": "EURUSD",
    "WTI": "WTI",
    "IG_OAS": "IG OAS",
}
NOTES = {
    "IG_OAS": (
        "ICE BofA US corporate OAS (FRED BAMLC0A0CM). "
        "CDX IG is not on a free feed, so this OAS is the credit proxy."
    ),
    "USD_10Y": "FRED DGS10, percent converted to decimal.",
    "VIX": "Strip only. Not a P&L factor for these books.",
    "SPX": "Daily percent change enters equity VaR.",
    "EURUSD": "Daily percent change enters FX VaR.",
    "WTI": "Daily percent change enters commodity VaR.",
}

_OFFLINE_VALUES = {"1", "true", "yes", "on"}
# FRED's edge rejects a custom client string. Curl's own UA is accepted;
# httpx uses a browser UA because Yahoo's chart host expects one.
_HTTPX_UA = "Mozilla/5.0"
_CURL_UA = None


class MarketDataUnavailable(RuntimeError):
    """Cache and live history cannot supply a factor panel."""


class MarketOfflineError(MarketDataUnavailable):
    """Network download was requested while the offline flag is set."""


@dataclass
class MarketSnapshot:
    """Last levels on or before the pinned as-of, plus aligned daily moves."""

    as_of: Optional[date]
    requested_as_of: date
    source: str
    levels: Dict[str, float]
    panel: pd.DataFrame
    moves: pd.DataFrame
    labels: Dict[str, str]
    notes: Dict[str, str]

    def to_public_dict(self) -> Dict[str, object]:
        return {
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "requested_as_of": self.requested_as_of.isoformat(),
            "source": self.source,
            "levels": self.levels,
            "labels": self.labels,
            "notes": self.notes,
            "sessions": int(len(self.moves)),
        }


def is_offline() -> bool:
    return os.environ.get("QUANTAGENT_MARKET_OFFLINE", "").strip().lower() in _OFFLINE_VALUES


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "data").is_dir():
            return parent
    return here.parents[3]


def default_cache_path() -> Path:
    override = os.environ.get("QUANTAGENT_MARKET_CACHE", "").strip()
    if override:
        return Path(override)
    return project_root() / "data" / "market" / "factors.csv"


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    stamps = pd.to_datetime(index)
    if getattr(stamps, "tz", None) is not None:
        stamps = stamps.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(stamps).normalize()


def read_factor_cache(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Return cached levels, or None when the file is missing."""
    cache = path or default_cache_path()
    if not cache.is_file():
        return None
    frame = pd.read_csv(cache)
    if "date" not in frame.columns:
        raise MarketDataUnavailable(f"{cache} has no date column")
    frame["date"] = _normalize_index(frame["date"])
    frame = frame.set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    keep = [column for column in LEVEL_COLUMNS if column in frame.columns]
    return frame[keep].apply(pd.to_numeric, errors="coerce")


def write_factor_cache(frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
    cache = path or default_cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out.index = _normalize_index(out.index)
    out.index.name = "date"
    columns = [column for column in LEVEL_COLUMNS if column in out.columns]
    out[columns].to_csv(cache, date_format="%Y-%m-%d", float_format="%.10g")
    return cache


def slice_sessions(
    frame: pd.DataFrame,
    as_of: date,
    sessions: int = LOOKBACK_SESSIONS,
) -> pd.DataFrame:
    """Last ``sessions`` changes need ``sessions + 1`` level rows, on or before as-of."""
    if frame.empty:
        return frame
    stamp = pd.Timestamp(as_of)
    sliced = frame.loc[frame.index <= stamp].sort_index()
    if sliced.empty:
        return sliced
    return sliced.tail(sessions + 1)


def factor_moves(levels: pd.DataFrame) -> pd.DataFrame:
    """Daily percent changes for prices; daily differences for yields and OAS."""
    moves = pd.DataFrame(index=levels.index)
    for column in PRICE_FACTORS:
        if column in levels.columns:
            moves[column] = pd.to_numeric(levels[column], errors="coerce").pct_change()
    for column in DIFF_FACTORS:
        if column in levels.columns:
            moves[column] = pd.to_numeric(levels[column], errors="coerce").diff()
    moves = moves.replace([float("inf"), float("-inf")], pd.NA)
    return moves.dropna(how="any")


def _last_levels(levels: pd.DataFrame) -> Dict[str, float]:
    if levels.empty:
        return {}
    row = levels.iloc[-1]
    values: Dict[str, float] = {}
    for column in LEVEL_COLUMNS:
        if column not in levels.columns:
            continue
        series = levels[column].dropna()
        if series.empty:
            continue
        raw = row[column] if column in row.index else series.iloc[-1]
        if pd.isna(raw):
            raw = series.iloc[-1]
        values[column] = float(raw)
    return values


def _snapshot_from_levels(
    levels: pd.DataFrame,
    requested_as_of: date,
    source: str,
) -> MarketSnapshot:
    clean = levels.dropna(subset=[column for column in PRICE_FACTORS + DIFF_FACTORS if column in levels.columns])
    if clean.empty:
        raise MarketDataUnavailable("Factor panel has no complete sessions")
    as_of = pd.Timestamp(clean.index.max()).date()
    return MarketSnapshot(
        as_of=as_of,
        requested_as_of=requested_as_of,
        source=source,
        levels=_last_levels(clean),
        panel=clean,
        moves=factor_moves(clean),
        labels=dict(LABELS),
        notes=dict(NOTES),
    )


def load_market_snapshot(
    as_of: Optional[date] = None,
    path: Optional[Path] = None,
) -> MarketSnapshot:
    """Load the cache first. Download only when it cannot cover the as-of date."""
    requested = as_of or PINNED_AS_OF
    cache = path or default_cache_path()
    source = "cache"
    frame: Optional[pd.DataFrame] = None
    cached = read_factor_cache(cache)
    if cached is not None:
        sliced = slice_sessions(cached, requested)
        if len(sliced) >= MIN_LEVEL_ROWS:
            frame = sliced
    if frame is None:
        if is_offline():
            raise MarketDataUnavailable(
                f"No usable factor cache at {cache} and QUANTAGENT_MARKET_OFFLINE=1"
            )
        downloaded = download_factor_levels(requested)
        frame = slice_sessions(downloaded, requested)
        source = "live"
    if frame is None or len(frame) < MIN_LEVEL_ROWS:
        raise MarketDataUnavailable(
            f"Need at least {MIN_LEVEL_ROWS} level rows on or before {requested.isoformat()}"
        )
    return _snapshot_from_levels(frame, requested, source)


def download_factor_levels(as_of: date = PINNED_AS_OF) -> pd.DataFrame:
    """Download Yahoo and FRED history and return the aligned level window."""
    if is_offline():
        raise MarketOfflineError(
            "QUANTAGENT_MARKET_OFFLINE=1; refusing to download market data"
        )
    start = as_of - timedelta(days=CALENDAR_PAD_DAYS)
    yahoo = _download_yahoo(YAHOO_TICKERS, start, as_of)
    fred = _download_fred(FRED_SERIES, start, as_of)
    panel = yahoo.join(fred, how="inner")
    panel = panel.replace([float("inf"), float("-inf")], pd.NA).dropna(how="any")
    panel = panel.sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    missing = [column for column in LEVEL_COLUMNS if column not in panel.columns]
    if missing:
        raise MarketDataUnavailable(f"Download missing columns: {', '.join(missing)}")
    window = slice_sessions(panel[LEVEL_COLUMNS], as_of)
    if len(window) < MIN_LEVEL_ROWS:
        raise MarketDataUnavailable(
            f"Download produced {len(window)} sessions on or before {as_of.isoformat()}"
        )
    return window


def _download_yahoo(tickers: Mapping[str, str], start: date, end: date) -> pd.DataFrame:
    if _python_tls_works():
        try:
            return _download_yahoo_yfinance(tickers, start, end)
        except MarketDataUnavailable:
            pass
    return _download_yahoo_chart(tickers, start, end)


def _download_yahoo_yfinance(tickers: Mapping[str, str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    symbols = list(tickers.values())
    raw = yf.download(
        symbols,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw is None or len(raw) == 0:
        raise MarketDataUnavailable("yfinance returned no prices")
    close = _extract_close(raw, symbols)
    rename = {ticker: name for name, ticker in tickers.items()}
    close = close.rename(columns=rename)
    close.index = _normalize_index(close.index)
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]
    return close.apply(pd.to_numeric, errors="coerce")


def _download_yahoo_chart(tickers: Mapping[str, str], start: date, end: date) -> pd.DataFrame:
    """Yahoo chart endpoint. Used when yfinance cannot verify TLS."""
    period1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    period2 = int(
        datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp()
        + 24 * 60 * 60
    )
    frames = []
    for name, symbol in tickers.items():
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
        payload = json.loads(
            _http_get(
                url,
                {"interval": "1d", "period1": str(period1), "period2": str(period2)},
            )
        )
        result = (payload.get("chart") or {}).get("result") or []
        if not result or not result[0].get("timestamp"):
            error = (payload.get("chart") or {}).get("error")
            raise MarketDataUnavailable(f"Yahoo chart returned no prices for {symbol}: {error}")
        block = result[0]
        stamps = pd.to_datetime(block["timestamp"], unit="s", utc=True).tz_convert(None)
        indicators = block.get("indicators") or {}
        quote_block = (indicators.get("quote") or [{}])[0]
        values = quote_block.get("close")
        adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose")
        if adjusted and any(item is not None for item in adjusted):
            values = adjusted
        series = pd.Series(values, index=_normalize_index(stamps), name=name, dtype="float64")
        series = series[~series.index.duplicated(keep="last")].sort_index()
        frames.append(series)
        time.sleep(0.4)
    return pd.concat(frames, axis=1, sort=True)


def _extract_close(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns:
            raise MarketDataUnavailable(f"Unexpected yfinance columns: {list(raw.columns)}")
        out = raw[["Close"]].copy()
        out.columns = [symbols[0]]
        return out
    level0 = set(raw.columns.get_level_values(0))
    if "Close" in level0:
        return raw["Close"].copy()
    frames = []
    for symbol in symbols:
        if symbol not in level0:
            continue
        block = raw[symbol]
        if "Close" not in block.columns:
            continue
        frames.append(block["Close"].rename(symbol))
    if not frames:
        raise MarketDataUnavailable("yfinance response had no Close prices")
    return pd.concat(frames, axis=1, sort=True)


def _download_fred(series_ids: Mapping[str, str], start: date, end: date) -> pd.DataFrame:
    columns = {}
    for name, series_id in series_ids.items():
        body = _http_get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            {"id": series_id, "cosd": start.isoformat(), "coed": end.isoformat()},
        )
        frame = pd.read_csv(io.StringIO(body.decode("utf-8")))
        date_col = "observation_date" if "observation_date" in frame.columns else "DATE"
        if date_col not in frame.columns:
            raise MarketDataUnavailable(
                f"FRED {series_id} CSV has no date column ({list(frame.columns)})"
            )
        value_col = series_id if series_id in frame.columns else frame.columns[-1]
        values = pd.to_numeric(frame[value_col], errors="coerce") / 100.0
        series = pd.Series(values.to_numpy(), index=_normalize_index(frame[date_col]), name=name)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        columns[name] = series
    return pd.DataFrame(columns)


_TLS_OK: Optional[bool] = None


def _python_tls_works() -> bool:
    """True when this interpreter can verify a public HTTPS certificate."""
    global _TLS_OK
    if _TLS_OK is not None:
        return _TLS_OK
    import httpx

    try:
        httpx.get("https://fred.stlouisfed.org", timeout=8.0, follow_redirects=True)
    except Exception as exc:
        _TLS_OK = not _tls_failure(exc)
    else:
        _TLS_OK = True
    return _TLS_OK


def _tls_failure(exc: BaseException) -> bool:
    text = repr(exc).lower()
    return "certificate" in text or "ssl" in text or "tls" in text


def _http_get(url: str, params: Optional[Mapping[str, str]] = None) -> bytes:
    """GET bytes. Falls back to the OS curl when Python cannot verify TLS."""
    import httpx

    query = url
    if params:
        query = f"{url}?{urlencode(params)}"
    if not _python_tls_works():
        return _curl_get(query)
    try:
        response = httpx.get(
            url,
            params=params,
            headers={"User-Agent": _HTTPX_UA},
            timeout=60.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.content
    except Exception as exc:
        if not _tls_failure(exc):
            raise MarketDataUnavailable(f"Download failed for {url}: {exc}") from exc
        return _curl_get(query)


def _curl_get(url: str) -> bytes:
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if not curl:
        raise MarketDataUnavailable(
            "Python TLS verification failed and curl is not on PATH"
        )
    command = [curl, "--http1.1", "--max-time", "40", "-fsSL", url]
    if _CURL_UA:
        command = [curl, "--http1.1", "-A", _CURL_UA, "--max-time", "40", "-fsSL", url]
    detail = ""
    for attempt in range(3):
        if attempt:
            time.sleep(2 * attempt)
        completed = subprocess.run(command, check=False, capture_output=True)
        if completed.returncode == 0 and completed.stdout:
            return completed.stdout
        detail = completed.stderr.decode("utf-8", errors="replace")[:400]
        retryable = completed.returncode in {22, 28, 56} or "429" in detail
        if not retryable:
            break
    raise MarketDataUnavailable(f"curl failed for {url}: {detail}")
