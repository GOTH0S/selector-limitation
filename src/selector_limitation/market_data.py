from __future__ import annotations

import csv
import hashlib
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SOURCE_COMMIT = "3187e756957af549e54e470e8064dbe782d319b8"
SOURCE_URL = (
    "https://raw.githubusercontent.com/marcoreyess22/jump-risk-engine/"
    f"{SOURCE_COMMIT}/data/prices.csv"
)
SOURCE_SHA256 = "0f1c7534aed1afc5d431be99f5c0357243e7357905e60b74b40a8711689f71bd"
UNIVERSE = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "DBC")

FloatArray = NDArray[np.float64]
DateArray = NDArray[np.datetime64]


@dataclass(frozen=True)
class PricePanel:
    dates: DateArray
    tickers: tuple[str, ...]
    prices: FloatArray

    def __post_init__(self) -> None:
        if self.prices.ndim != 2:
            raise ValueError("prices must be two-dimensional")
        if self.prices.shape != (self.dates.shape[0], len(self.tickers)):
            raise ValueError("date/ticker axes do not match prices")
        if not np.isfinite(self.prices).all() or np.any(self.prices <= 0):
            raise ValueError("prices must be finite and positive")
        if np.any(self.dates[1:] <= self.dates[:-1]):
            raise ValueError("dates must be strictly increasing")


def _verified_bytes(payload: bytes) -> bytes:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"price snapshot hash mismatch: {digest}")
    return payload


def download_snapshot(cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        _verified_bytes(cache_path.read_bytes())
        return cache_path

    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "selector-limitation/0.1 research replication"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    cache_path.write_bytes(_verified_bytes(payload))
    return cache_path


def parse_prices(text: str, *, end_date: str = "2019-12-31") -> PricePanel:
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    if tuple(header[1:]) != UNIVERSE or header[0] != "Date":
        raise ValueError(f"unexpected price header: {header}")

    cutoff = np.datetime64(end_date, "D")
    dates: list[np.datetime64] = []
    rows: list[list[float]] = []
    for row in reader:
        if not row:
            continue
        date = np.datetime64(row[0], "D")
        if date > cutoff:
            break
        dates.append(date)
        rows.append([float(value) for value in row[1:]])

    if not rows:
        raise ValueError("price snapshot contains no rows before cutoff")
    return PricePanel(
        dates=np.asarray(dates, dtype="datetime64[D]"),
        tickers=UNIVERSE,
        prices=np.asarray(rows, dtype=np.float64),
    )


def load_snapshot(cache_path: Path, *, end_date: str = "2019-12-31") -> PricePanel:
    path = download_snapshot(cache_path)
    return parse_prices(path.read_text(encoding="utf-8"), end_date=end_date)
