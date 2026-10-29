"""
BOM/Open-Meteo hourly temperature downloader for the study window.

BOM's own climate-data service was unavailable (as of retrieval date), so we
use the Open-Meteo historical archive API (free, no key) for Sydney
(Observatory Hill coordinates ~ -33.86, 151.21). Output is an hourly CSV
indexed by Sydney-local time, which `data_loader.load_temperature` resamples
to 30-minute periods.

Output:
  data/raw/temperature_hourly.csv   columns: [timestamp (local), temperature]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_DIR as RAW_DIR  # noqa: E402

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = -33.86, 151.21
START = "2025-01-01"
END = "2026-07-01"


def fetch(start: str, end: str, url: str = ARCHIVE_URL, timeout: int = 120) -> pd.DataFrame:
    """Hourly Sydney temperature for start..end (YYYY-MM-DD, inclusive) from an
    Open-Meteo endpoint (the historical archive by default)."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "timezone": "Australia/Sydney",
    }
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": "dsh-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    times = pd.to_datetime(data["hourly"]["time"], format="%Y-%m-%dT%H:%M")
    temp = data["hourly"]["temperature_2m"]
    return pd.DataFrame({"temperature": temp}, index=pd.DatetimeIndex(times, name="timestamp"))


def main() -> None:
    t0 = time.perf_counter()
    # Fetch in yearly chunks (API is friendlier in smaller requests)
    chunks = []
    for year in (2025, 2026):
        s = f"{year}-01-01"
        e = f"{year}-12-31" if year == 2025 else END
        print(f"[temp] fetching {s} .. {e}")
        chunks.append(fetch(s, e))
    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.index.name = "timestamp"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "temperature_hourly.csv"
    df.to_csv(out)
    print(f"[temp] Wrote {len(df)} hourly rows to {out}")
    print(f"[temp] Range: {df.index.min()} .. {df.index.max()}")
    print(f"[temp] temp stats: min={df.temperature.min():.1f} max={df.temperature.max():.1f} "
          f"mean={df.temperature.mean():.1f}")
    print(f"[temp] Elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
