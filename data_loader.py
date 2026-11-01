"""Dataset loading, column auto-mapping and synthetic demo data.

Nothing here needs the internet or any Spotify credentials.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "acousticness", "instrumentalness",
    "speechiness", "tempo", "loudness", "liveness",
]

# canonical column -> accepted names in the CSV (checked in this order)
COLUMN_ALIASES = {
    "track_id": ["track_id", "id", "uri", "track_uri"],
    "track_name": ["track_name", "name", "song_name", "song", "title"],
    "artists": ["artists", "artist", "artist_name", "artist_names", "artists_name"],
    "album_name": ["album_name", "album"],
    "track_genre": ["track_genre", "genre", "genres", "playlist_genre", "main_genre"],
    "popularity": ["popularity", "track_popularity"],
    "explicit": ["explicit"],
    "duration_ms": ["duration_ms"],
}
for _f in AUDIO_FEATURES:
    COLUMN_ALIASES[_f] = [_f]


def find_local_csv() -> Path | None:
    """Return the biggest CSV inside dataset/ (the main Spotify file), if any."""
    if not DATASET_DIR.exists():
        return None
    files = sorted(DATASET_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    return files[0] if files else None


def read_csv_source(source) -> pd.DataFrame:
    """Read a CSV from a path or raw bytes, trying common encodings."""
    last_error = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if isinstance(source, (bytes, bytearray)):
                return pd.read_csv(io.BytesIO(source), encoding=enc, low_memory=False)
            return pd.read_csv(source, encoding=enc, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError("The CSV file is empty.") from exc
    raise ValueError(f"Could not decode the CSV file: {last_error}")


def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename whatever the CSV calls its columns to MUSIQ's canonical names.

    Returns the reduced dataframe plus a {canonical: original_name} report.
    """
    lookup = {}
    for col in df.columns:
        lookup.setdefault(str(col).strip().lower().replace(" ", "_"), col)

    rename, report = {}, {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                rename[lookup[alias]] = canonical
                report[canonical] = lookup[alias]
                break
    return df[list(rename)].rename(columns=rename).copy(), report


def make_demo_data(n_per_genre: int = 250, seed: int = 42) -> pd.DataFrame:
    """Small SYNTHETIC dataset so the app can be shown without the real CSV.

    Feature values are random draws around rough per-genre centres.
    They are NOT real Spotify data.
    """
    rng = np.random.default_rng(seed)
    # danceability, energy, valence, acousticness, instrumentalness, speechiness, tempo, loudness, liveness
    centres = {
        "pop":        [0.68, 0.72, 0.60, 0.15, 0.00, 0.06, 120, -6, 0.17],
        "rock":       [0.50, 0.80, 0.50, 0.10, 0.03, 0.06, 126, -6, 0.20],
        "edm":        [0.64, 0.88, 0.45, 0.04, 0.35, 0.07, 128, -5, 0.18],
        "hip-hop":    [0.80, 0.65, 0.55, 0.15, 0.00, 0.25, 115, -7, 0.18],
        "acoustic":   [0.55, 0.30, 0.50, 0.85, 0.02, 0.04, 105, -11, 0.15],
        "classical":  [0.30, 0.15, 0.20, 0.95, 0.85, 0.04, 100, -20, 0.12],
        "jazz":       [0.55, 0.35, 0.55, 0.75, 0.40, 0.05, 110, -12, 0.17],
        "metal":      [0.40, 0.92, 0.35, 0.02, 0.10, 0.08, 135, -5, 0.20],
        "indie":      [0.55, 0.55, 0.45, 0.35, 0.10, 0.04, 118, -8, 0.16],
        "r-n-b":      [0.68, 0.55, 0.50, 0.30, 0.00, 0.08, 108, -8, 0.15],
        "ambient":    [0.30, 0.20, 0.25, 0.70, 0.80, 0.04, 95, -18, 0.12],
        "latin":      [0.75, 0.75, 0.75, 0.25, 0.00, 0.08, 112, -6, 0.17],
    }
    spread = np.array([0.12, 0.12, 0.17, 0.12, 0.08, 0.04, 12, 2.5, 0.07])
    adjectives = ["Neon", "Midnight", "Golden", "Silent", "Electric", "Velvet", "Crimson", "Lunar", "Wild", "Hollow"]
    nouns = ["Echo", "Horizon", "Pulse", "Garden", "Signal", "Mirage", "Harbor", "Skyline", "Ember", "Drift"]

    rows, track_no = [], 0
    for genre, centre in centres.items():
        centre = np.array(centre, dtype=float)
        vals = rng.normal(centre, spread, size=(n_per_genre, len(centre)))
        for v in vals:
            track_no += 1
            rows.append({
                "track_id": f"demo{track_no:05d}",
                "track_name": f"{rng.choice(adjectives)} {rng.choice(nouns)} #{track_no}",
                "artists": f"Demo Artist {int(rng.integers(1, 80))}",
                "album_name": "Synthetic Album",
                "track_genre": genre,
                "popularity": int(np.clip(rng.normal(45, 20), 0, 100)),
                **dict(zip(AUDIO_FEATURES, v)),
            })
    df = pd.DataFrame(rows)
    for col in ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness"]:
        df[col] = df[col].clip(0, 1)
    df["tempo"] = df["tempo"].clip(50, 220)
    df["loudness"] = df["loudness"].clip(-40, 0)
    return df
