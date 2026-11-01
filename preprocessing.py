"""Cleaning, feature scaling and the MusicData container used by the whole app."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loader import AUDIO_FEATURES, map_columns
from .mood_analysis import assign_moods

UNIT_FEATURES = ["danceability", "energy", "valence", "acousticness",
                 "instrumentalness", "speechiness", "liveness"]


@dataclass(eq=False)
class MusicData:
    df: pd.DataFrame                 # cleaned tracks (index 0..n-1)
    features: list                   # audio features actually present
    X: np.ndarray                    # standardised features (z-scores)
    Xn: np.ndarray                   # X with every row scaled to length 1 (fast cosine)
    scaler: StandardScaler
    has_genre: bool
    has_pop: bool
    has_mood: bool
    source: str
    is_demo: bool
    notes: dict = field(default_factory=dict)
    radar_ref: dict = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.source}|{len(self.df)}"

    @property
    def n_genres(self) -> int:
        return int(self.df["track_genre"].nunique()) if self.has_genre else 0


def prepare(raw: pd.DataFrame, source: str, is_demo: bool = False) -> MusicData:
    """Turn any reasonable Spotify-style dataframe into a MusicData object."""
    rows_in = len(raw)
    df, mapping = map_columns(raw)

    if "track_name" not in df:
        raise ValueError(
            "No song-name column found. Expected one of: track_name, name, song, title. "
            f"Columns in your file: {list(raw.columns)[:15]}"
        )
    features = [f for f in AUDIO_FEATURES if f in df]
    if len(features) < 3:
        raise ValueError(
            f"Only {len(features)} audio features found ({features}). "
            "At least 3 of danceability, energy, valence, acousticness, tempo ... are needed."
        )

    # ---- types & missing values ---------------------------------------
    for col in features + ["popularity", "duration_ms"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["track_name"] = df["track_name"].astype(str).str.strip()
    df = df[~df["track_name"].str.lower().isin(["", "nan", "none"])]
    df = df.dropna(subset=features, how="all")

    if "artists" in df:
        df["artists"] = df["artists"].fillna("Unknown artist").astype(str).str.replace(";", ", ", regex=False)
    else:
        df["artists"] = "Unknown artist"

    has_genre = "track_genre" in df
    if has_genre:
        df["track_genre"] = df["track_genre"].fillna("Unknown").astype(str).str.strip()
        has_genre = df["track_genre"].nunique() > 1
    else:
        df["track_genre"] = "Unknown"

    has_pop = "popularity" in df
    if has_pop:
        df["popularity"] = df["popularity"].fillna(df["popularity"].median()).round().astype(int)
    else:
        df["popularity"] = 0

    # ---- duplicates: same song+artist (keep the most popular copy) -----
    df["dup_key"] = df["track_name"].str.lower() + "||" + df["artists"].str.lower()
    if has_pop:
        df = df.sort_values("popularity", ascending=False, kind="stable")
    before = len(df)
    df = df.drop_duplicates("dup_key", keep="first")
    if "track_id" in df:
        df = df.drop_duplicates("track_id", keep="first")
    dups_removed = before - len(df)
    df = df.reset_index(drop=True)

    # remaining gaps in audio features -> column median
    df[features] = df[features].fillna(df[features].median())

    # ---- standardise (needed for K-Means, PCA and cosine similarity) ---
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features]).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-9)

    df["search"] = (df["track_name"] + " " + df["artists"]).str.lower()
    df["label"] = df["track_name"] + " — " + df["artists"]

    moods = assign_moods(df)
    has_mood = moods is not None
    df["mood"] = moods if has_mood else "Unavailable"

    # reference ranges so tempo / loudness can sit on a 0-1 radar chart
    radar_ref = {f: (0.0, 1.0) for f in UNIT_FEATURES if f in features}
    for f in ("tempo", "loudness"):
        if f in features:
            radar_ref[f] = (float(df[f].quantile(0.01)), float(df[f].quantile(0.99)))

    notes = {
        "rows_in": rows_in,
        "rows_out": len(df),
        "duplicates_removed": dups_removed,
        "mapping": mapping,
        "missing_features": [f for f in AUDIO_FEATURES if f not in features],
    }
    return MusicData(df=df, features=features, X=X, Xn=Xn, scaler=scaler,
                     has_genre=has_genre, has_pop=has_pop, has_mood=has_mood,
                     source=source, is_demo=is_demo, notes=notes, radar_ref=radar_ref)
