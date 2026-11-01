"""Content-based recommendations with cosine similarity.

Similarity is computed ONLY for the query (one matrix-vector product),
never as a full 114k x 114k matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESULT_COLUMNS = ["idx", "track_name", "artists", "track_genre", "popularity", "similarity"]


def search_tracks(data, query: str, limit: int = 50) -> np.ndarray:
    """Row indices of tracks whose name/artist contains every word of the query."""
    tokens = [t for t in query.lower().split() if t]
    if not tokens or sum(len(t) for t in tokens) < 2:
        return np.array([], dtype=int)
    mask = np.ones(len(data.df), dtype=bool)
    for tok in tokens:
        mask &= data.df["search"].str.contains(tok, regex=False, na=False).to_numpy()
    idx = np.flatnonzero(mask)
    if idx.size and data.has_pop:
        idx = idx[np.argsort(-data.df["popularity"].to_numpy()[idx], kind="stable")]
    return idx[:limit]


def profile_vector(data, selected) -> np.ndarray:
    """Average standardised feature vector of the selected songs."""
    return data.X[list(selected)].mean(axis=0)


def recommend(data, selected, n=10, exclude_genres=None, include_genres=None,
              min_popularity=0, max_per_genre=None) -> pd.DataFrame:
    """Top-n tracks by cosine similarity to the average of the selected songs."""
    selected = list(dict.fromkeys(int(i) for i in selected))
    if not selected:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    q = profile_vector(data, selected)
    q = q / max(np.linalg.norm(q), 1e-9)
    sims = data.Xn @ q                                   # cosine similarity to every track

    df = data.df
    mask = np.ones(len(df), dtype=bool)
    mask[selected] = False                               # never recommend the input songs
    mask &= ~df["dup_key"].isin(set(df["dup_key"].iloc[selected])).to_numpy()
    if exclude_genres:
        mask &= ~df["track_genre"].isin(exclude_genres).to_numpy()
    if include_genres:
        mask &= df["track_genre"].isin(include_genres).to_numpy()
    if min_popularity and data.has_pop:
        mask &= df["popularity"].to_numpy() >= min_popularity

    cand = np.flatnonzero(mask)
    if cand.size == 0:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    order = cand[np.argsort(-sims[cand], kind="stable")]

    genres = df["track_genre"].to_numpy()
    names = df["track_name"].str.lower().to_numpy()
    seen_titles, per_genre, picked = set(), {}, []
    for i in order:
        if names[i] in seen_titles:                      # avoid same title appearing twice
            continue
        if max_per_genre and per_genre.get(genres[i], 0) >= max_per_genre:
            continue
        picked.append(i)
        seen_titles.add(names[i])
        per_genre[genres[i]] = per_genre.get(genres[i], 0) + 1
        if len(picked) >= n:
            break

    out = df.iloc[picked][["track_name", "artists", "track_genre", "popularity"]].copy()
    out.insert(0, "idx", picked)
    out["similarity"] = sims[picked]
    return out.reset_index(drop=True)


def _fmt(feature: str, value: float) -> str:
    if feature == "tempo":
        return f"{value:.0f} BPM"
    if feature == "loudness":
        return f"{value:.1f} dB"
    return f"{value:.2f}"


def explain_match(data, selected, rec_idx: int, top: int = 3) -> str:
    """Plain-language reason: the features where the track is closest to the profile."""
    q = profile_vector(data, selected)
    diff = np.abs(data.X[rec_idx] - q)
    best = np.argsort(diff)[:top]
    raw_profile = data.df.iloc[list(selected)][data.features].mean()
    parts = [
        f"{data.features[j]} ({_fmt(data.features[j], data.df.iloc[rec_idx][data.features[j]])}"
        f" vs {_fmt(data.features[j], raw_profile[data.features[j]])})"
        for j in best
    ]
    return "Closest on " + ", ".join(parts)
