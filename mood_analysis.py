"""Rule-based mood labelling.

IMPORTANT: these are transparent IF/ELSE rules on valence, energy and
acousticness. They are NOT the output of a trained classifier.
Rules are evaluated top to bottom; the first one that matches wins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOOD_RULES = {
    "Energetic": {
        "rule": "energy >= 0.70",
        "color": "#f472b6",
        "blurb": "Intense, driving tracks with very high energy.",
    },
    "Happy / Positive": {
        "rule": "valence >= 0.60 (and energy < 0.70)",
        "color": "#facc15",
        "blurb": "Bright, cheerful-sounding tracks with a positive feel.",
    },
    "Calm / Relaxed": {
        "rule": "energy < 0.45 and acousticness >= 0.50 and valence >= 0.35",
        "color": "#22d3ee",
        "blurb": "Soft, acoustic, low-energy tracks for winding down.",
    },
    "Melancholic": {
        "rule": "valence < 0.35 and energy < 0.60",
        "color": "#818cf8",
        "blurb": "Low-valence, lower-energy tracks with a sad or moody feel.",
    },
    "Balanced": {
        "rule": "everything that matches none of the rules above",
        "color": "#a78bfa",
        "blurb": "Mid-range tracks that sit between the other moods.",
    },
}
MOOD_ORDER = list(MOOD_RULES)


def assign_moods(df: pd.DataFrame) -> pd.Series | None:
    """Vectorised mood labels, or None if energy/valence are unavailable."""
    if "energy" not in df or "valence" not in df:
        return None
    e = df["energy"].to_numpy()
    v = df["valence"].to_numpy()

    calm = (e < 0.45) & (v >= 0.35)
    if "acousticness" in df:
        calm &= df["acousticness"].to_numpy() >= 0.50

    conditions = [e >= 0.70, v >= 0.60, calm, (v < 0.35) & (e < 0.60)]
    labels = np.select(conditions, MOOD_ORDER[:4], default="Balanced")
    return pd.Series(labels, index=df.index, name="mood")
