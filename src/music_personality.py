"""Musical DNA: average audio profile + a descriptive (not scientific) report."""
from __future__ import annotations

import numpy as np

RADAR_ORDER = ["danceability", "energy", "valence", "acousticness",
               "instrumentalness", "speechiness", "liveness", "tempo", "loudness"]


def user_profile(data, selected) -> dict:
    """Mean raw value of each audio feature over the selected songs."""
    return data.df.iloc[list(selected)][data.features].mean().to_dict()


def dataset_profile(data) -> dict:
    return data.df[data.features].mean().to_dict()


def radar_axes(data) -> list:
    return [f for f in RADAR_ORDER if f in data.features]


def to_radar(data, profile: dict) -> tuple[list, list]:
    """Scale each feature to 0-1 so everything fits on one radar chart."""
    axes = radar_axes(data)
    vals = []
    for f in axes:
        lo, hi = data.radar_ref[f]
        vals.append(float(np.clip((profile[f] - lo) / (hi - lo if hi > lo else 1), 0, 1)))
    return axes, vals


def generate_report(profile: dict) -> dict:
    """Turn the averages into a short, friendly description."""
    g = profile.get
    traits, points = [], []

    dance, energy, val, ac = g("danceability"), g("energy"), g("valence"), g("acousticness")
    inst, speech, tempo = g("instrumentalness"), g("speechiness"), g("tempo")

    if energy is not None and energy >= 0.65 and dance is not None and dance >= 0.65:
        traits.append("Energetic Mover")
        points.append(f"High energy ({energy:.2f}) and high danceability ({dance:.2f}): you lean towards music that makes you move.")
    elif energy is not None and energy >= 0.65:
        traits.append("Intensity Seeker")
        points.append(f"High energy ({energy:.2f}): you prefer powerful, driving tracks.")
    elif energy is not None and energy <= 0.40:
        traits.append("Calm Listener")
        points.append(f"Low energy ({energy:.2f}): you favour gentle, laid-back music.")
    elif dance is not None and dance >= 0.65:
        traits.append("Groove Lover")
        points.append(f"High danceability ({dance:.2f}): rhythm and groove matter to you.")

    if ac is not None and ac >= 0.5:
        traits.append("Acoustic Soul")
        points.append(f"High acousticness ({ac:.2f}): you are drawn to natural, acoustic sounds over electronic production.")
    if inst is not None and inst >= 0.25:
        traits.append("Instrumental Explorer")
        points.append(f"High instrumentalness ({inst:.2f}): you are comfortable with music that has few or no vocals.")
    if speech is not None and speech >= 0.2:
        traits.append("Lyric Follower")
        points.append(f"High speechiness ({speech:.2f}): rap or spoken-word style tracks feature in your picks.")
    if val is not None and val >= 0.6:
        traits.append("Sunny Optimist")
        points.append(f"High valence ({val:.2f}): you prefer positive, cheerful-sounding music.")
    elif val is not None and val <= 0.35:
        traits.append("Moody Dreamer")
        points.append(f"Low valence ({val:.2f}): you are drawn to darker, more emotional-sounding music.")
    if tempo is not None and tempo >= 125:
        points.append(f"Fast average tempo ({tempo:.0f} BPM).")
    elif tempo is not None and tempo <= 95:
        points.append(f"Slow average tempo ({tempo:.0f} BPM).")

    if not traits:
        traits = ["Balanced Explorer"]
        points.append("Your picks sit near the middle on most audio features: you enjoy a wide mix of styles.")

    return {
        "title": "The " + " · ".join(traits[:2]) if len(traits) > 1 else "The " + traits[0],
        "points": points,
        "disclaimer": ("This is a descriptive summary of the audio features of the songs you chose. "
                       "It is not a scientific measurement of personality."),
    }
