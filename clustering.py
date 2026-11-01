"""K-Means clustering + PCA projection + plain-language cluster descriptions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# (phrase when the cluster is HIGH on the feature, phrase when LOW)
PHRASES = {
    "danceability": ("danceable", "less danceable"),
    "energy": ("high-energy", "low-energy"),
    "valence": ("upbeat and positive", "dark or moody"),
    "acousticness": ("acoustic", "electronic / produced"),
    "instrumentalness": ("mostly instrumental", "vocal-driven"),
    "speechiness": ("speech-heavy (rap / spoken word)", "melodic, little speech"),
    "tempo": ("fast tempo", "slow tempo"),
    "loudness": ("loud", "quiet"),
    "liveness": ("live-sounding", "studio-polished"),
}


def fit_pca(X: np.ndarray, sample_size: int = 8000, seed: int = 42) -> dict:
    """Reduce the standardised features to 2 principal components."""
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    return {"coords": coords, "sample": np.sort(sample),
            "explained": pca.explained_variance_ratio_, "n_features": X.shape[1]}


def fit_kmeans(X: np.ndarray, k: int, seed: int = 42) -> KMeans:
    return KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X)


def describe_clusters(df: pd.DataFrame, features: list, labels: np.ndarray) -> dict:
    """Profile each cluster by comparing its feature means with the whole dataset."""
    work = df[features].copy()
    work["cluster"] = labels
    means = work.groupby("cluster")[features].mean()
    g_mean, g_std = df[features].mean(), df[features].std().replace(0, 1)
    z = (means - g_mean) / g_std
    sizes = work["cluster"].value_counts().sort_index()

    descriptions = []
    for c in means.index:
        ranked = z.loc[c].abs().sort_values(ascending=False)
        top = [f for f in ranked.index if ranked[f] >= 0.35][:3]
        if top:
            words = [PHRASES[f][0] if z.loc[c, f] > 0 else PHRASES[f][1] for f in top]
            title = ", ".join(words)
            detail = "; ".join(f"{f} {means.loc[c, f]:.2f} (dataset avg {g_mean[f]:.2f})" for f in top)
        else:
            title = "close to the dataset average"
            detail = "no feature differs strongly from the overall average"
        descriptions.append({
            "cluster": int(c), "size": int(sizes[c]), "share": float(sizes[c] / len(df)),
            "title": title, "detail": detail,
        })
    return {"means": means, "z": z, "descriptions": descriptions}


def build_bundle(data, k: int, pca_result: dict) -> dict:
    """Everything the Song Galaxy page needs for one value of k."""
    km = fit_kmeans(data.X, k)
    labels = km.labels_
    rng = np.random.default_rng(0)
    idx = rng.choice(len(labels), size=min(5000, len(labels)), replace=False)
    sil = float(silhouette_score(data.X[idx], labels[idx])) if len(set(labels[idx])) > 1 else float("nan")
    info = describe_clusters(data.df, data.features, labels)
    return {"k": k, "labels": labels, "inertia": float(km.inertia_), "silhouette": sil, **info}
