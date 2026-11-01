"""MUSIQ - Intelligent Music Discovery & Personality Engine.

Run with:  streamlit run app.py
"""
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import clustering, data_loader, music_personality as mp, preprocessing, recommendation as rec
from src.mood_analysis import MOOD_ORDER, MOOD_RULES

st.set_page_config(page_title="MUSIQ", page_icon="🎧", layout="wide", initial_sidebar_state="expanded")

PAGES = ["Home Dashboard", "Musical DNA", "Song Galaxy", "Smart Recommendations",
         "Mood Explorer", "Genre Escape", "About the Project"]
PALETTE = ["#8b5cf6", "#3b82f6", "#22d3ee", "#e879f9", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#60a5fa", "#fb7185"]

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 600px at 88% -8%, rgba(124,58,237,.20), transparent 60%),
    radial-gradient(900px 500px at -8% 8%, rgba(37,99,235,.15), transparent 55%),
    #0c0a14;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1280px; }
h1, h2, h3, h4 { font-family: 'Sora', 'Segoe UI', sans-serif !important; letter-spacing: -0.01em; }

section[data-testid="stSidebar"] { background: #0a0812; border-right: 1px solid rgba(255,255,255,.06); }
.brand { font-family:'Sora',sans-serif; font-weight:800; font-size:1.9rem; letter-spacing:.04em;
  background: linear-gradient(90deg,#a78bfa,#60a5fa 60%,#22d3ee); -webkit-background-clip:text; background-clip:text; color:transparent; }
.brand-sub { color:#8f93b3; font-size:.82rem; margin-top:-4px; margin-bottom:12px; }
section[data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
section[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 9px 14px; border-radius: 12px; width: 100%; border: 1px solid transparent; cursor: pointer; }
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(139,92,246,.10); }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, rgba(139,92,246,.30), rgba(59,130,246,.18)); border-color: rgba(139,92,246,.45); }
section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }

.hero { padding: 34px 38px; border-radius: 26px; border: 1px solid rgba(255,255,255,.08);
  background: linear-gradient(135deg, rgba(139,92,246,.22), rgba(59,130,246,.10) 55%, rgba(34,211,238,.06)); margin-bottom: 22px; }
.hero h1 { font-size: 4rem; font-weight: 800; margin: 0; line-height: 1;
  background: linear-gradient(90deg,#c4b5fd,#818cf8 45%,#38bdf8); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero .tag { font-family:'Sora',sans-serif; font-size:1.25rem; color:#e8e9f5; margin: 10px 0 12px; }
.hero p { color:#aab0d0; max-width: 760px; line-height:1.6; margin:0; }

.page-title { font-family:'Sora',sans-serif; font-size:2rem; font-weight:700; margin-bottom:2px; }
.page-sub { color:#9a9fc0; margin-bottom:20px; max-width: 820px; }

.stat { min-height: 104px; padding: 18px 20px; border-radius: 18px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); }
.stat .v { font-family:'Sora',sans-serif; font-size:1.9rem; font-weight:700; color:#f4f3ff; }
.stat .l { color:#9a9fc0; font-size:.85rem; margin-top:2px; }
.card { padding: 18px 22px; border-radius: 18px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); margin-bottom: 12px; }
.card h4 { margin: 0 0 6px 0; font-size: 1.05rem; }
.card p, .card li { color:#b5b9d6; font-size:.93rem; line-height:1.55; }
.demo-banner { padding: 12px 18px; border-radius: 14px; background: rgba(251,191,36,.10); border: 1px solid rgba(251,191,36,.35); color:#fde68a; margin-bottom: 16px; }

.track { display:flex; align-items:center; gap:14px; padding: 12px 16px; margin-bottom: 8px; border-radius: 16px;
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.07); }
.track .rank { width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  font-family:'Sora',sans-serif; font-weight:700; background: linear-gradient(135deg,#8b5cf6,#3b82f6); color:white; flex-shrink:0; }
.track .meta { flex:1; min-width:0; }
.track .t { font-weight:600; color:#f4f3ff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.track .a { color:#9a9fc0; font-size:.88rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.track .why { color:#8ea0ff; font-size:.8rem; margin-top:3px; }
.chip { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size:.76rem; margin-left:6px;
  background: rgba(139,92,246,.18); border: 1px solid rgba(139,92,246,.4); color:#ddd6fe; }
.chip.pop { background: rgba(59,130,246,.15); border-color: rgba(59,130,246,.4); color:#bfdbfe; }
.sim { text-align:right; min-width:64px; font-family:'Sora',sans-serif; font-weight:700; font-size:1.15rem; color:#a5f3fc; }
.sim small { display:block; font-family:'DM Sans',sans-serif; font-weight:400; font-size:.7rem; color:#8f93b3; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }

.stButton > button { border-radius: 12px; border: 1px solid rgba(139,92,246,.5);
  background: linear-gradient(90deg, rgba(139,92,246,.35), rgba(59,130,246,.25)); color: #f4f3ff; font-weight: 600; }
.stButton > button:hover { border-color:#a78bfa; background: linear-gradient(90deg, rgba(139,92,246,.6), rgba(59,130,246,.4)); color:white; }
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {
  border-radius: 12px !important; background: rgba(255,255,255,.05) !important; }
button[data-baseweb="tab"] { font-weight: 600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def md(s: str):
    st.markdown(s, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def plot(fig, key=None):
    """st.plotly_chart that works on both old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def style(fig, height=380):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Segoe UI, sans-serif", color="#cbd0e6"),
        margin=dict(l=10, r=10, t=45, b=10), height=height, colorway=PALETTE,
        title=dict(font=dict(family="Sora, Segoe UI, sans-serif", size=16)),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def page_header(title, sub):
    md(f'<div class="page-title">{esc(title)}</div><div class="page-sub">{esc(sub)}</div>')


def stat_card(value, label):
    return f'<div class="stat"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div></div>'


def go_to(page):
    st.session_state["page"] = page


def track_cards(df: pd.DataFrame, reasons=None, show_match=True, ranked=True):
    """Render a dataframe (track_name, artists, track_genre, popularity[, similarity]) as cards."""
    parts = []
    for n, (_, r) in enumerate(df.iterrows(), start=1):
        chips = f'<span class="chip">{esc(r["track_genre"])}</span>' if r["track_genre"] != "Unknown" else ""
        if data.has_pop:
            chips += f'<span class="chip pop">popularity {int(r["popularity"])}</span>'
        why = f'<div class="why">{esc(reasons[n - 1])}</div>' if reasons else ""
        sim = ""
        if show_match and "similarity" in r:
            sim = f'<div class="sim">{max(r["similarity"], 0) * 100:.0f}%<small>match</small></div>'
        rank = f'<div class="rank">{n}</div>' if ranked else ""
        parts.append(
            f'<div class="track">{rank}<div class="meta"><div class="t">{esc(r["track_name"])}</div>'
            f'<div class="a">{esc(r["artists"])}</div>{why}</div><div>{chips}</div>{sim}</div>'
        )
    md("".join(parts))


def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def radar(data_, series: dict, title="", height=420):
    """series = {name: profile_dict}; plotted on a 0-1 scale."""
    fig = go.Figure()
    for i, (name, prof) in enumerate(series.items()):
        axes, vals = mp.to_radar(data_, prof)
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=axes + axes[:1], name=name, fill="toself",
            line=dict(color=color, width=2.5), fillcolor=hex_rgba(color, 0.25),
        ))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,.12)", tickfont=dict(size=9)),
                                 angularaxis=dict(gridcolor="rgba(255,255,255,.12)")),
                      title=title, legend=dict(orientation="h", y=-0.08))
    return style(fig, height)


# ----------------------------------------------------------------------------
# Data loading (cached - runs once per dataset)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading and cleaning the dataset...")
def load_path(path: str, mtime: float):
    return preprocessing.prepare(data_loader.read_csv_source(path), f"dataset/{Path(path).name}")


@st.cache_resource(show_spinner="Reading your uploaded CSV...")
def load_upload(name: str, size: int, _bytes: bytes):
    return preprocessing.prepare(data_loader.read_csv_source(_bytes), f"Uploaded: {name}")


@st.cache_resource(show_spinner=False)
def load_demo():
    return preprocessing.prepare(data_loader.make_demo_data(), "DEMO (synthetic data)", is_demo=True)


@st.cache_resource(show_spinner="Running PCA...")
def get_pca(_data, uid: str):
    return clustering.fit_pca(_data.X)


@st.cache_resource(show_spinner="Clustering songs with K-Means...")
def get_clusters(_data, uid: str, k: int):
    return clustering.build_bundle(_data, k, get_pca(_data, uid))


def load_data():
    with st.sidebar:
        md('<div class="brand">MUSIQ</div><div class="brand-sub">Discover the science behind your sound.</div>')
    nav = st.sidebar.radio("Navigate", PAGES, key="page", label_visibility="collapsed")

    st.sidebar.markdown("---")
    upload = st.sidebar.file_uploader("Spotify tracks CSV", type=["csv"],
                                      help="Or place the file in the dataset/ folder and restart.")
    error, data_ = None, None
    try:
        if upload is not None:
            data_ = load_upload(upload.name, upload.size, upload.getvalue())
        elif (local := data_loader.find_local_csv()) is not None:
            data_ = load_path(str(local), local.stat().st_mtime)
    except Exception as exc:  # show a friendly message instead of crashing
        error = str(exc)
    if data_ is None:
        data_ = load_demo()
    if error:
        st.sidebar.error(f"Could not use that file: {error}")

    tag = "Synthetic demo data" if data_.is_demo else data_.source
    st.sidebar.caption(f"Data source: **{tag}**  \n{len(data_.df):,} tracks loaded")
    return nav, data_


# ----------------------------------------------------------------------------
# Reusable song pickers
# ----------------------------------------------------------------------------
def pick_one(data_, key: str):
    q = st.text_input("Search for a song or artist", key=f"{key}_q", placeholder="e.g. shape of you, or: coldplay yellow")
    if not q.strip():
        st.info("Type a song or artist name to start (at least 2 letters).")
        return None
    hits = rec.search_tracks(data_, q)
    if len(hits) == 0:
        st.warning("No songs matched that search. Try fewer or different words.")
        return None
    labels = data_.df["label"].to_numpy()
    genres = data_.df["track_genre"].to_numpy()
    return int(st.selectbox(f"{len(hits)} match(es) shown (most popular first)", [int(h) for h in hits],
                            format_func=lambda i: f"{labels[i]}  [{genres[i]}]", key=f"{key}_sel"))


def favourites_picker(data_, scope: str):
    """Shared favourites list used by Musical DNA, Recommendations and Genre Escape."""
    ss = st.session_state
    ss.setdefault("fav", [])
    ss.setdefault("fav_v", 0)
    labels = data_.df["label"].to_numpy()
    genres = data_.df["track_genre"].to_numpy()

    def add(idx):
        if idx is not None and idx not in ss["fav"]:
            ss["fav"] = ss["fav"] + [idx]
            ss["fav_v"] += 1

    def clear():
        ss["fav"] = []
        ss["fav_v"] += 1

    def add_random():
        pool = data_.df.nlargest(max(500, len(data_.df) // 20), "popularity").index.to_numpy() if data_.has_pop \
            else np.arange(len(data_.df))
        picks = np.random.default_rng().choice(pool, size=min(5, len(pool)), replace=False)
        ss["fav"] = list(dict.fromkeys(ss["fav"] + [int(p) for p in picks]))
        ss["fav_v"] += 1

    c1, c2 = st.columns([3, 2])
    with c1:
        choice = pick_one(data_, f"{scope}_fav")
    with c2:
        st.write("")
        st.write("")
        b1, b2, b3 = st.columns(3)
        b1.button("Add song", key=f"{scope}_add", on_click=add, args=(choice,), disabled=choice is None)
        b2.button("Clear all", key=f"{scope}_clr", on_click=clear)
        b3.button("Surprise me", key=f"{scope}_rnd", on_click=add_random, help="Add 5 random popular songs")

    if ss["fav"]:
        kept = st.multiselect("Your favourite songs (remove any with the x)", options=ss["fav"], default=ss["fav"],
                              format_func=lambda i: f"{labels[i]}  [{genres[i]}]", key=f"{scope}_ms_{ss['fav_v']}")
        ss["fav"] = list(kept)
    return list(ss["fav"])


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home(data_):
    md('<div class="hero"><h1>MUSIQ</h1><div class="tag">Discover the science behind your sound.</div>'
       '<p>MUSIQ reads the audio fingerprint of every track - energy, danceability, mood, acousticness, tempo - '
       'and uses machine learning to group songs, recommend similar music, map your musical DNA and help you '
       'escape your usual genre.</p></div>')
    if data_.is_demo:
        md('<div class="demo-banner"><b>Demo mode - synthetic data.</b> The real Spotify dataset was not found. '
           'Upload the CSV in the sidebar, or place it in the <code>dataset/</code> folder and restart. '
           'All numbers below come from generated sample data, not real songs.</div>')

    df = data_.df
    k = st.session_state.setdefault("k", 5)
    clusters = get_clusters(data_, data_.uid, k)
    cols = st.columns(5)
    stats = [(f"{len(df):,}", "Tracks analysed"),
             (f"{data_.n_genres}" if data_.has_genre else "n/a", "Genres"),
             (f"{clusters['k']}", "Song clusters (K-Means)"),
             (f"{df['energy'].mean():.2f}" if "energy" in df else "n/a", "Average energy"),
             (f"{df['danceability'].mean():.2f}" if "danceability" in df else "n/a", "Average danceability")]
    for c, (v, l) in zip(cols, stats):
        c.markdown(stat_card(v, l), unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Jump into a feature")
    tiles = [("Musical DNA", "Pick favourites, see your audio profile on a radar chart."),
             ("Song Galaxy", "Explore K-Means clusters on an interactive PCA map."),
             ("Smart Recommendations", "Find 10 songs that sound like the ones you love."),
             ("Mood Explorer", "Browse music by mood, using transparent rules."),
             ("Genre Escape", "Same feeling, different genre. Break out of your bubble.")]
    tcols = st.columns(5)
    for c, (name, desc) in zip(tcols, tiles):
        with c:
            md(f'<div class="card" style="height:150px"><h4 style="font-size:.98rem">{esc(name)}</h4><p>{esc(desc)}</p></div>')
            st.button("Open", key=f"tile_{name}", on_click=go_to, args=(name,))

    st.write("")
    left, right = st.columns(2)
    with left:
        feat = st.selectbox("Audio feature distribution", data_.features, key="home_feat")
        fig = go.Figure(go.Histogram(x=df[feat], nbinsx=50, marker_color="#8b5cf6", opacity=0.9))
        fig.update_layout(title=f"Distribution of {feat}", bargap=0.03, xaxis_title=feat, yaxis_title="tracks")
        plot(style(fig, 360))
    with right:
        if data_.has_mood:
            counts = df["mood"].value_counts().reindex(MOOD_ORDER).dropna()
            fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55,
                                   marker=dict(colors=[MOOD_RULES[m]["color"] for m in counts.index]),
                                   textinfo="label+percent"))
            fig.update_layout(title="Mood distribution (rule-based labels)", showlegend=False)
            plot(style(fig, 360))
    left, right = st.columns(2)
    with left:
        corr = df[data_.features].corr()
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
                                   colorscale=[[0, "#2563eb"], [0.5, "#17142a"], [1, "#a855f7"]],
                                   text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=10)))
        fig.update_layout(title="Correlation between audio features", yaxis=dict(autorange="reversed"))
        plot(style(fig, 460))
        if st.toggle("Show the Seaborn / Matplotlib version", key="sns_toggle"):
            import matplotlib.pyplot as plt
            import seaborn as sns
            f, ax = plt.subplots(figsize=(6, 4.6))
            f.patch.set_facecolor("#0c0a14")
            ax.set_facecolor("#0c0a14")
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
            ax.tick_params(colors="#cbd0e6")
            st.pyplot(f)
            plt.close(f)
    with right:
        if data_.has_genre:
            top = df["track_genre"].value_counts().head(15).sort_values()
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                   marker=dict(color=top.values, colorscale=[[0, "#3b82f6"], [1, "#a855f7"]])))
            fig.update_layout(title="Top 15 genres by number of tracks", xaxis_title="tracks")
            plot(style(fig, 460))
        else:
            st.info("No genre column in this dataset, so the genre chart is unavailable.")

    with st.expander("Data cleaning summary"):
        n = data_.notes
        st.write(f"Rows in file: **{n['rows_in']:,}**  |  rows after cleaning: **{n['rows_out']:,}**  |  "
                 f"duplicate songs removed: **{n['duplicates_removed']:,}**")
        st.write("Audio features used: " + ", ".join(data_.features))
        if n["missing_features"]:
            st.warning("Not found in your file and skipped: " + ", ".join(n["missing_features"]))


def page_dna(data_):
    page_header("Musical DNA", "Choose songs you love. MUSIQ averages their audio features into a profile and describes it.")
    fav = favourites_picker(data_, "dna")
    if not fav:
        st.info("Add at least one song above (or press Surprise me) to see your Musical DNA.")
        return

    prof = mp.user_profile(data_, fav)
    avg = mp.dataset_profile(data_)
    shown = [f for f in ["danceability", "energy", "valence", "acousticness", "instrumentalness", "tempo"] if f in prof]
    cols = st.columns(len(shown))
    for c, f in zip(cols, shown):
        val = f"{prof[f]:.0f} BPM" if f == "tempo" else f"{prof[f]:.2f}"
        c.markdown(stat_card(val, f"Avg {f}  (dataset {avg[f]:.0f})" if f == "tempo" else f"Avg {f}  (dataset {avg[f]:.2f})"),
                   unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        plot(radar(data_, {"Your DNA": prof, "Dataset average": avg}, "Your audio profile vs the dataset average"))
        st.caption("Tempo and loudness are rescaled to 0-1 using the dataset's 1st-99th percentile range so every axis fits one chart.")
    with right:
        rep = mp.generate_report(prof)
        items = "".join(f"<li>{esc(p)}</li>" for p in rep["points"])
        md(f'<div class="card"><h4>{esc(rep["title"])}</h4><ul>{items}</ul><p><i>{esc(rep["disclaimer"])}</i></p></div>')
        k = st.session_state.get("k", 5)
        cl = get_clusters(data_, data_.uid, k)
        counts = pd.Series(cl["labels"][fav]).value_counts()
        txt = ", ".join(f"Cluster {c + 1} ({n} song{'s' if n > 1 else ''})" for c, n in counts.items())
        md(f'<div class="card"><h4>Where your songs live</h4><p>In the K-Means map (k={k}) your songs fall in: {esc(txt)}.</p></div>')
        st.button("Find songs like my DNA", key="dna_to_rec", on_click=go_to, args=("Smart Recommendations",))
    st.markdown("#### Your selected songs")
    sel = data_.df.iloc[fav][["track_name", "artists", "track_genre", "popularity"]]
    track_cards(sel, show_match=False)


def page_galaxy(data_):
    page_header("Song Galaxy", "K-Means groups songs by their audio features. PCA squashes the features into two axes so the groups can be drawn.")
    k = st.slider("Number of clusters (k)", 2, 10, st.session_state.get("k", 5), key="k_slider")
    st.session_state["k"] = k
    cl = get_clusters(data_, data_.uid, k)
    pca = get_pca(data_, data_.uid)

    c = st.columns(4)
    c[0].markdown(stat_card(f"{pca['explained'].sum() * 100:.0f}%", f"Variance kept by 2 PCA axes (from {pca['n_features']} features)"), unsafe_allow_html=True)
    c[1].markdown(stat_card("n/a" if np.isnan(cl["silhouette"]) else f"{cl['silhouette']:.2f}", "Silhouette score (5,000-song sample)"), unsafe_allow_html=True)
    c[2].markdown(stat_card(f"{cl['inertia']:,.0f}", "K-Means inertia (lower = tighter)"), unsafe_allow_html=True)
    c[3].markdown(stat_card(f"{k}", "Clusters"), unsafe_allow_html=True)
    st.write("")

    s = pca["sample"]
    plot_df = data_.df.iloc[s]
    fig = go.Figure()
    for cid in range(k):
        m = cl["labels"][s] == cid
        fig.add_trace(go.Scattergl(
            x=pca["coords"][s][m, 0], y=pca["coords"][s][m, 1], mode="markers", name=f"Cluster {cid + 1}",
            marker=dict(size=5, color=PALETTE[cid % len(PALETTE)], opacity=0.7),
            customdata=np.stack([plot_df["track_name"].to_numpy()[m], plot_df["artists"].to_numpy()[m],
                                 plot_df["track_genre"].to_numpy()[m]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}<extra>Cluster " + str(cid + 1) + "</extra>"))
    fig.update_layout(title=f"Song map: {len(s):,} songs sampled from {len(data_.df):,} (PCA, coloured by K-Means cluster)",
                      xaxis_title=f"PC1 ({pca['explained'][0] * 100:.0f}% of variance)",
                      yaxis_title=f"PC2 ({pca['explained'][1] * 100:.0f}% of variance)")
    plot(style(fig, 560), key="galaxy")

    st.markdown("#### What each cluster sounds like")
    st.caption("Descriptions are generated from the cluster's actual feature averages compared with the whole dataset.")
    per_row = 3
    for start in range(0, k, per_row):
        row = st.columns(per_row)
        for col, d in zip(row, cl["descriptions"][start:start + per_row]):
            colr = PALETTE[d["cluster"] % len(PALETTE)]
            col.markdown(f'<div class="card"><h4><span class="dot" style="background:{colr}"></span>Cluster {d["cluster"] + 1}: {esc(d["title"])}</h4>'
                         f'<p><b>{d["size"]:,}</b> songs ({d["share"] * 100:.1f}%)<br>{esc(d["detail"])}</p></div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        sizes = [d["size"] for d in cl["descriptions"]]
        fig = go.Figure(go.Bar(x=[f"Cluster {i + 1}" for i in range(k)], y=sizes,
                               marker_color=[PALETTE[i % len(PALETTE)] for i in range(k)]))
        fig.update_layout(title="Songs per cluster", yaxis_title="songs")
        plot(style(fig, 380))
    with right:
        z = cl["z"]
        fig = go.Figure(go.Heatmap(z=z.values, x=list(z.columns), y=[f"Cluster {i + 1}" for i in z.index],
                                   colorscale=[[0, "#2563eb"], [0.5, "#17142a"], [1, "#d946ef"]], zmid=0,
                                   text=np.round(z.values, 1), texttemplate="%{text}", colorbar=dict(title="z-score")))
        fig.update_layout(title="Cluster comparison: feature mean vs dataset average (z-score)", yaxis=dict(autorange="reversed"))
        plot(style(fig, 380))

    st.markdown("#### Look inside a cluster")
    pick = st.selectbox("Cluster", list(range(k)), format_func=lambda i: f"Cluster {i + 1}: {cl['descriptions'][i]['title']}", key="gal_cluster")
    members = data_.df[cl["labels"] == pick]
    a, b = st.columns(2)
    with a:
        if data_.has_genre:
            g = members["track_genre"].value_counts().head(10).sort_values()
            fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=PALETTE[pick % len(PALETTE)]))
            fig.update_layout(title="Top genres in this cluster", xaxis_title="songs")
            plot(style(fig, 380))
    with b:
        st.caption("Most popular songs in this cluster" if data_.has_pop else "Sample songs in this cluster")
        top = members.nlargest(8, "popularity") if data_.has_pop else members.head(8)
        track_cards(top, show_match=False)


def rec_results(data_, selected, results, title="Recommendations", reasons=None):
    if results.empty:
        st.warning("No songs matched these filters. Try relaxing them.")
        return
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"#### {title}")
        track_cards(results, reasons=reasons)
    with right:
        base = mp.user_profile(data_, selected)
        got = data_.df.iloc[results["idx"].tolist()][data_.features].mean().to_dict()
        plot(radar(data_, {"Your selection": base, "Recommended (average)": got}, "Selection vs recommendations", 400))


def page_recs(data_):
    page_header("Smart Recommendations", "Content-based filtering: MUSIQ compares standardised audio features with cosine similarity.")
    t1, t2 = st.tabs(["Similar to one song", "Based on my favourites"])
    n = st.session_state.get("rec_n", 10)
    with t1:
        c1, c2 = st.columns([3, 1])
        with c1:
            idx = pick_one(data_, "rec1")
        with c2:
            n = st.slider("How many?", 5, 20, 10, key="rec_n")
        min_pop = st.slider("Minimum popularity", 0, 100, 0, key="rec_pop") if data_.has_pop else 0
        if idx is not None:
            row = data_.df.iloc[idx]
            md(f'<div class="card"><h4>Selected: {esc(row["track_name"])}</h4><p>{esc(row["artists"])} · {esc(row["track_genre"])}</p></div>')
            res = rec.recommend(data_, [idx], n=n, min_popularity=min_pop)
            reasons = [rec.explain_match(data_, [idx], int(i)) for i in res["idx"]] if not res.empty else None
            rec_results(data_, [idx], res, f"Top {len(res)} songs that sound like it", reasons)
    with t2:
        fav = favourites_picker(data_, "rec")
        if not fav:
            st.info("Add one or more favourite songs. MUSIQ will recommend tracks close to their average profile.")
        else:
            res = rec.recommend(data_, fav, n=n)
            reasons = [rec.explain_match(data_, fav, int(i)) for i in res["idx"]] if not res.empty else None
            rec_results(data_, fav, res, f"Top {len(res)} songs for your taste profile", reasons)


def page_mood(data_):
    page_header("Mood Explorer", "Moods come from simple, published rules on valence, energy and acousticness. They are not a trained model.")
    if not data_.has_mood:
        st.error("This dataset needs 'energy' and 'valence' columns for mood labels.")
        return
    df = data_.df
    mood = st.radio("Pick a mood", MOOD_ORDER, horizontal=True, key="mood_pick")
    info = MOOD_RULES[mood]
    subset = df[df["mood"] == mood]
    md(f'<div class="card"><h4><span class="dot" style="background:{info["color"]}"></span>{esc(mood)}: {len(subset):,} songs ({len(subset) / len(df) * 100:.1f}%)</h4>'
       f'<p>{esc(info["blurb"])}<br><b>Rule:</b> <code>{esc(info["rule"])}</code></p></div>')

    c1, c2, c3 = st.columns(3)
    sort = c1.selectbox("Order by", ["Most popular", "Random shuffle"] if data_.has_pop else ["Random shuffle"], key="mood_sort")
    count = c2.slider("Songs to show", 5, 30, 12, key="mood_n")
    genres = c3.multiselect("Limit to genres", sorted(subset["track_genre"].unique()), key="mood_genres") if data_.has_genre else []
    if genres:
        subset = subset[subset["track_genre"].isin(genres)]
    if subset.empty:
        st.warning("No songs match that mood and genre combination.")
    else:
        if sort == "Most popular":
            shown = subset.nlargest(count, "popularity")
        else:
            seed = st.session_state.setdefault("mood_seed", 0)
            st.button("Shuffle again", key="mood_shuffle", on_click=lambda: st.session_state.update(mood_seed=seed + 1))
            shown = subset.sample(min(count, len(subset)), random_state=seed)
        track_cards(shown, show_match=False)

    a, b = st.columns(2)
    with a:
        counts = df["mood"].value_counts().reindex(MOOD_ORDER).dropna()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[MOOD_RULES[m]["color"] for m in counts.index]))
        fig.update_layout(title="Songs per mood", yaxis_title="songs")
        plot(style(fig, 360))
    with b:
        feats = [f for f in ["valence", "energy", "acousticness", "danceability"] if f in data_.features]
        means = df.groupby("mood")[feats].mean().reindex(MOOD_ORDER).dropna()
        fig = go.Figure()
        for i, f in enumerate(feats):
            fig.add_trace(go.Bar(name=f, x=means.index, y=means[f], marker_color=PALETTE[i]))
        fig.update_layout(title="Average audio features by mood", barmode="group", yaxis_title="average value")
        plot(style(fig, 360))


def page_escape(data_):
    page_header("Genre Escape", "Keep the feeling, change the genre. MUSIQ finds songs with a similar sound outside the genres you already listen to.")
    if not data_.has_genre:
        st.error("Genre Escape is unavailable: this dataset has no usable genre column.")
        return
    fav = favourites_picker(data_, "esc")
    if not fav:
        st.info("Add one or more favourite songs to see what you could discover outside your usual genre.")
        return
    df = data_.df
    my_genres = list(dict.fromkeys(df.iloc[fav]["track_genre"]))
    md(f'<div class="card"><h4>Your genres</h4><p>{esc(", ".join(my_genres))} - these are excluded from the results.</p></div>')

    c1, c2, c3 = st.columns(3)
    n = c1.slider("How many songs?", 5, 20, 10, key="esc_n")
    cap = c2.slider("Max songs per genre", 1, 5, 2, key="esc_cap", help="Keeps results varied.")
    options = sorted(set(df["track_genre"].unique()) - set(my_genres))
    target = c3.multiselect("Only explore these genres (optional)", options, key="esc_target")
    res = rec.recommend(data_, fav, n=n, exclude_genres=my_genres, include_genres=target or None, max_per_genre=cap)
    if res.empty:
        st.warning("Nothing found. Try removing the genre filter.")
        return
    reasons = [f"{rec.explain_match(data_, fav, int(i))}. Different genre: {g} instead of {', '.join(my_genres[:2])}."
               for i, g in zip(res["idx"], res["track_genre"])]
    rec_results(data_, fav, res, f"{len(res)} songs from other genres with a similar sound", reasons)
    gcount = res["track_genre"].value_counts()
    fig = go.Figure(go.Bar(x=gcount.index, y=gcount.values, marker_color="#8b5cf6"))
    fig.update_layout(title="Genres you could explore", yaxis_title="songs")
    plot(style(fig, 300))


def page_about(data_):
    page_header("About the Project", "MUSIQ - Intelligent Music Discovery & Personality Engine")
    n = data_.notes
    sections = {
        "Problem statement": "Most music recommenders keep you inside one genre. Genre labels say little about how a song actually sounds, "
                             "so listeners miss music they would enjoy in other genres.",
        "Objectives": "<ul><li>Group songs by sound using unsupervised learning.</li><li>Recommend similar songs from audio features.</li>"
                      "<li>Summarise a listener's taste as a Musical DNA profile.</li><li>Let people explore by mood.</li>"
                      "<li>Recommend similar-sounding songs from different genres (Genre Escape).</li></ul>",
        "Dataset": f"Spotify Tracks Dataset. Currently loaded: <b>{esc(data_.source)}</b>, {len(data_.df):,} tracks after cleaning "
                   f"({n['duplicates_removed']:,} duplicate songs removed). Audio features used: {esc(', '.join(data_.features))}.",
        "Methodology": "<ol><li>Load CSV, auto-map column names, convert types, fill gaps with the median, remove duplicate songs.</li>"
                       "<li>Standardise features with StandardScaler (mean 0, std 1).</li>"
                       "<li>K-Means groups songs; PCA projects them to 2D for plotting.</li>"
                       "<li>Cosine similarity ranks songs against the average of the user's picks.</li>"
                       "<li>Mood labels come from fixed rules; Genre Escape applies genre exclusion on top of cosine similarity.</li></ol>",
    }
    for title, body in sections.items():
        md(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>')
    md('<div class="card"><h4>Algorithms in plain language</h4><ul>'
       '<li><b>StandardScaler:</b> puts every feature on the same scale so tempo (about 120) does not overpower energy (0 to 1).</li>'
       '<li><b>K-Means:</b> places k centres, assigns each song to the nearest one, moves the centres, and repeats until stable.</li>'
       '<li><b>PCA:</b> finds the two directions that keep the most variation, so 9 features can be shown on a flat chart.</li>'
       '<li><b>Cosine similarity:</b> compares the direction of two feature vectors; closer direction means a more similar sound.</li>'
       '<li><b>Mood rules:</b> fixed thresholds, not a trained model:</li></ul>'
       + "<ul>" + "".join(f"<li><b>{esc(m)}</b>: <code>{esc(v['rule'])}</code></li>" for m, v in MOOD_RULES.items()) + "</ul></div>")
    md('<div class="card"><h4>Technologies</h4><p>Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Matplotlib, Seaborn.</p></div>')
    md('<div class="card"><h4>Limitations</h4><ul><li>Audio features only: lyrics, language and artist context are not used.</li>'
       '<li>Silhouette score is computed on a 5,000-song sample for speed.</li>'
       '<li>Mood labels and personality text are descriptive rules, not scientific measurements.</li>'
       '<li>Duplicate songs are collapsed to the most popular copy, so a song keeps only one genre label.</li></ul></div>')


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
st.session_state.setdefault("page", PAGES[0])
nav, data = load_data()
ROUTES = {"Home Dashboard": page_home, "Musical DNA": page_dna, "Song Galaxy": page_galaxy,
          "Smart Recommendations": page_recs, "Mood Explorer": page_mood, "Genre Escape": page_escape,
          "About the Project": page_about}
# Home tiles use short names; map them to full page names
ALIASES = {"Home": "Home Dashboard"}
ROUTES.get(ALIASES.get(nav, nav), page_home)(data)
