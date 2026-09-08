"""TrustRUL — dashboard for uncertainty-aware turbofan RUL prediction (C-MAPSS FD001).

Runs on a precomputed artifact bundle (app/artifacts/), so the default deploy needs
only pandas / numpy / plotly / streamlit.  If `tensorflow` is importable, the
"Live MC-Dropout" tab additionally re-samples the confidence band on demand.

    streamlit run app/streamlit_app.py

Coursework demo (Module M507). Not for operational maintenance decisions.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"
DATA = Path(__file__).parent / "data"
HAS_TF = importlib.util.find_spec("tensorflow") is not None

PAPER, INK, SOFT = "#fbfaf7", "#1c1b18", "#4f4c45"
ACCENT, AUTO, FLAG = "#7c5a2e", "#5c6f3c", "#b0521f"

st.set_page_config(page_title="TrustRUL dashboard", page_icon="✈️", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{ background:{PAPER}; }}
    h1,h2,h3,h4 {{ color:{INK}; font-family:"Source Serif 4",Georgia,serif; }}
    [data-testid="stMetricValue"] {{ font-variant-numeric:tabular-nums; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:2px; }}
    </style>""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------- loaders ---
@st.cache_data
def load_meta() -> dict:
    return json.loads((ART / "meta.json").read_text())


@st.cache_data
def load_record() -> pd.DataFrame:
    return pd.read_csv(ART / "transparency_record.csv")


@st.cache_data
def load_series() -> pd.DataFrame:
    return pd.read_csv(ART / "test_sensor_series.csv.gz")


@st.cache_data
def load_experiments() -> pd.DataFrame:
    return pd.read_csv(ART / "experiments.csv")


@st.cache_data
def load_attention() -> pd.DataFrame:
    return pd.read_csv(ART / "attention_profile.csv")


meta = load_meta()
record = load_record()
series = load_series()
SENSORS = meta["keep_sensors"]
WIN = meta["win"]
THRESH = meta["review_threshold_band"]

# --------------------------------------------------------------------- head ---
st.title("TrustRUL — turbofan remaining-useful-life with a transparency record")
st.caption(
    "NASA C-MAPSS FD001 · best model: "
    f"{meta['best']['arch'].upper()} (window {WIN}, dropout {meta['best']['dropout']}, "
    f"{meta['best']['loss']} loss) · Monte Carlo Dropout ×{meta['mc_passes']} · "
    "coursework demo, not for operational use"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Test RMSE (cycles)", f"{meta['test_rmse']:.2f}")
k2.metric("C-MAPSS asymmetric score", f"{meta['test_score']:,}")
k3.metric("Engines routed to human review", f"{meta['n_flagged']} / 100")
k4.metric("Review threshold (90% band)", f"> {THRESH:.1f} cyc")

# ------------------------------------------------------------------ sidebar ---
st.sidebar.header("Engine")
only_flag = st.sidebar.toggle("Show only human-review engines", value=False)
pool = record[record.review == "HUMAN"] if only_flag else record
engine = st.sidebar.selectbox("Test engine", pool.engine.tolist(),
                              format_func=lambda e: f"Engine {e:>3}")
row = record.loc[record.engine == engine].iloc[0]
st.sidebar.markdown(
    f"**Predicted RUL** {row.pred_RUL:.1f} cycles  \n"
    f"**90% interval** {row.CI90_low:.0f} – {row.CI90_high:.0f}  \n"
    f"**Band width** {row.band:.1f} cycles  \n"
    f"**True RUL** {row.true_RUL:.0f}  ·  **abs error** {row.abs_err:.1f}"
)
st.sidebar.divider()
st.sidebar.caption(
    "EU AI Act framing: Annex I / Art. 6(1) high-risk. The confidence band (Art. 13) "
    "and the automatic human-review routing (Art. 14) are model outputs, not add-ons."
)

tab_rec, tab_fleet, tab_cmp, tab_live, tab_about = st.tabs(
    ["Transparency record", "Fleet view", "Model comparison", "Live MC-Dropout", "About"]
)

# ============================================================ transparency ===
with tab_rec:
    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"Engine {engine}")
        flag = row.review == "HUMAN"
        st.markdown(
            f"<div style='padding:.6rem 1rem;border-radius:8px;"
            f"background:{'#f6e3d6' if flag else '#e7ecdd'};color:{INK};font-weight:600'>"
            f"{'⚑ ROUTE TO HUMAN REVIEWER' if flag else '✓ automatic — within confidence tolerance'}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[row.CI90_low, row.CI90_high], y=["RUL", "RUL"], mode="lines",
            line=dict(color=ACCENT, width=10), opacity=0.35, name="90% MC interval",
            hovertemplate="90%% interval: %{x:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[row.pred_RUL], y=["RUL"], mode="markers", name="prediction",
            marker=dict(color=ACCENT, size=16, symbol="diamond")))
        fig.add_trace(go.Scatter(
            x=[row.true_RUL], y=["RUL"], mode="markers", name="true RUL",
            marker=dict(color=INK, size=13, symbol="line-ns", line=dict(width=3, color=INK))))
        fig.update_layout(
            height=180, margin=dict(l=10, r=10, t=10, b=60),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="cycles remaining", yaxis=dict(showticklabels=False, title=None),
            legend=dict(orientation="h", y=-0.8, x=0), font=dict(color=INK))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"**Top contributing sensors** (input-gradient saliency): "
            f"`{row.top_sensors}`"
        )

    with right:
        st.subheader("Sensor trajectory")
        g = series[series.engine == engine]
        pick = st.multiselect("Sensors", SENSORS,
                              default=[s.strip() for s in row.top_sensors.split(",")])
        if pick:
            f2 = go.Figure()
            for s in pick:
                col = g[s].astype(float)
                z = (col - col.mean()) / (col.std() or 1.0)          # standardise: compare shape, not level
                f2.add_trace(go.Scatter(x=g.cycle, y=z, mode="lines", name=s))
            f2.add_vrect(x0=g.cycle.max() - WIN + 1, x1=g.cycle.max(),
                         fillcolor=ACCENT, opacity=0.08, line_width=0,
                         annotation_text=f"model input window ({WIN})", annotation_position="top left")
            f2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             xaxis_title="cycle", yaxis_title="standardised value (z)",
                             font=dict(color=INK), legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(f2, use_container_width=True)
        st.caption(
            "Each sensor is standardised (z-score) so trends are comparable across the very "
            "different raw scales. Shaded = the last-window sequence fed to the model; the "
            "engine's series ends before failure and RUL is inferred from the window's shape."
        )

# =================================================================== fleet ===
with tab_fleet:
    st.subheader("All 100 test engines")
    c1, c2 = st.columns([3, 2])

    with c1:
        f = go.Figure()
        for lab, col in (("auto", AUTO), ("HUMAN", FLAG)):
            d = record[record.review == lab]
            f.add_trace(go.Scatter(
                x=d.pred_RUL, y=d.band, mode="markers", name=lab,
                marker=dict(color=col, size=9, line=dict(width=0.5, color="white")),
                customdata=d.engine,
                hovertemplate="engine %{customdata}<br>pred %{x:.0f}<br>band %{y:.1f}<extra></extra>"))
        f.add_hline(y=THRESH, line_dash="dot", line_color=INK,
                    annotation_text=f"review threshold {THRESH:.1f}")
        f.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="predicted RUL (cycles)", yaxis_title="90% band width (cycles)",
                        font=dict(color=INK), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(f, use_container_width=True)
        st.caption("Uncertainty grows with predicted RUL — long-horizon predictions are the least certain, "
                   "and the threshold sends that quartile to a planner.")

    with c2:
        e_auto = record[record.review == "auto"].abs_err.mean()
        e_flag = record[record.review == "HUMAN"].abs_err.mean()
        st.metric("Mean abs error — auto", f"{e_auto:.1f}")
        st.metric("Mean abs error — flagged", f"{e_flag:.1f}")
        st.caption(
            f"On this test set the flagged (widest-band) engines do **not** carry larger error "
            f"than the auto set ({e_flag:.1f} vs {e_auto:.1f} cycles): MC-Dropout band width and "
            f"actual error are only loosely coupled here. The flag is still a defensible Art. 14 "
            f"oversight policy — route the least-confident predictions to a human — but it is not a "
            f"calibrated error predictor. See *About*."
        )

    st.dataframe(
        record.style.apply(
            lambda s: [f"background-color:{'#f6e3d6' if v == 'HUMAN' else ''}" for v in s],
            subset=["review"]),
        use_container_width=True, height=330)

# ============================================================== comparison ===
with tab_cmp:
    exp = load_experiments()
    st.subheader("§5 experiment table — 12 configurations")
    st.dataframe(exp, use_container_width=True, height=460)

    c1, c2 = st.columns(2)
    with c1:
        ex = exp.sort_values("val_RMSE")
        bar = go.Figure(go.Bar(
            x=ex.val_RMSE, y=[f"{r.arch} w{r.win} d{r.dropout} {r.loss}/{r.norm}/{r.sensors}"
                              for r in ex.itertuples()],
            orientation="h", marker_color=[ACCENT if a == "gru" else "#b8ad97" for a in ex.arch]))
        bar.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), title="validation RMSE",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=INK), yaxis=dict(autorange="reversed"))
        st.plotly_chart(bar, use_container_width=True)
    with c2:
        att = load_attention()
        line = go.Figure(go.Scatter(x=att.window_pos, y=att.mean_attention, mode="lines+markers",
                                    line=dict(color=ACCENT)))
        line.add_hline(y=1 / WIN, line_dash="dot", line_color=SOFT,
                       annotation_text="uniform (1/window)")
        line.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                           title="1D-CNN+attention — mean weight by window position",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis_title="position in window (oldest → newest)",
                           yaxis_title="mean attention", font=dict(color=INK))
        st.plotly_chart(line, use_container_width=True)
        st.caption("Attention stays close to uniform and does not concentrate on recent cycles — "
                   "consistent with this architecture's weak fit (RMSE 22 vs GRU 12).")

# ============================================================ live MC panel ===
with tab_live:
    st.subheader("Re-sample the confidence band")
    if not HAS_TF:
        st.info(
            "TensorFlow is not installed in this deployment, so the band shown elsewhere is the "
            "precomputed 50-pass result. To enable live re-sampling, add `tensorflow-cpu` to "
            "`requirements.txt` (see `requirements-live.txt`) and redeploy."
        )
    else:
        import tensorflow as tf  # noqa
        from tensorflow import keras
        import pickle

        @st.cache_resource
        def _load_model():
            m = keras.models.load_model(ART / "model.keras", compile=False)
            sc = pickle.loads((ART / "scaler.pkl").read_bytes())
            return m, sc

        import trustrul_core as tc

        model, scaler = _load_model()
        n_pass = st.slider("MC-Dropout passes", 10, 300, 50, step=10)
        raw = tc.load_txt(DATA / "test_FD001.txt")
        g = raw[raw.unit == engine].copy()
        g[SENSORS] = scaler.transform(g[SENSORS])
        X, _, _ = tc.make_windows(g.assign(unit=engine), SENSORS, WIN, last_only=True)
        P = tc.mc_predict(model, X, n=n_pass)[:, 0]
        lo, hi = np.percentile(P, [5, 95])

        cc1, cc2 = st.columns([3, 2])
        with cc1:
            h = go.Figure(go.Histogram(x=P, nbinsx=30, marker_color=ACCENT, opacity=0.8))
            h.add_vline(x=P.mean(), line_color=INK, annotation_text=f"mean {P.mean():.1f}")
            h.add_vline(x=row.true_RUL, line_color=FLAG, line_dash="dash",
                        annotation_text=f"true {row.true_RUL:.0f}")
            h.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="predicted RUL (cycles)", font=dict(color=INK))
            st.plotly_chart(h, use_container_width=True)
        with cc2:
            st.metric("Live mean RUL", f"{P.mean():.1f}")
            st.metric("Live 90% band", f"{lo:.0f} – {hi:.0f}  ({hi-lo:.1f})")
            st.metric("Precomputed band", f"{row.CI90_low:.0f} – {row.CI90_high:.0f}  ({row.band:.1f})")
            st.caption("Small differences from the precomputed band are the stochastic nature of "
                       "MC-Dropout; they shrink as passes increase.")

# =================================================================== about ===
with tab_about:
    st.markdown(
        f"""
**What this is.** A read-out of the TrustRUL pipeline (Module M507 assessment): a GRU trained on
NASA C-MAPSS **FD001** predicts remaining useful life for 100 held-out turbofan engines, and every
prediction ships with a *transparency record* — point RUL, a Monte Carlo Dropout 90% interval,
the top contributing sensors (input-gradient saliency), and an automatic human-review flag when
the interval is in the widest quartile (band > {THRESH:.1f} cycles).

**Headline test result.** RMSE {meta['test_rmse']:.2f} cycles · C-MAPSS asymmetric score
{meta['test_score']:,} · {meta['n_flagged']}/100 engines routed to a human.

**Data.** NASA C-MAPSS, *Turbofan Engine Degradation Simulation Data Set*, subset FD001 —
one operating condition, one fault mode (HPC degradation).
Source: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

**Not** an operational tool. Intervals are approximate (uncalibrated MC-Dropout); the model is
trained and validated on a single operating condition only.
"""
    )
    st.caption(f"TensorFlow available in this deployment: {HAS_TF}")
