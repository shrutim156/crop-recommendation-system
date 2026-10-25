"""
app.py
------
Streamlit front-end for the Crop Recommendation System.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from src.data_preprocessing import FEATURE_RANGES
from src.evaluate_model import evaluate, get_feature_importance, load_comparison_metrics
from src.prediction import (
    FEATURE_LABELS,
    explain_feature_contributions,
    load_model_artifact,
    predict_crop,
)
from utils.crop_info import CROP_INFO, get_crop_info, list_crops

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    color: #2B2A25;
}

.crs-hero {
    background: linear-gradient(135deg, #3F6C4F 0%, #2E5240 100%);
    padding: 2rem 2.25rem;
    border-radius: 14px;
    color: #FBF8F1;
    margin-bottom: 1.5rem;
}
.crs-hero h1 {
    color: #FBF8F1;
    margin: 0 0 0.35rem 0;
    font-size: 2.1rem;
}
.crs-hero p {
    color: #E4E9DE;
    margin: 0;
    font-size: 1.02rem;
}

.crs-result-card {
    background: #F0EBDD;
    border-left: 6px solid #C98A2D;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.crs-result-card h2 {
    margin: 0 0 0.2rem 0;
    text-transform: capitalize;
    color: #2E5240;
}
.crs-confidence {
    font-size: 1.05rem;
    color: #6B4A2F;
    font-weight: 600;
}

.crs-badge {
    display: inline-block;
    background: #3F6C4F;
    color: #FBF8F1;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
}

.crs-disclaimer {
    background: #FCEFD8;
    border: 1px solid #E4C388;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #6B4A2F;
}

.crs-warning {
    background: #FBE6E1;
    border: 1px solid #E9AFA0;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.88rem;
    color: #8A3B26;
    margin-bottom: 0.5rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def get_artifact():
    return load_model_artifact()


try:
    artifact = get_artifact()
    MODEL_LOAD_ERROR = None
except FileNotFoundError as e:
    artifact = None
    MODEL_LOAD_ERROR = str(e)


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="crs-hero">
        <h1>🌾 Crop Recommendation System</h1>
        <p>AI-powered guidance on the most suitable crop for your soil and climate conditions —
        with confidence scores, plain-language reasoning, and what-if analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if MODEL_LOAD_ERROR:
    st.error(
        f"{MODEL_LOAD_ERROR}\n\nRun `python -m src.train_model` from the project "
        "root to train and save the model before launching the app."
    )
    st.stop()

model_name = artifact["model_name"]
feature_columns = artifact["feature_columns"]


# ---------------------------------------------------------------------------
# Sidebar: input form
# ---------------------------------------------------------------------------

st.sidebar.header("🧪 Soil & Climate Inputs")
st.sidebar.caption("Adjust the sliders to match your field's conditions.")

DEFAULTS = {"N": 90, "P": 42, "K": 43, "temperature": 20.9, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}

if "inputs" not in st.session_state:
    st.session_state.inputs = DEFAULTS.copy()


def sidebar_slider(key, label, min_val, max_val, step, help_text=None):
    return st.sidebar.slider(
        label,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(st.session_state.inputs[key]),
        step=float(step),
        help=help_text,
        key=f"slider_{key}",
    )


n_val = sidebar_slider("N", "Nitrogen - N (kg/ha)", 0, 140, 1, "Typical realistic range: 0-140 kg/ha")
p_val = sidebar_slider("P", "Phosphorus - P (kg/ha)", 5, 145, 1, "Typical realistic range: 5-145 kg/ha")
k_val = sidebar_slider("K", "Potassium - K (kg/ha)", 5, 205, 1, "Typical realistic range: 5-205 kg/ha")
temp_val = sidebar_slider("temperature", "Temperature (°C)", 8, 45, 0.1)
humidity_val = sidebar_slider("humidity", "Humidity (%)", 14, 100, 0.1)
ph_val = sidebar_slider("ph", "Soil pH", 3.5, 9.5, 0.1)
rainfall_val = sidebar_slider("rainfall", "Rainfall (mm)", 20, 300, 0.1)

current_inputs = {
    "N": n_val,
    "P": p_val,
    "K": k_val,
    "temperature": temp_val,
    "humidity": humidity_val,
    "ph": ph_val,
    "rainfall": rainfall_val,
}
st.session_state.inputs = current_inputs

if st.sidebar.button("↺ Reset to defaults"):
    st.session_state.inputs = DEFAULTS.copy()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Model in use: **{model_name}**")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_predict, tab_whatif, tab_compare, tab_info, tab_dashboard = st.tabs(
    ["🔮 Prediction", "🧭 What-If Analysis", "⚖️ Crop Comparison", "📖 Crop Information", "📊 Model Dashboard"]
)


# --- Tab 1: Prediction ------------------------------------------------------

with tab_predict:
    result = predict_crop(current_inputs, artifact, top_k=3)

    if result["warnings"]:
        for w in result["warnings"]:
            st.markdown(f'<div class="crs-warning">⚠️ {w}</div>', unsafe_allow_html=True)

    col_main, col_side = st.columns([1.4, 1])

    with col_main:
        st.markdown(
            f"""
            <div class="crs-result-card">
                <span class="crs-badge">Top Recommendation</span>
                <h2>{result['top_crop']}</h2>
                <div class="crs-confidence">Confidence: {result['confidence']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(result["explanation"])

        st.subheader("Top 3 Recommended Crops")
        top3_df = pd.DataFrame(result["top_k"], columns=["Crop", "Probability"])
        top3_df["Probability (%)"] = (top3_df["Probability"] * 100).round(2)
        fig = px.bar(
            top3_df,
            x="Probability (%)",
            y="Crop",
            orientation="h",
            color="Probability (%)",
            color_continuous_scale=["#F0EBDD", "#3F6C4F"],
            text="Probability (%)",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
            coloraxis_showscale=False,
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.subheader("Why this crop?")
        st.caption("Estimated contribution of each input toward the recommendation.")
        try:
            contrib_df = explain_feature_contributions(current_inputs, artifact, top_n=7)
            colors = ["#3F6C4F" if v >= 0 else "#C0563E" for v in contrib_df["Contribution"]]
            fig2 = go.Figure(
                go.Bar(
                    x=contrib_df["Contribution"],
                    y=contrib_df["Feature"],
                    orientation="h",
                    marker_color=colors,
                )
            )
            fig2.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Contribution",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Green bars push toward the recommendation; red bars push away from it.")
        except Exception as e:
            st.info("Feature contribution explanation is unavailable for this model.")

    st.markdown(
        """
        <div class="crs-disclaimer">
        ⚠️ <b>Disclaimer:</b> This recommendation is generated by a machine learning model trained on
        historical soil/climate data. It should be treated as a starting point, not a guarantee.
        Always validate against local agricultural extension services, soil testing, and real field
        conditions before making planting decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Tab 2: What-If Analysis -------------------------------------------------

with tab_whatif:
    st.subheader("Explore how conditions change the recommendation")
    st.caption(
        "Pick one input to vary while holding the others at their current sidebar values, "
        "and see how the model's recommendation shifts."
    )

    whatif_feature = st.selectbox(
        "Feature to vary",
        options=feature_columns,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
    )
    low, high = FEATURE_RANGES[whatif_feature]
    n_steps = st.slider("Number of steps", min_value=5, max_value=30, value=15)

    sweep_values = np.linspace(low, high, n_steps)
    rows = []
    for v in sweep_values:
        trial_inputs = current_inputs.copy()
        trial_inputs[whatif_feature] = float(v)
        trial_result = predict_crop(trial_inputs, artifact, top_k=1)
        rows.append(
            {
                FEATURE_LABELS.get(whatif_feature, whatif_feature): v,
                "Predicted Crop": trial_result["top_crop"],
                "Confidence (%)": round(trial_result["confidence"] * 100, 1),
            }
        )
    sweep_df = pd.DataFrame(rows)

    fig3 = px.scatter(
        sweep_df,
        x=FEATURE_LABELS.get(whatif_feature, whatif_feature),
        y="Confidence (%)",
        color="Predicted Crop",
        symbol="Predicted Crop",
        size=[14] * len(sweep_df),
    )
    fig3.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("View sweep data"):
        st.dataframe(sweep_df, use_container_width=True, hide_index=True)

    st.info(
        f"Currently holding all other inputs at their sidebar values while sweeping "
        f"**{FEATURE_LABELS.get(whatif_feature, whatif_feature)}** from {low} to {high}. "
        "Note where the predicted crop changes — that boundary is where your soil/climate "
        "conditions sit at the edge between two suitable crops."
    )


# --- Tab 3: Crop Comparison --------------------------------------------------

with tab_compare:
    st.subheader("Compare the Top 3 Recommended Crops")

    result = predict_crop(current_inputs, artifact, top_k=3)
    top3_names = [c for c, _ in result["top_k"]]

    compare_rows = []
    for crop in top3_names:
        info = get_crop_info(crop)
        conditions = info.get("ideal_conditions", {})
        compare_rows.append(
            {
                "Crop": crop.title(),
                "Category": info.get("category", "-"),
                **{FEATURE_LABELS.get(k, k): v for k, v in conditions.items()},
            }
        )
    compare_df = pd.DataFrame(compare_rows).set_index("Crop")
    st.dataframe(compare_df, use_container_width=True)

    st.markdown("#### Descriptions")
    for crop in top3_names:
        info = get_crop_info(crop)
        st.markdown(f"**{crop.title()}** — {info.get('description', 'No description available.')}")

    st.caption(
        "Ideal condition ranges are approximate, hand-curated agronomic references — always confirm "
        "with local soil testing and extension guidance before planting."
    )


# --- Tab 4: Crop Information --------------------------------------------------

with tab_info:
    st.subheader("Crop Reference Library")
    search_col, filter_col = st.columns([2, 1])
    with search_col:
        query = st.text_input("Search crops", placeholder="e.g. rice, banana, cotton")
    with filter_col:
        categories = sorted({v["category"] for v in CROP_INFO.values()})
        selected_cat = st.selectbox("Filter by category", options=["All"] + categories)

    crops_to_show = list_crops()
    if query:
        crops_to_show = [c for c in crops_to_show if query.lower() in c.lower()]
    if selected_cat != "All":
        crops_to_show = [c for c in crops_to_show if get_crop_info(c)["category"] == selected_cat]

    if not crops_to_show:
        st.warning("No crops match your search/filter.")

    cols = st.columns(3)
    for i, crop in enumerate(crops_to_show):
        info = get_crop_info(crop)
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{crop.title()}**  \n`{info['category']}`")
                st.caption(info["description"])
                conditions = info.get("ideal_conditions", {})
                cond_lines = "  \n".join(
                    f"- {FEATURE_LABELS.get(k, k)}: {v}" for k, v in conditions.items()
                )
                st.markdown(cond_lines)


# --- Tab 5: Model Performance Dashboard --------------------------------------

with tab_dashboard:
    st.subheader("Model Performance Dashboard")

    comparison_df = load_comparison_metrics()
    if not comparison_df.empty:
        st.markdown("#### Model Comparison")
        st.dataframe(
            comparison_df.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-Score"], color="#DDEBD9"
            ),
            use_container_width=True,
        )

        fig4 = px.bar(
            comparison_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score"]),
            x="Model",
            y="value",
            color="variable",
            barmode="group",
            labels={"value": "Score", "variable": "Metric"},
            color_discrete_sequence=["#3F6C4F", "#C98A2D", "#6B4A2F", "#8FAE8B"],
        )
        fig4.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_range=[0.9, 1.0])
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No comparison metrics found. Run `python -m src.train_model` to generate them.")

    st.markdown(f"#### Deployed Model: {model_name}")

    try:
        report, cm_df, class_names = evaluate(artifact)

        col_a, col_b = st.columns([1, 1.4])
        with col_a:
            st.markdown("**Classification Report**")
            st.text(report)

        with col_b:
            st.markdown("**Confusion Matrix**")
            fig5 = px.imshow(
                cm_df,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                color_continuous_scale=["#FBF8F1", "#3F6C4F"],
                aspect="auto",
            )
            fig5.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig5, use_container_width=True)

        st.markdown("**Feature Importance**")
        fi_df = get_feature_importance(artifact)
        if not fi_df.empty:
            fi_df["Feature"] = fi_df["Feature"].map(lambda c: FEATURE_LABELS.get(c, c))
            fig6 = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=["#3F6C4F"],
            )
            fig6.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("This model type does not expose feature importances directly.")

    except FileNotFoundError as e:
        st.error(str(e))

    st.markdown(
        """
        <div class="crs-disclaimer">
        Model confidence and accuracy describe statistical performance on historical data —
        they are <b>not a guarantee of real-world crop success</b>, which also depends on
        seed quality, pest/disease pressure, irrigation, and farming practices.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("Crop Recommendation System · Built with Streamlit & scikit-learn · For educational/portfolio use.")
