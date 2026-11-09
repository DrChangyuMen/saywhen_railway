"""
This application LOOKS UP precomputed results. It does not train a model.
"""
from __future__ import annotations

import math
from pathlib import Path

from predictor import CATEGORY_LABELS, load_data, lookup_prediction

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
import streamlit as st


st.set_page_config(
    page_title="The Prototyped Decision Support System",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# The radio-choice order corresponds to the seven ordinal categories provided.
FEATURES = CATEGORY_LABELS
DATA_FOLDER = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def read_data(first_version: int, second_version: int):
    """File modification times in the cache key make updated CSV data reload."""
    return load_data(
        DATA_FOLDER / "webapp.csv",
        DATA_FOLDER / "webapp2.csv",
    )


try:
    csv1 = DATA_FOLDER / "webapp.csv"
    csv2 = DATA_FOLDER / "webapp2.csv"
    data = read_data(csv1.stat().st_mtime_ns, csv2.stat().st_mtime_ns)
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Unable to load the two CSV datasets: {error}")
    st.info("Make sure webapp.csv and webapp2.csv are next to app.py.")
    st.stop()


def make_gauge(
    value: float,
    low: float,
    high: float,
    *,
    color: str | None = None,
    needle: bool = False,
    number_suffix: str = "%",
    number_decimals: int = 0,
):
    """Draw the screenshot-style semicircular gauge with Matplotlib."""
    fig, ax = plt.subplots(figsize=(6.8, 3.3), dpi=135)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")
    line_color = "#555555"
    tick_color = "#7d8ca4"

    # Upper half of concentric circles + baseline.
    for radius in (1.0, 0.77):
        ax.add_patch(
            Arc((0, 0), 2 * radius, 2 * radius, theta1=0, theta2=180,
                lw=0.8, color=line_color)
        )
    ax.plot([-1, -0.77], [0, 0], color=line_color, lw=0.8)
    ax.plot([0.77, 1], [0, 0], color=line_color, lw=0.8)

    if color is not None:
        fraction = max(0.0, min(1.0, (value - low) / (high - low)))
        if fraction > 0:
            ax.add_patch(
                Arc(
                    (0, 0),
                    1.76,
                    1.76,
                    theta1=180 * (1.0 - fraction),
                    theta2=180,
                    lw=10,
                    color=color,
                    capstyle="butt",
                )
            )

    marks = [-50, -25, 0, 25, 50] if low < 0 else [0, 20, 40, 60, 80, 100]
    for tick in marks:
        theta = math.pi * (1 - (tick - low) / (high - low))
        cx, sy = math.cos(theta), math.sin(theta)
        ax.plot([cx * 1.005, cx * 1.03], [sy * 1.005, sy * 1.03],
                color="#8a8a8a", lw=0.55)
        ax.text(
            cx * 1.075, sy * 1.075, str(tick),
            ha="center", va="center", fontsize=10, color=tick_color
        )

    if needle:
        fraction = max(0.0, min(1.0, (value - low) / (high - low)))
        theta = math.pi * (1.0 - fraction)

        # Increase arrow length from 0.65 to 0.92
        tip = (0.92 * math.cos(theta), 0.92 * math.sin(theta))

        ax.add_patch(FancyArrowPatch(
            (0.0, 0.0), tip,
            arrowstyle="-|>",
            mutation_scale=25,
            color="#2860f4",
            lw=1.7
        ))

    formatted_value = f"{value:.{number_decimals}f}{number_suffix}"
    ax.text(
        0, 0.055, formatted_value, ha="center", va="center",
        fontweight="bold",
        fontsize=12, color="#8497b6"
    )
    ax.set_xlim(-1.16, 1.16)
    ax.set_ylim(-0.09, 1.24)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.06)
    return fig


st.markdown(
    """
    <style>
      .block-container {
        padding-top: 2.5rem;
        max-width: 1800px;
      }

      h1 {
        text-align: center;
        color: var(--text-color);
        font-weight: 750;
        font-size: 24px !important;
      }

      .category-heading {
        font-weight: 700;
        border-left: 4px solid #2463ed;
        padding-left: 8px;
        margin-top: 19px;
        margin-bottom: -4px;
        font-size: 0.93rem;
        color: var(--text-color);
      }

      .feature-summary {
        border-radius: 5px;
        padding: 15px 13px;
        background: #f1f2f4;
        margin: 0 0 22px 0;
        font-size: 0.93rem;
        color: #1e293b;
      }

      .metric-heading {
        min-height: 58px;
        font-weight: 750;
        font-size: 1.07rem;
        line-height: 1.35;
        text-align: center;
        color: var(--text-color);
      }

      .tree-summary {
        text-align: center;
        font-size: 18px;
        font-weight: 700;
        color: var(--text-color);
        padding-top: 20px;
        padding-bottom: 18px;
      }

      /* Radio options arranged horizontally */
      div[data-testid="stRadio"] > div {
        gap: 0.85rem;
      }

      .stButton > button[kind="secondary"],
      .stFormSubmitButton > button {
        border: 1px solid #d7dce3;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if "view" not in st.session_state:
    st.session_state.view = "input"

if st.session_state.view == "input":
    st.title("The Prototyped Decision Support System")
    st.write("")
    with st.form("input_form"):
        for key, header, choices in FEATURES:
            st.markdown(
                f'<div class="category-heading">{header}</div>',
                unsafe_allow_html=True,
            )
            st.radio(
                header,
                options=range(len(choices)),
                format_func=lambda i, options=choices: options[i],
                horizontal=True,
                key=key,
                label_visibility="collapsed",
            )

        st.write("")
        submitted = st.form_submit_button("START", type="secondary")

    if submitted:
        st.session_state.view = "results"
        st.rerun()

else:
    indices = {key: st.session_state.get(key, 0) for key, _, _ in FEATURES}
    description = "; ".join(choices[indices[key]] for key, _, choices in FEATURES)
    try:
        prediction, encoded_features, source_row = lookup_prediction(data, indices)
    except (LookupError, ValueError) as error:
        st.error(f"Cannot retrieve the selected combination: {error}")
        st.stop()

    st.markdown(
        f'<div class="feature-summary"><b>Current feature combination:</b> '
        f'{description}</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="medium")
    metrics = [
        (
            col1,
            "Directional stability：Share of trees predicting override harms train punctuality",
            prediction.harm_probability * 100,
            0,
            100,
            "#ec2727",
            False,
            "%",
            2,
        ),
        (
            col2,
            "Predicted change in train delay if overridden",
            prediction.delay_seconds,
            -50,
            50,
            None,
            True,
            " seconds per train section",
            2,
        ),
        (
            col3,
            "Directional stability：Share of trees predicting override improves train punctuality",
            prediction.improvement_probability * 100,
            0,
            100,
            "#06a274",
            False,
            "%",
            2,
        ),
    ]
    for col, heading, value, lo, hi, color, needle, suffix, decimals in metrics:
        with col:
            st.markdown(
                f'<div class="metric-heading">{heading}</div>',
                unsafe_allow_html=True,
            )
            fig = make_gauge(
                value, lo, hi, color=color, needle=needle,
                number_suffix=suffix, number_decimals=decimals
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown(
        '<div class="tree-summary">'
        'Percentage of retained decision trees used to generate '
        'decision-support recommendations: '
        f'{prediction.retained_trees_ratio * 100:.2f}%</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button("← Back to inputs", type="secondary"):
        st.session_state.view = "input"
        st.rerun()
