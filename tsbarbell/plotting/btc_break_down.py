import plotly.graph_objects as go
import pandas as pd
from typing import Dict
from .utils import apply_common_layout


def plot_break_down_probability(
    data: Dict[int, pd.DataFrame], title: str = "BTC Break-Down Probability"
):
    """
    Plot BTC Break-Down Probability for multiple horizons.

    Args:
        data (Dict[int, pd.DataFrame]): Dictionary mapping horizon (int) to DataFrame.
        title (str): Title of the plot.
    """
    fig = go.Figure()

    colors = {
        1: "#2196f3",  # Blue-ish
        2: "#9c27b0",  # Purple
        4: "orange",
        8: "violet",
    }

    # Sort horizons to ensure consistent legend order
    sorted_horizons = sorted(data.keys())

    for horizon in sorted_horizons:
        df = data[horizon]
        if df.empty:
            continue

        color = colors.get(horizon, "grey")

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["probability"],
                mode="lines",
                name=f"{horizon}h",
                line=dict(color=color, width=1.5),
            )
        )

    # Add threshold lines (using 0.60 for SHORT and 0.10 for EXIT as defaults from the snippet)
    fig.add_hline(
        y=0.60,
        line_dash="dash",
        line_color="red",
        annotation_text="Short (0.60)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=0.10,
        line_dash="dash",
        line_color="green",
        annotation_text="Exit (0.10)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        yaxis_title="Probability",
        yaxis=dict(range=[-0.05, 1.05]),
    )

    apply_common_layout(fig, title)

    return fig
