import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional
from .utils import create_base_figure, add_probability_trace, add_threshold_lines


def plot_volatility_probability(
    data: Dict[int, pd.DataFrame],
    title: str = "BTC Volatility Forecast",
    theme: str = "plotly_dark",
) -> go.Figure:
    """
    Plot BTC Volatility Probability for multiple horizons.

    Args:
        data: Dictionary mapping horizon (int) to DataFrame.
        title: Plot title.
        theme: Plotly template.

    Returns:
        go.Figure: Plotly figure.
    """
    fig = create_base_figure(title, theme)

    # Define colors for different horizons
    colors = {
        24: "#ff6b6b",  # Red
        48: "#feca57",  # Yellow
        96: "#48dbfb",  # Blue
    }

    for horizon, df in data.items():
        if df.empty:
            continue

        color = colors.get(horizon, "#ffffff")
        add_probability_trace(fig, df, f"P(Vol > Threshold) {horizon}h", color)

    # Add threshold lines (customizable per horizon in a real app, but fixed here for simplicity)
    # Using average thresholds from the JS code provided
    add_threshold_lines(fig, high=0.65, low=0.35)

    return fig
