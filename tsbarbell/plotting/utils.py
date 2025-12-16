import plotly.graph_objects as go
import pandas as pd


def apply_common_layout(fig: go.Figure, title: str):
    """
    Apply common layout settings to a Plotly figure.
    """
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )


def create_base_figure(title: str, template: str = "plotly_dark") -> go.Figure:
    """
    Create a base Plotly figure with common layout.
    """
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Probability",
        template=template,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        yaxis=dict(range=[-0.05, 1.05]),  # Probability range
    )
    return fig


def add_probability_trace(
    fig: go.Figure, df: pd.DataFrame, name: str, color: str
) -> None:
    """
    Add a probability trace to the figure.
    """
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["probability"],
            mode="lines",
            name=name,
            line=dict(color=color, width=1.5),
        )
    )


def add_threshold_lines(fig: go.Figure, high: float = 0.99, low: float = 0.01) -> None:
    """
    Add high and low threshold lines to the figure.
    """
    fig.add_hline(
        y=high,
        line_dash="dash",
        line_color="green",
        annotation_text=f"High ({high})",
        annotation_position="top right",
    )
    fig.add_hline(
        y=low,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Low ({low})",
        annotation_position="bottom right",
    )
