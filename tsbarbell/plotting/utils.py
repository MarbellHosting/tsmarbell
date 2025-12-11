import plotly.graph_objects as go


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
