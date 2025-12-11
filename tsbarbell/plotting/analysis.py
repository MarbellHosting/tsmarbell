import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any


def plot_statistical_analysis(
    df: pd.DataFrame,
    portfolio: Any,
    test_results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> go.Figure:
    """
    Create a comprehensive dashboard for statistical analysis results.

    Args:
        df: DataFrame containing price and indicator data.
        portfolio: vectorbt Portfolio object.
        test_results: List of dictionaries containing test results.
        metrics: Dictionary of performance metrics.

    Returns:
        Plotly Figure object.
    """

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Portfolio Equity Curve",
            "Statistical Test Results",
            "Monte Carlo Distribution",
            "Bootstrap Distribution",
            "Probability Signals",
            "Walk-Forward Returns",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "table"}],
            [{"type": "histogram"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # 1. Equity Curve
    equity = portfolio.value()
    buy_hold = df["close"] / df["close"].iloc[0] * equity.iloc[0]

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name="Strategy",
            line=dict(color="#00ff00", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=buy_hold.values,
            name="Buy & Hold",
            line=dict(color="#ffd700", width=1),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # 2. Test Results Table
    test_names = [t["name"] for t in test_results]
    test_passed = ["✅ PASS" if t["passed"] else "❌ FAIL" for t in test_results]
    test_pvalues = [
        f"p={t.get('p_value', 'N/A'):.4f}" if t.get("p_value") is not None else "N/A"
        for t in test_results
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Test", "Result", "p-value"],
                fill_color="#1a1a2e",
                font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=[test_names, test_passed, test_pvalues],
                fill_color=[["#16213e"] * len(test_names)],
                font=dict(color="white"),
                align="left",
            ),
        ),
        row=1,
        col=2,
    )

    # 3. Monte Carlo Distribution
    mc_result = next((t for t in test_results if "Monte Carlo" in t["name"]), None)
    if mc_result and "random_returns" in mc_result:
        actual_return = metrics["total_return"]
        fig.add_trace(
            go.Histogram(
                x=mc_result["random_returns"],
                name="Random Returns",
                marker_color="#4a4a6a",
                nbinsx=50,
            ),
            row=2,
            col=1,
        )
        # Add vertical line as a shape
        fig.add_shape(
            type="line",
            x0=actual_return,
            x1=actual_return,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="lime", width=2, dash="dash"),
            row=2,
            col=1,
        )

    # 4. Bootstrap Distribution
    bs_result = next((t for t in test_results if "Bootstrap" in t["name"]), None)
    if bs_result and "bootstrap_returns" in bs_result:
        fig.add_trace(
            go.Histogram(
                x=bs_result["bootstrap_returns"],
                name="Bootstrap Returns",
                marker_color="#6a4a6a",
                nbinsx=50,
            ),
            row=2,
            col=2,
        )
        if bs_result["ci_lower"] is not None:
            # Add CI lines as shapes
            for ci_val in [bs_result["ci_lower"], bs_result["ci_upper"]]:
                fig.add_shape(
                    type="line",
                    x0=ci_val,
                    x1=ci_val,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash"),
                    row=2,
                    col=2,
                )

    # 5. Probability Signals
    for h, color in [(4, "aqua"), (12, "#00e676"), (24, "orange"), (60, "violet")]:
        col_name = f"btc_slope_{h}h"
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    name=f"P({h}h)",
                    line=dict(color=color, width=1),
                ),
                row=3,
                col=1,
            )

    # 6. Walk-Forward Returns
    wf_result = next((t for t in test_results if "Walk-Forward" in t["name"]), None)
    if wf_result and "fold_returns" in wf_result:
        folds = list(range(1, len(wf_result["fold_returns"]) + 1))
        colors = ["green" if r > 0 else "red" for r in wf_result["fold_returns"]]
        fig.add_trace(
            go.Bar(
                x=[f"Fold {i}" for i in folds],
                y=wf_result["fold_returns"],
                marker_color=colors,
                name="Fold Return",
            ),
            row=3,
            col=2,
        )

    fig.update_layout(
        title="📊 BTC Slope Probability — Statistical Significance Analysis",
        template="plotly_dark",
        height=1000,
        showlegend=True,
    )

    return fig
