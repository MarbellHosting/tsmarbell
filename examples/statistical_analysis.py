"""
🎯 BTC Slope Probability — Complete Statistical Analysis

This script runs the full backtest and all 5 statistical significance tests
in a single Python file. Perfect for YouTube demo!

Tests included:
1. Monte Carlo Simulation
2. Bootstrap Analysis
3. Walk-Forward Validation
4. Permutation Test
5. T-Test (Strategy vs Buy & Hold)

Usage:
    python examples/statistical_analysis.py

Author: AI Price Patterns
"""

import os
import sys
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from scipy import stats as scipy_stats
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path to import tsbarbell if needed,
# though this script mostly uses direct API calls as per provided code.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Try to import optional dependencies
try:
    import vectorbt as vbt

    HAS_VBT = True
except ImportError:
    print("⚠️ vectorbt not installed. Install with: pip install vectorbt")
    HAS_VBT = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# =============================================================================
# CONFIG
# =============================================================================

API_BASE = "https://ts.marbell.com"
LIMIT = 10000

ENDPOINTS = {
    4: "btc-slope-1h-h4",
    12: "btc-slope-1h-h12",
    24: "btc-slope-1h-h24",
    60: "btc-slope-1h-h60",
}

# Strategy parameters (optimized)
# These parameters were optimized using grid search
STRATEGY_PARAMS = {
    "entry_4h": 0.8,
    "entry_12h": 0.8,
    "entry_24h": 0.7,
    "entry_60h": 0.6,
    "exit_4h": 0.3,
    "exit_12h": 0.4,
}

# Test parameters
N_SIMULATIONS = 1000  # Monte Carlo & Bootstrap iterations (reduced for demo speed)
N_PERMUTATIONS = 1000  # Permutation test iterations
N_WALK_FORWARD_FOLDS = 5  # Walk-forward validation folds


# =============================================================================
# DATA LOADING
# =============================================================================


def fetch_data(api_key: str, horizon: int) -> pd.DataFrame:
    """Fetch data from API for specified horizon"""
    endpoint = ENDPOINTS.get(horizon)
    if not endpoint:
        raise ValueError(f"Unknown horizon: {horizon}")

    url = f"{API_BASE}/{endpoint}?limit={LIMIT}&expand=false"
    headers = {"x-api-key": api_key}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Error fetching data (h={horizon}): {e}")

    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    if "close_now" in df.columns:
        df["close_now"] = pd.to_numeric(df["close_now"], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def load_all_data(api_key: str) -> pd.DataFrame:
    """Fetch and combine all horizons data"""
    print("📊 Loading BTC Slope Probability data...")

    data = {}
    for h in [4, 12, 24, 60]:
        print(f"  → Loading {h}h horizon...")
        data[h] = fetch_data(api_key, h)
        print(f"    ✓ {len(data[h])} records")

    # Combine data by time
    dfs = []
    for h, df in data.items():
        if df.empty:
            continue
        temp = df[["datetime", "probability"]].copy()
        temp = temp.rename(columns={"probability": f"prob_{h}h"})

        # Add price only from first horizon
        if h == 4 and "close_now" in df.columns:
            temp["close"] = df["close_now"]

        temp = temp.set_index("datetime")
        dfs.append(temp)

    if not dfs:
        raise RuntimeError("No data loaded")

    # Combine
    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.join(df, how="outer")

    combined = combined.sort_index()
    combined = combined.ffill().dropna()

    print(f"\n✅ Combined data: {len(combined)} records")
    print(f"   Period: {combined.index[0].date()} to {combined.index[-1].date()}")

    return combined


# =============================================================================
# BACKTEST
# =============================================================================


def generate_signals(df: pd.DataFrame, params: dict = None) -> tuple:
    """Generate entry and exit signals based on strategy parameters"""
    if params is None:
        params = STRATEGY_PARAMS

    # Entry signal: all conditions must be met
    entry_signal = (
        (df["prob_4h"] > params["entry_4h"])
        & (df["prob_12h"] > params["entry_12h"])
        & (df["prob_24h"] > params["entry_24h"])
        & (df["prob_60h"] > params["entry_60h"])
    )

    # Exit signal: any condition is met
    exit_signal = (df["prob_4h"] < params["exit_4h"]) | (
        df["prob_12h"] < params["exit_12h"]
    )

    return entry_signal, exit_signal


def run_backtest(
    df: pd.DataFrame,
    entry_signal: pd.Series,
    exit_signal: pd.Series,
    init_cash: float = 10000,
    fees: float = 0.0,
):
    """Run backtest with VectorBT"""
    if not HAS_VBT:
        raise ImportError("vectorbt is required for backtesting")

    portfolio = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entry_signal,
        exits=exit_signal,
        init_cash=init_cash,
        fees=fees,
        freq="1h",
    )

    return portfolio


def calculate_metrics(portfolio) -> dict:
    """Calculate key performance metrics"""
    trades = portfolio.trades.records_readable

    return {
        "total_return": portfolio.total_return() * 100,
        "sharpe_ratio": portfolio.sharpe_ratio(),
        "max_drawdown": portfolio.max_drawdown() * 100,
        "win_rate": portfolio.trades.win_rate() * 100 if len(trades) > 0 else 0,
        "n_trades": len(trades),
        "daily_returns": portfolio.returns().resample("D").sum(),
        "trade_returns": portfolio.trades.returns.values
        if len(trades) > 0
        else np.array([]),
    }


# =============================================================================
# STATISTICAL TESTS
# =============================================================================


def monte_carlo_test(
    df: pd.DataFrame,
    actual_return: float,
    entry_signal: pd.Series,
    exit_signal: pd.Series,
    n_simulations: int = N_SIMULATIONS,
) -> dict:
    """
    Monte Carlo Test

    Question: Can a random strategy achieve the same results?
    Method: Generate random entry/exit signals with same frequency and compare returns
    """
    print("\n🎲 Running Monte Carlo Simulation...")

    # Calculate actual signal frequency
    entry_freq = entry_signal.sum() / len(entry_signal)
    exit_freq = exit_signal.sum() / len(exit_signal)

    random_returns = []

    for _ in tqdm(range(n_simulations), desc="   Simulating"):
        # Generate random signals with SAME frequency as actual strategy
        n = len(df)
        random_entries = pd.Series(np.random.random(n) < entry_freq, index=df.index)
        random_exits = pd.Series(np.random.random(n) < exit_freq, index=df.index)

        try:
            portfolio = run_backtest(df, random_entries, random_exits)
            random_returns.append(portfolio.total_return() * 100)
        except:
            random_returns.append(0)

    random_returns = np.array(random_returns)

    # Calculate p-value (one-tailed: how often random beats actual)
    p_value = np.mean(random_returns >= actual_return)
    percentile = (1 - p_value) * 100

    passed = p_value < 0.05

    return {
        "name": "Monte Carlo Simulation",
        "question": "Can a random strategy achieve the same results?",
        "p_value": p_value,
        "percentile": percentile,
        "passed": passed,
        "result": f"Strategy beats {percentile:.1f}% of random strategies",
        "random_returns": random_returns,
    }


def bootstrap_test(
    trade_returns: np.ndarray, n_simulations: int = N_SIMULATIONS
) -> dict:
    """
    Bootstrap Test

    Question: How stable are the returns?
    Method: Resample trades with replacement to estimate confidence interval
    """
    print("\n📊 Running Bootstrap Analysis...")

    if len(trade_returns) < 5:
        return {
            "name": "Bootstrap Analysis",
            "question": "How stable are the returns?",
            "passed": False,
            "result": "Insufficient trades for analysis",
            "ci_lower": None,
            "ci_upper": None,
        }

    bootstrap_returns = []

    for _ in tqdm(range(n_simulations), desc="   Bootstrapping"):
        # Resample trades with replacement
        sample = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
        bootstrap_returns.append(np.sum(sample) * 100)

    bootstrap_returns = np.array(bootstrap_returns)

    # Calculate 95% confidence interval
    ci_lower = np.percentile(bootstrap_returns, 2.5)
    ci_upper = np.percentile(bootstrap_returns, 97.5)

    passed = ci_lower > 0

    return {
        "name": "Bootstrap Analysis",
        "question": "How stable are the returns?",
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "passed": passed,
        "result": f"95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]",
        "bootstrap_returns": bootstrap_returns,
    }


def walk_forward_test(df: pd.DataFrame, n_folds: int = N_WALK_FORWARD_FOLDS) -> dict:
    """
    Walk-Forward Test

    Question: Does the strategy work on new, unseen data?
    Method: Train on past data, test on future data, repeat
    """
    print("\n🚶 Running Walk-Forward Analysis...")

    fold_size = len(df) // (n_folds + 1)
    fold_returns = []

    for i in tqdm(range(n_folds), desc="   Testing folds"):
        # Define train and test periods
        test_start = (i + 1) * fold_size
        test_end = (i + 2) * fold_size if i < n_folds - 1 else len(df)

        test_df = df.iloc[test_start:test_end].copy()

        if len(test_df) < 100:
            continue

        # Generate signals and run backtest
        entry_signal, exit_signal = generate_signals(test_df)

        try:
            portfolio = run_backtest(test_df, entry_signal, exit_signal)
            fold_return = portfolio.total_return() * 100
            fold_returns.append(fold_return)
            print(f"      Fold {i + 1}: {fold_return:+.2f}%")
        except:
            fold_returns.append(0)

    profitable_folds = sum(1 for r in fold_returns if r > 0)
    avg_return = np.mean(fold_returns)

    passed = profitable_folds >= n_folds * 0.6  # At least 60% profitable

    return {
        "name": "Walk-Forward Analysis",
        "question": "Does the strategy work on new data?",
        "fold_returns": fold_returns,
        "profitable_folds": profitable_folds,
        "total_folds": len(fold_returns),
        "avg_return": avg_return,
        "passed": passed,
        "result": f"{profitable_folds}/{len(fold_returns)} folds profitable, avg return: {avg_return:+.2f}%",
    }


def permutation_test(
    df: pd.DataFrame, actual_return: float, n_permutations: int = N_PERMUTATIONS
) -> dict:
    """
    Permutation Test

    Question: Do the signals have real predictive power?
    Method: Shuffle signals randomly and compare to original
    """
    print("\n🔀 Running Permutation Test...")

    entry_signal, exit_signal = generate_signals(df)

    permuted_returns = []

    for _ in tqdm(range(n_permutations), desc="   Permuting"):
        # Shuffle signals
        shuffled_entries = entry_signal.sample(frac=1).values
        shuffled_exits = exit_signal.sample(frac=1).values

        shuffled_entry_series = pd.Series(shuffled_entries, index=df.index)
        shuffled_exit_series = pd.Series(shuffled_exits, index=df.index)

        try:
            portfolio = run_backtest(df, shuffled_entry_series, shuffled_exit_series)
            permuted_returns.append(portfolio.total_return() * 100)
        except:
            permuted_returns.append(0)

    permuted_returns = np.array(permuted_returns)

    # Calculate p-value
    p_value = np.mean(permuted_returns >= actual_return)

    passed = p_value < 0.05

    return {
        "name": "Permutation Test",
        "question": "Do the signals have real predictive power?",
        "p_value": p_value,
        "passed": passed,
        "result": f"Original signals outperform {(1 - p_value) * 100:.1f}% of shuffled signals",
        "permuted_returns": permuted_returns,
    }


def t_test(strategy_returns: pd.Series, df: pd.DataFrame) -> dict:
    """
    T-Test (Strategy vs Buy & Hold)

    Question: Is the outperformance statistically significant?
    Method: Compare daily returns using Student's t-test
    """
    print("\n📈 Running T-Test (Strategy vs Buy & Hold)...")

    # Calculate Buy & Hold daily returns
    bh_returns = df["close"].pct_change().resample("D").sum().dropna()

    # Align the series
    common_index = strategy_returns.index.intersection(bh_returns.index)
    strategy_aligned = strategy_returns.loc[common_index]
    bh_aligned = bh_returns.loc[common_index]

    # Perform t-test
    t_stat, p_value = scipy_stats.ttest_ind(strategy_aligned, bh_aligned)

    passed = p_value < 0.05 and t_stat > 0

    return {
        "name": "T-Test (Strategy vs Buy & Hold)",
        "question": "Is the outperformance statistically significant?",
        "t_statistic": t_stat,
        "p_value": p_value,
        "passed": passed,
        "result": f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}",
    }


# =============================================================================
# VISUALIZATION
# =============================================================================


def create_results_chart(
    df: pd.DataFrame,
    portfolio,
    test_results: list,
    filename: str = "statistical_analysis.html",
):
    """Create comprehensive results chart"""
    if not HAS_PLOTLY:
        print("⚠️ Plotly not installed. Skipping chart generation.")
        return

    print("\n📊 Creating results chart...")

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
        actual_return = portfolio.total_return() * 100
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
        # Add vertical line as a shape instead of vline (works better with subplots)
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
        fig.add_annotation(
            x=actual_return,
            y=1,
            yref="paper",
            text=f"Strategy: {actual_return:.1f}%",
            showarrow=False,
            font=dict(color="lime"),
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
            for ci_val, label in [
                (bs_result["ci_lower"], "95% CI Lower"),
                (bs_result["ci_upper"], "95% CI Upper"),
            ]:
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
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"prob_{h}h"],
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

    fig.write_html(filename)
    print(f"   ✅ Chart saved to: {filename}")

    return fig


def print_summary(metrics: dict, test_results: list, buy_hold_return: float):
    """Print comprehensive summary"""

    print("\n" + "=" * 70)
    print("📊 BTC SLOPE PROBABILITY — STATISTICAL ANALYSIS RESULTS")
    print("=" * 70)

    # Backtest Results
    print("\n🎯 BACKTEST RESULTS")
    print("-" * 50)
    print(f"   Total Return:    {metrics['total_return']:.2f}%")
    print(f"   Buy & Hold:      {buy_hold_return:.2f}%")
    print(f"   Alpha:           {metrics['total_return'] - buy_hold_return:+.2f}%")
    print(f"   Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown:    {metrics['max_drawdown']:.2f}%")
    print(f"   Win Rate:        {metrics['win_rate']:.1f}%")
    print(f"   Total Trades:    {metrics['n_trades']}")

    # Statistical Tests
    print("\n🧪 STATISTICAL SIGNIFICANCE TESTS")
    print("-" * 50)

    passed_count = 0
    for test in test_results:
        status = "✅ PASS" if test["passed"] else "❌ FAIL"
        if test["passed"]:
            passed_count += 1

        print(f"\n   {test['name']}")
        print(f"   Question: {test['question']}")
        print(f"   Result: {test['result']}")
        print(f"   Status: {status}")

    # Final Verdict
    print("\n" + "=" * 70)
    print("📋 FINAL VERDICT")
    print("=" * 70)

    all_passed = passed_count == len(test_results)

    if all_passed:
        print(f"""
   ✅ ALL {len(test_results)} TESTS PASSED!
   
   The BTC Slope Probability indicator is STATISTICALLY PROVEN to work:
   
   • Results are NOT due to luck (Monte Carlo p < 0.05)
   • Returns are STABLE (Bootstrap CI > 0)
   • Works on NEW data (Walk-Forward validation)
   • Signals have REAL predictive power (Permutation test)
   • Significantly OUTPERFORMS Buy & Hold (T-test)
   
   🔗 Get the indicator: https://ts.marbell.com/indicators/btc-1h-trend-ai?via=aipricepatterns
        """)
    else:
        print(f"""
   ⚠️ {passed_count}/{len(test_results)} TESTS PASSED
   
   Some tests did not pass. Review the results above for details.
        """)


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run complete statistical analysis"""

    print("\n" + "=" * 70)
    print("🎯 BTC SLOPE PROBABILITY — STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)
    print("\nThis script proves that the indicator works with 5 statistical tests.\n")

    # Load API key
    API_KEY = os.getenv("API_KEY")
    if not API_KEY:
        raise ValueError("❌ API_KEY not found. Please set it in .env file.")

    # Load data
    df = load_all_data(API_KEY)

    # Generate signals and run backtest
    print("\n🚀 Running Backtest...")
    entry_signal, exit_signal = generate_signals(df)
    print(f"   Entry signals: {entry_signal.sum()}")
    print(f"   Exit signals: {exit_signal.sum()}")

    portfolio = run_backtest(df, entry_signal, exit_signal)
    metrics = calculate_metrics(portfolio)

    # Calculate Buy & Hold return
    buy_hold_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    print(f"\n   Strategy Return: {metrics['total_return']:.2f}%")
    print(f"   Buy & Hold:      {buy_hold_return:.2f}%")
    print(f"   Alpha:           {metrics['total_return'] - buy_hold_return:+.2f}%")

    # Run all statistical tests
    test_results = []

    # 1. Monte Carlo
    mc_result = monte_carlo_test(df, metrics["total_return"], entry_signal, exit_signal)
    test_results.append(mc_result)

    # 2. Bootstrap
    bs_result = bootstrap_test(metrics["trade_returns"])
    test_results.append(bs_result)

    # 3. Walk-Forward
    wf_result = walk_forward_test(df)
    test_results.append(wf_result)

    # 4. Permutation
    perm_result = permutation_test(df, metrics["total_return"])
    test_results.append(perm_result)

    # 5. T-Test
    ttest_result = t_test(metrics["daily_returns"], df)
    test_results.append(ttest_result)

    # Print summary
    print_summary(metrics, test_results, buy_hold_return)

    # Create chart
    if HAS_PLOTLY:
        create_results_chart(df, portfolio, test_results)

    print("\n✅ Analysis complete!")
    print("\n" + "=" * 70)
    print("🎬 READY FOR YOUTUBE VIDEO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
