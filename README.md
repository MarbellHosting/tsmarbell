# TSBarbell Python Library

A comprehensive Python library for interacting with TSBarbell AI trading indicators, visualizing data, and performing rigorous statistical analysis.

## Installation

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Fetching & Visualizing Indicators

```python
from tsbarbell import (
    TSBarbellClient, 
    plot_slope_probability, 
    plot_break_up_probability
)

# Initialize the client
client = TSBarbellClient(api_key='YOUR_API_KEY')

# Fetch BTC Slope Probability (Trend)
slope_data = client.btc_slope.get_all_horizons(limit=1000)
plot_slope_probability(slope_data).show()

# Fetch BTC Break-Up Probability (Volatility)
break_up_data = client.btc_break_up.get_all_horizons(limit=1000)
plot_break_up_probability(break_up_data).show()

# Fetch Combined Data for Backtesting
df = client.get_combined_data([
    ('btc_slope', 4),
    ('btc_break_up', 1)
], limit=2000)
```

### 2. Statistical Analysis & Backtesting

Validate your strategies using professional statistical tests (Monte Carlo, Bootstrap, Walk-Forward, etc.).

```python
from tsbarbell import StatisticalTests, run_backtest, plot_statistical_analysis

# ... (define your entry_signal and exit_signal logic) ...

# Run Backtest
portfolio = run_backtest(df['close'], entry_signal, exit_signal)

# Run Statistical Tests
mc_result = StatisticalTests.monte_carlo_test(
    close_price=df['close'],
    actual_return=portfolio.total_return() * 100,
    entry_signal=entry_signal,
    exit_signal=exit_signal
)

# Visualize Results Dashboard
fig = plot_statistical_analysis(
    df=df,
    portfolio=portfolio,
    test_results=[mc_result], # Add other test results here
    metrics={"total_return": portfolio.total_return() * 100}
)
fig.show()
```

## Features

- **AI Indicators**: Access to BTC Slope (Trend), Break-Up, and Break-Down probabilities.
- **Data Management**: Easy fetching and combining of multiple indicators into a single Pandas DataFrame.
- **Visualization**: Interactive Plotly charts for all indicators.
- **Statistical Validation**: Built-in suite of tests to prove strategy significance:
    - Monte Carlo Simulation
    - Bootstrap Analysis
    - Walk-Forward Validation
    - Permutation Tests
    - T-Tests
- **Backtesting**: Integration with `vectorbt` for fast portfolio simulation.

## Structure

- `tsbarbell/client.py`: Main API client.
- `tsbarbell/indicators/`: Modules for specific indicators.
- `tsbarbell/plotting/`: Visualization tools and dashboards.
- `tsbarbell/analysis.py`: Statistical tests and backtesting utilities.
- `examples/`: Demo notebooks and scripts.
