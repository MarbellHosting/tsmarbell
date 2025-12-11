import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm.auto import tqdm
from typing import Dict, Any, Tuple, Optional, Callable

# Try to import vectorbt
try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

def run_backtest(
    close_price: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float = 10000,
    fees: float = 0.0,
    freq: str = "1h",
) -> Any:
    """
    Run a backtest using VectorBT.
    
    Args:
        close_price: Series of close prices.
        entries: Boolean Series for entry signals.
        exits: Boolean Series for exit signals.
        init_cash: Initial capital.
        fees: Fee per transaction.
        freq: Data frequency.
        
    Returns:
        vectorbt.Portfolio object
    """
    if not HAS_VBT:
        raise ImportError("vectorbt is required for backtesting. Install with: pip install vectorbt")

    # Disable widgets for performance in loops
    vbt.settings.plotting['use_widgets'] = False

    portfolio = vbt.Portfolio.from_signals(
        close=close_price,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
        freq=freq,
    )
    return portfolio

class StatisticalTests:
    """
    A suite of statistical tests for trading strategy validation.
    """
    
    @staticmethod
    def monte_carlo_test(
        close_price: pd.Series,
        actual_return: float,
        entry_signal: pd.Series,
        exit_signal: pd.Series,
        n_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Monte Carlo Test: Can a random strategy achieve the same results?
        Generates random entry/exit signals with same frequency and compares returns.
        """
        print("🎲 Running Monte Carlo Simulation...")

        # Calculate actual signal frequency
        entry_freq = entry_signal.sum() / len(entry_signal)
        exit_freq = exit_signal.sum() / len(exit_signal)

        random_returns = []

        for _ in tqdm(range(n_simulations), desc="Simulating"):
            # Generate random signals with SAME frequency as actual strategy
            n = len(close_price)
            random_entries = pd.Series(np.random.random(n) < entry_freq, index=close_price.index)
            random_exits = pd.Series(np.random.random(n) < exit_freq, index=close_price.index)

            try:
                portfolio = run_backtest(close_price, random_entries, random_exits)
                random_returns.append(portfolio.total_return() * 100)
            except Exception:
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

    @staticmethod
    def bootstrap_test(
        trade_returns: np.ndarray, 
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Bootstrap Test: How stable are the returns?
        Resamples trades with replacement to estimate confidence interval.
        """
        print("📊 Running Bootstrap Analysis...")

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

        for _ in tqdm(range(n_simulations), desc="Bootstrapping"):
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

    @staticmethod
    def walk_forward_test(
        df: pd.DataFrame, 
        signal_generator: Callable[[pd.DataFrame], Tuple[pd.Series, pd.Series]],
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Walk-Forward Test: Does the strategy work on new, unseen data?
        
        Args:
            df: DataFrame containing price and indicator data. Must have 'close' column.
            signal_generator: Function that takes a DataFrame and returns (entries, exits) Series.
            n_folds: Number of folds to split the data into.
        """
        print("🚶 Running Walk-Forward Analysis...")

        fold_size = len(df) // (n_folds + 1)
        fold_returns = []

        for i in tqdm(range(n_folds), desc="Testing folds"):
            # Define train and test periods
            test_start = (i + 1) * fold_size
            test_end = (i + 2) * fold_size if i < n_folds - 1 else len(df)

            test_df = df.iloc[test_start:test_end].copy()

            if len(test_df) < 100:
                continue

            # Generate signals and run backtest
            entry_signal, exit_signal = signal_generator(test_df)

            try:
                portfolio = run_backtest(test_df["close"], entry_signal, exit_signal)
                fold_return = portfolio.total_return() * 100
                fold_returns.append(fold_return)
            except Exception:
                fold_returns.append(0)

        profitable_folds = sum(1 for r in fold_returns if r > 0)
        avg_return = np.mean(fold_returns) if fold_returns else 0

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

    @staticmethod
    def permutation_test(
        close_price: pd.Series,
        actual_return: float,
        entry_signal: pd.Series,
        exit_signal: pd.Series,
        n_permutations: int = 1000
    ) -> Dict[str, Any]:
        """
        Permutation Test: Do the signals have real predictive power?
        Shuffles signals randomly and compares to original.
        """
        print("🔀 Running Permutation Test...")

        permuted_returns = []

        for _ in tqdm(range(n_permutations), desc="Permuting"):
            # Shuffle signals
            shuffled_entries = entry_signal.sample(frac=1).values
            shuffled_exits = exit_signal.sample(frac=1).values

            shuffled_entry_series = pd.Series(shuffled_entries, index=close_price.index)
            shuffled_exit_series = pd.Series(shuffled_exits, index=close_price.index)

            try:
                portfolio = run_backtest(close_price, shuffled_entry_series, shuffled_exit_series)
                permuted_returns.append(portfolio.total_return() * 100)
            except Exception:
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

    @staticmethod
    def t_test(
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """
        T-Test (Strategy vs Benchmark)
        Question: Is the outperformance statistically significant?
        """
        print("📈 Running T-Test (Strategy vs Benchmark)...")

        # Align the series
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_aligned = strategy_returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]

        # Perform t-test
        t_stat, p_value = scipy_stats.ttest_ind(strategy_aligned, benchmark_aligned)

        passed = p_value < 0.05 and t_stat > 0

        return {
            "name": "T-Test (Strategy vs Benchmark)",
            "question": "Is the outperformance statistically significant?",
            "t_statistic": t_stat,
            "p_value": p_value,
            "passed": passed,
            "result": f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}",
        }
