import sys
import os

# Add the parent directory to sys.path to allow importing the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tsbarbell import (
    TSBarbellClient,
    plot_slope_probability,
    plot_break_up_probability,
    plot_break_down_probability,
    backtest_strategy,
)


def main():
    # Load API Key from environment variables
    from dotenv import load_dotenv

    load_dotenv()

    API_KEY = os.getenv("API_KEY")
    if not API_KEY:
        print("Error: API_KEY not found in environment variables.")
        return

    print("Initializing client...")
    client = TSBarbellClient(api_key=API_KEY)

    # --- BTC Slope ---
    print("\n=== BTC Slope Probability ===")
    print("Fetching data for all horizons...")
    # Accessing the specific indicator via client.btc_slope
    data_slope = client.btc_slope.get_all_horizons(limit=1000)

    for h, df in data_slope.items():
        print(f"Horizon {h}h: {len(df)} rows fetched.")
        if not df.empty:
            # Run a quick backtest
            stats = backtest_strategy(df, buy_threshold=0.99, sell_threshold=0.01)
            print(
                f"  Strategy Return: {stats.get('total_return_pct')}% ({stats.get('trades_count')} trades)"
            )

    print("Generating Slope plot...")
    fig_slope = plot_slope_probability(data_slope)
    # fig_slope.show()

    # --- BTC Break-Up ---
    print("\n=== BTC Break-Up Probability ===")
    print("Fetching data for all horizons...")
    # Accessing the specific indicator via client.btc_break_up
    data_break = client.btc_break_up.get_all_horizons(limit=1000)

    for h, df in data_break.items():
        print(f"Horizon {h}h: {len(df)} rows fetched.")
        if not df.empty:
            # Run a quick backtest with different thresholds (0.60 / 0.10)
            stats = backtest_strategy(df, buy_threshold=0.60, sell_threshold=0.10)
            print(
                f"  Strategy Return: {stats.get('total_return_pct')}% ({stats.get('trades_count')} trades)"
            )

    print("Generating Break-Up plot...")
    fig_break = plot_break_up_probability(data_break)
    # fig_break.show()

    # --- BTC Break-Down ---
    print("\n=== BTC Break-Down Probability ===")
    print("Fetching data for all horizons...")
    # Accessing the specific indicator via client.btc_break_down
    data_down = client.btc_break_down.get_all_horizons(limit=1000)

    for h, df in data_down.items():
        print(f"Horizon {h}h: {len(df)} rows fetched.")
        if not df.empty:
            # Run a quick backtest with different thresholds (0.60 for SHORT, 0.10 for EXIT)
            # Note: backtest_strategy currently assumes LONG only.
            # For SHORT strategy, we would need a different backtest function or logic.
            # Here we just print the data count.
            pass

    print("Generating Break-Down plot...")
    fig_down = plot_break_down_probability(data_down)
    # fig_down.show()

    # --- Combined Data ---
    print("\n=== Combined Data Example ===")
    print("Fetching combined data for Slope 4h, Break-Up 1h, Break-Down 1h...")

    combined_df = client.get_combined_data(
        [("btc_slope", 4), ("btc_break_up", 1), ("btc_break_down", 1)], limit=100
    )

    if not combined_df.empty:
        print(combined_df.head())
        print(f"Combined shape: {combined_df.shape}")

        # Example: Correlation matrix
        print("\nCorrelation Matrix:")
        print(combined_df.corr(numeric_only=True))

    # Optionally save to HTML
    # fig.write_html("btc_slope_probability.html")
    # print("Plot saved to btc_slope_probability.html")


if __name__ == "__main__":
    main()
