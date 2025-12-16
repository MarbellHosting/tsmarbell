import requests
import pandas as pd
from typing import List, Dict, Any, Tuple
from .indicators.btc_slope import BTCSlopeIndicator
from .indicators.btc_break_up import BTCBreakUpIndicator
from .indicators.btc_break_down import BTCBreakDownIndicator
from .indicators.btc_volatility import BTCVolatility


class TSBarbellClient:
    BASE_URL = "https://ts.marbell.com"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": self.api_key})

        # Initialize indicators
        self.btc_slope = BTCSlopeIndicator(self)
        self.btc_break_up = BTCBreakUpIndicator(self)
        self.btc_break_down = BTCBreakDownIndicator(self)
        self.btc_volatility = BTCVolatility(self)

    def _fetch(self, endpoint: str, limit: int = 10000) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}/{endpoint}"
        params = {"limit": limit, "expand": "false"}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_combined_data(
        self, indicators: List[Tuple[str, int]], limit: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch and combine data from multiple indicators into a single DataFrame.

        Args:
            indicators: List of tuples (indicator_name, horizon).
                        Example: [('btc_slope', 4), ('btc_break_up', 1)]
            limit: Number of data points to fetch per indicator.

        Returns:
            pd.DataFrame: Combined DataFrame indexed by timestamp, with columns for each indicator.
        """
        dfs = []
        price_df = None

        for ind_name, horizon in indicators:
            indicator = getattr(self, ind_name, None)
            if not indicator:
                print(f"Warning: Unknown indicator '{ind_name}', skipping.")
                continue

            try:
                df = indicator.get_probability(horizon, limit)
            except Exception as e:
                print(f"Error fetching {ind_name} (h={horizon}): {e}")
                continue

            if df.empty:
                continue

            # Capture price from the first available source
            if price_df is None and "close_now" in df.columns:
                price_df = (
                    df[["timestamp", "close_now"]]
                    .set_index("timestamp")
                    .rename(columns={"close_now": "close"})
                )

            # Rename probability column to be unique
            col_name = f"{ind_name}_{horizon}h"
            subset = df[["timestamp", "probability"]].rename(
                columns={"probability": col_name}
            )
            subset = subset.set_index("timestamp")
            dfs.append(subset)

        if not dfs:
            return pd.DataFrame()

        # Add price dataframe if we found one
        if price_df is not None:
            dfs.append(price_df)

        # Combine all dataframes
        combined = pd.concat(dfs, axis=1).sort_index()
        return combined.reset_index()
