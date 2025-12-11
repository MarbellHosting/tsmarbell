import pandas as pd
from typing import Dict, Union
from .base import BaseIndicator


class BTCBreakDownIndicator(BaseIndicator):
    """
    Indicator: BTC Break-Down Probability
    """

    def get_probability(self, horizon: int, limit: int = 10000) -> pd.DataFrame:
        """
        Fetch BTC Break-Down Probability for a specific horizon.

        Args:
            horizon (int): The horizon in hours. Supported values: 1, 2, 4, 8.
            limit (int): Number of data points to fetch.

        Returns:
            pd.DataFrame: DataFrame containing timestamp, probability, and interval.
        """
        endpoint_map = {
            1: "btc-break-1h-down-h1",
            2: "btc-break-1h-down-h2",
            4: "btc-break-1h-down-h4",
            8: "btc-break-1h-down-h8",
        }

        if horizon not in endpoint_map:
            raise ValueError(
                f"Unsupported horizon: {horizon}. Supported: {list(endpoint_map.keys())}"
            )

        data = self.client._fetch(endpoint_map[horizon], limit)

        df = pd.DataFrame(data)
        if not df.empty:
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Ensure probability is numeric
            df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
            # Sort by timestamp just in case
            df = df.sort_values("timestamp")

        return df

    def get_all_horizons(self, limit: int = 10000) -> Dict[int, pd.DataFrame]:
        """
        Fetch data for all supported horizons.
        """
        results = {}
        for h in [1, 2, 4, 8]:
            try:
                results[h] = self.get_probability(h, limit)
            except Exception as e:
                print(f"Failed to fetch horizon {h}: {e}")
        return results
