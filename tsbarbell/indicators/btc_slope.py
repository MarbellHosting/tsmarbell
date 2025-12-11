import pandas as pd
from typing import Dict, Union
from .base import BaseIndicator


class BTCSlopeIndicator(BaseIndicator):
    """
    Indicator: BTC Slope Probability
    """

    def get_probability(self, horizon: int, limit: int = 10000) -> pd.DataFrame:
        """
        Fetch BTC Slope Probability for a specific horizon.

        Args:
            horizon (int): The horizon in hours. Supported values: 4, 12, 24, 60.
            limit (int): Number of data points to fetch.

        Returns:
            pd.DataFrame: DataFrame containing timestamp, probability, and interval.
        """
        endpoint_map = {
            4: "btc-slope-1h-h4",
            12: "btc-slope-1h-h12",
            24: "btc-slope-1h-h24",
            60: "btc-slope-1h-h60",
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
        for h in [4, 12, 24, 60]:
            try:
                results[h] = self.get_probability(h, limit)
            except Exception as e:
                print(f"Failed to fetch horizon {h}: {e}")
        return results
