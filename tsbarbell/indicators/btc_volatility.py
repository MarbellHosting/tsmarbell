import pandas as pd
from typing import Optional, Dict, Any
from .base import BaseIndicator


class BTCVolatility(BaseIndicator):
    """
    BTC Volatility Forecast Indicator.

    This indicator forecasts the probability of high volatility for Bitcoin
    over different time horizons (24h, 48h, 96h).
    """

    def __init__(self, client):
        super().__init__(client)
        self.base_endpoint = "btc-volatility-1h"

    def get_probability(self, horizon: int, limit: int = 1000) -> pd.DataFrame:
        """
        Get BTC volatility probability for a specific horizon.

        Args:
            horizon (int): Forecast horizon in hours (24, 48, 96).
            limit (int): Number of data points to fetch.

        Returns:
            pd.DataFrame: DataFrame with timestamp and probability columns.
        """
        if horizon not in [24, 48, 96]:
            raise ValueError("Horizon must be one of: 24, 48, 96")

        endpoint = f"{self.base_endpoint}-h{horizon}"
        data = self.client._fetch(endpoint, limit)

        df = pd.DataFrame(data)
        if not df.empty:
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Ensure probability is numeric
            df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
            # Sort by timestamp just in case
            df = df.sort_values("timestamp")

        return df

    # Alias for backward compatibility
    get_volatility_probability = get_probability

    def get_all_horizons(self, limit: int = 1000) -> Dict[int, pd.DataFrame]:
        """
        Get BTC volatility probability for all supported horizons.

        Args:
            limit (int): Number of data points to fetch per horizon.

        Returns:
            Dict[int, pd.DataFrame]: Dictionary mapping horizon to DataFrame.
        """
        return {h: self.get_probability(h, limit) for h in [24, 48, 96]}
