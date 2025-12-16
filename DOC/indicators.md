# Available Indicators

The `tsbarbell` library provides access to several advanced predictive indicators for Bitcoin (BTC). These indicators are designed to help traders and analysts assess market probabilities for trend, volatility, and breakouts.

## 1. BTC Slope Probability

**Access:** `client.btc_slope`

This indicator forecasts the probability of the price trend slope being positive or negative over specific time horizons. It helps identify the strength and direction of the current trend.

- **Available Horizons:** 4h, 12h, 24h, 60h
- **Output:** Probability (0.0 to 1.0)
- **Interpretation:**
  - High probability (> 0.8) suggests a strong upward trend.
  - Low probability (< 0.2) suggests a strong downward trend.

## 2. BTC Break-Up Probability

**Access:** `client.btc_break_up`

This indicator estimates the likelihood of a significant upward volatility breakout. It is useful for identifying potential long entry points during consolidation phases.

- **Available Horizons:** 1h (Short-term forecast)
- **Output:** Probability (0.0 to 1.0)
- **Interpretation:** Higher values indicate a higher likelihood of a sudden price increase.

## 3. BTC Break-Down Probability

**Access:** `client.btc_break_down`

This indicator estimates the likelihood of a significant downward volatility breakout (crash or correction). It is useful for risk management and identifying short entry points.

- **Available Horizons:** 1h (Short-term forecast)
- **Output:** Probability (0.0 to 1.0)
- **Interpretation:** Higher values indicate a higher likelihood of a sudden price drop.

## 4. BTC Volatility Forecast

**Access:** `client.btc_volatility`

This indicator forecasts the probability of high market volatility, regardless of direction. It is particularly useful for options traders or for adjusting position sizing based on expected market turbulence.

- **Available Horizons:** 24h, 48h, 96h
- **Output:** Probability (0.0 to 1.0)
- **Interpretation:**
  - High values suggest expected turbulence/instability.
  - Low values suggest a calm or ranging market.
