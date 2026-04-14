from __future__ import annotations

import numpy as np

from trading_algo.quant_core.strategies.options.iv_rank import (
    iv_rank,
    iv_series_from_prices,
    realized_volatility,
)


def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's smoothed RSI. Returns array same length as prices, NaN-padded."""
    out = np.full(len(prices), np.nan)
    if len(prices) < period + 1:
        return out

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return out


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Standard ADX with Wilder smoothing. Returns array same length as closes, NaN-padded."""
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period * 2 + 1:
        return out

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        h_l = highs[i] - lows[i]
        h_pc = abs(highs[i] - closes[i - 1])
        l_pc = abs(lows[i] - closes[i - 1])
        tr[i] = max(h_l, h_pc, l_pc)

        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    # Wilder smoothing for TR, +DM, -DM
    atr = np.sum(tr[1 : period + 1])
    apdm = np.sum(plus_dm[1 : period + 1])
    amdm = np.sum(minus_dm[1 : period + 1])

    dx_vals: list[float] = []

    for i in range(period + 1, n):
        atr = atr - atr / period + tr[i]
        apdm = apdm - apdm / period + plus_dm[i]
        amdm = amdm - amdm / period + minus_dm[i]

        if atr == 0:
            dx_vals.append(0.0)
            continue

        plus_di = 100.0 * apdm / atr
        minus_di = 100.0 * amdm / atr
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_vals.append(0.0)
        else:
            dx_vals.append(100.0 * abs(plus_di - minus_di) / di_sum)

    if len(dx_vals) < period:
        return out

    adx_val = np.mean(dx_vals[:period])
    out[period * 2] = adx_val
    for j in range(period, len(dx_vals)):
        adx_val = (adx_val * (period - 1) + dx_vals[j]) / period
        out[period + 1 + j] = adx_val

    return out


class ATLASFeatureComputer:
    def compute_features(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """Returns (T, 12) array of market features. Last 4 features (position state) added separately."""
        closes = np.asarray(closes, dtype=np.float64)
        highs = np.asarray(highs, dtype=np.float64)
        lows = np.asarray(lows, dtype=np.float64)
        volumes = np.asarray(volumes, dtype=np.float64)

        T = len(closes)
        features = np.full((T, 12), np.nan, dtype=np.float64)

        log_prices = np.log(closes)

        # r_1: 1-day log return
        features[1:, 0] = np.diff(log_prices)

        # r_5: 5-day cumulative log return
        for t in range(5, T):
            features[t, 1] = log_prices[t] - log_prices[t - 5]

        # r_21: 21-day cumulative log return
        for t in range(21, T):
            features[t, 2] = log_prices[t] - log_prices[t - 21]

        # r_63: 63-day cumulative log return
        for t in range(63, T):
            features[t, 3] = log_prices[t] - log_prices[t - 63]

        # rv_30: 30-day realized volatility (annualized)
        rv_30 = realized_volatility(closes, window=30)
        features[:, 4] = rv_30

        # iv_est: estimated IV
        iv_est = iv_series_from_prices(closes, rv_window=30)
        features[:, 5] = iv_est

        # iv_rank_val: IV rank over 252-day lookback
        for t in range(T):
            if not np.isnan(iv_est[t]):
                features[t, 6] = iv_rank(iv_est, t, lookback=252)

        # rsi_14: RSI(14) rescaled to [-1, 1]
        rsi_raw = _rsi(closes, period=14)
        features[:, 7] = (rsi_raw - 50.0) / 50.0

        # adx_14: ADX(14) rescaled to [0, 1]
        adx_raw = _adx(highs, lows, closes, period=14)
        features[:, 8] = adx_raw / 100.0

        # vol_ratio: 5-day avg volume / 20-day avg volume
        for t in range(19, T):
            avg5 = np.mean(volumes[max(0, t - 4) : t + 1])
            avg20 = np.mean(volumes[t - 19 : t + 1])
            if avg20 > 0:
                features[t, 9] = avg5 / avg20
            else:
                features[t, 9] = 1.0

        # ts_ratio: 10-day RV / 60-day RV (term structure)
        rv_10 = realized_volatility(closes, window=10)
        rv_60 = realized_volatility(closes, window=60)
        for t in range(T):
            if not np.isnan(rv_10[t]) and not np.isnan(rv_60[t]) and rv_60[t] > 0:
                features[t, 10] = rv_10[t] / rv_60[t]

        # skew_est: iv_est / rv_30 - 1.0
        for t in range(T):
            if not np.isnan(iv_est[t]) and not np.isnan(rv_30[t]) and rv_30[t] > 0:
                features[t, 11] = iv_est[t] / rv_30[t] - 1.0

        return features


class RollingNormalizer:
    def normalize(
        self, features: np.ndarray, lookback: int = 252
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (normalized_features, mu_array, sigma_array) -- all shape (T, F)."""
        T, F = features.shape
        mu = np.full((T, F), np.nan, dtype=np.float64)
        sigma = np.full((T, F), np.nan, dtype=np.float64)
        normed = np.full((T, F), np.nan, dtype=np.float64)
        eps = 1e-8

        for t in range(T):
            start = max(0, t - lookback + 1)
            window = features[start : t + 1]
            for f in range(F):
                col = window[:, f]
                valid = col[~np.isnan(col)]
                if len(valid) < 2:
                    mu[t, f] = 0.0
                    sigma[t, f] = 1.0
                    if not np.isnan(features[t, f]):
                        normed[t, f] = features[t, f]
                    continue
                m = np.mean(valid)
                s = np.std(valid, ddof=1)
                mu[t, f] = m
                sigma[t, f] = s
                if not np.isnan(features[t, f]):
                    normed[t, f] = (features[t, f] - m) / (s + eps)

        return normed, mu, sigma
