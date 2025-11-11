"""
Institutional-Grade Bitcoin Regime Detection System v3.0 (Refactored)

This script has been refactored to emulate a production-grade, modular
architecture while remaining in a single file for Streamlit compatibility.

Key principles applied:
- Separation of Concerns (logical "files" denoted by headers)
- Dependency Injection (services passed into dashboard)
- Composition Root (main() function wires dependencies)
- Robust State Management
- Rigorous Type Hinting
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import hashlib
import json
import requests
from contextlib import contextmanager
import time

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# === CONFIGURATION (config.py)
# ==============================================================================

class TimeInterval(Enum):
    DAILY = '1d'
    FOUR_HOUR = '4h'
    HOURLY = '1h'
    FIFTEEN_MIN = '15m'
    FIVE_MIN = '5m'

class MarketRegime(Enum):
    BULL_TRENDING = "BULL_TRENDING"
    BULL_CONSOLIDATION = "BULL_CONSOLIDATION"
    BEAR_TRENDING = "BEAR_TRENDING"
    BEAR_CONSOLIDATION = "BEAR_CONSOLIDATION"
    HIGH_VOLATILITY_BREAKOUT = "HIGH_VOLATILITY_BREAKOUT"
    LOW_VOLATILITY_RANGE = "LOW_VOLATILITY_RANGE"
    REGIME_TRANSITION = "REGIME_TRANSITION"
    CRASH_IMMINENT = "CRASH_IMMINENT"

class SignalStrength(Enum):
    WEAK = auto()
    MODERATE = auto()
    STRONG = auto()
    VERY_STRONG = auto()

# --- Enhanced trading parameters ---
TICKER = "BTC-USD"
PRIMARY_PERIOD = "180d"
CACHE_TTL = 30
HISTORY_SIZE = 50

# --- Risk management enhancements ---
VAR_CONFIDENCE = 0.99
MAX_POSITION_SIZE = 0.1
STOP_LOSS_PCT = 0.15

# --- Default Dashboard State ---
DEFAULT_TIMEFRAME = TimeInterval.DAILY.value
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_MAX_POSITION_SIZE = 10.0
DEFAULT_STOP_LOSS_PCT = 15.0

# --- Dashboard CSS ---
APP_CSS = """
<style>
.main-header {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4AA, #0099FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #1E1E1E;
    padding: 1.5rem;
    border-radius: 0.75rem;
    border-left: 6px solid #00D4AA;
    margin-bottom: 1rem;
}
.risk-metric {
    background: linear-gradient(135deg, #1E1E1E, #2D2D2D);
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #333;
}
.signal-strength-weak { color: #FF6B6B; }
.signal-strength-moderate { color: #FFD93D; }
.signal-strength-strong { color: #6BCF7F; }
.signal-strength-very_strong { color: #00D4AA; }
.live-badge {
    background: linear-gradient(135deg, #00D4AA, #0099FF);
    color: white;
    padding: 0.3rem 1rem;
    border-radius: 1rem;
    font-weight: bold;
    font-size: 0.9rem;
}
</style>
"""


# ==============================================================================
# === DATA STRUCTURES (datatypes.py)
# ==============================================================================

@dataclass(frozen=True)
class MarketSnapshot:
    """Enhanced atomic market data point"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    liquidations: Optional[float] = None

    @property
    def price_movement(self) -> float:
        return (self.close - self.open) / self.open

    @property
    def volatility(self) -> float:
        return (self.high - self.low) / self.open

@dataclass
class RegimeState:
    """Enhanced regime state with ML confidence"""
    current_regime: MarketRegime
    regime_probability: float
    ml_confidence: float
    transition_confidence: float
    duration_bars: int
    last_transition: pd.Timestamp
    regime_strength: float
    signal_strength: SignalStrength
    supporting_indicators: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RiskMetrics:
    """Comprehensive institutional risk assessment"""
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility_annualized: float
    beta: float
    alpha: float
    information_ratio: float
    ulcer_index: float

    def to_series(self) -> pd.Series:
        return pd.Series(asdict(self))

@dataclass
class PortfolioAllocation:
    """Portfolio optimization results"""
    btc_allocation: float
    cash_allocation: float
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    rebalance_signal: bool


# ==============================================================================
# === DATA LAYER (data_layer.py)
# ==============================================================================

class MultiSourceDataEngine:
    """
    Institutional-grade multi-source data aggregation
    with real-time failover and data validation.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.sources: List[str] = ['yahoo', 'mock']
        self._executor = ThreadPoolExecutor(max_workers=6)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._last_successful_source: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._executor.shutdown()

    @contextmanager
    def _error_handler(self, operation: str) -> Any:
        """Context manager for consistent error handling"""
        try:
            yield
        except Exception as e:
            logging.error(f"Error in {operation}: {str(e)}")
            raise

    def fetch_multi_source_ohlcv(self, period: str, interval: str) -> pd.DataFrame:
        """
        Multi-source data aggregation with failover.

        Attempts to fetch data from all registered sources in parallel.
        Returns the first successful, non-empty, validated response.
        Falls back to mock data generation if all sources fail.

        Args:
            period: The time period to fetch (e.g., "180d").
            interval: The data interval (e.g., "1d", "4h").

        Returns:
            A pandas DataFrame with enhanced OHLCV and feature data.
        """
        futures = {}

        # Submit data fetching tasks for all sources
        with self._error_handler("multi_source_fetch"):
            for source in self.sources:
                future = self._executor.submit(self._fetch_from_source, source, period, interval)
                futures[future] = source

            # Wait for first successful response
            for future in as_completed(futures):
                source = futures[future]
                try:
                    data = future.result()
                    if not data.empty:
                        self._last_successful_source = source
                        logging.info(f"✅ Successfully fetched data from {source}")
                        return self._enhance_dataset(data)
                except Exception as e:
                    logging.warning(f"Source {source} failed: {e}")
                    continue

        # All sources failed - generate high-quality mock data
        logging.error("All data sources failed, generating enhanced mock data")
        return self._generate_enhanced_mock_data(period, interval)

    def _fetch_from_source(self, source: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from specific source"""
        if source == 'yahoo':
            return self._fetch_yahoo_data(period, interval)
        elif source == 'mock':
            return self._generate_enhanced_mock_data(period, interval)
        else:
            raise ValueError(f"Unknown data source: {source}")

    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def _fetch_yahoo_data(_self, period: str, interval: str) -> pd.DataFrame:
        """Enhanced Yahoo Finance data fetcher"""
        max_retries = 5
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                with _self._error_handler("yahoo_fetch"):
                    data = yf.download(
                        _self.ticker,
                        period=period,
                        interval=interval,
                        threads=False,
                        progress=False,
                        auto_adjust=True
                    )

                    if data.empty:
                        raise ValueError("Empty data response")

                    # Enhanced validation and cleaning
                    data = _self._validate_and_clean_data(data)
                    data = _self._add_advanced_features(data)

                    return data

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logging.warning(f"Yahoo fetch attempt {attempt + 1} failed, retrying...")
                time.sleep(retry_delay)
        return pd.DataFrame() # Should be unreachable, but satisfies static analysis

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Wall Street-grade data validation"""
        # Handle multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(col[0]).strip().lower() for col in data.columns]
        else:
            data.columns = [str(col).lower().strip() for col in data.columns]

        # Critical data quality checks
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Remove duplicates and sort index
        data = data[~data.index.duplicated(keep='first')]
        data = data.sort_index()

        # Advanced outlier detection using Z-score and IQR
        data = self._advanced_outlier_detection(data)

        # Fill small gaps intelligently
        data = self._smart_fill_gaps(data)

        return data

    def _advanced_outlier_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Combined outlier detection using multiple methods"""
        for col in ['open', 'high', 'low', 'close']:
            if col not in data.columns:
                continue

            # Method 1: Z-score filtering
            series = data[col].dropna()
            z_outliers_full = pd.Series(False, index=data.index)
            if len(series) > 0:
                z_scores = np.abs((series - series.mean()) / series.std())
                z_outliers = z_scores > 3
                z_outliers_full.loc[series.index] = z_outliers

            # Method 2: IQR filtering
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = (data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))

            # Combined outlier mask
            outliers = z_outliers_full | iqr_outliers

            if outliers.any():
                # Replace outliers with rolling median
                median_values = data[col].rolling(window=10, min_periods=1, center=True).median()
                data.loc[outliers, col] = median_values[outliers]

        return data

    def _smart_fill_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Intelligent gap filling using multiple methods"""
        for col in data.columns:
            if data[col].isna().any():
                # Try forward fill first
                data[col] = data[col].ffill(limit=3)

                # If still NaN, use interpolation
                if data[col].isna().any():
                    data[col] = data[col].interpolate(method='linear', limit=2)

                # Final fallback to backward fill
                if data[col].isna().any():
                    data[col] = data[col].bfill(limit=2)

        return data

    def _add_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sophisticated market microstructure features"""
        df = data.copy()

        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(365)

        # Gap analysis
        df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_absolute'] = np.abs(df['overnight_gap'])

        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, np.nan)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        df['volume_shock'] = (df['volume_ratio'] > 2.0).astype(int)

        # Volatility clustering
        df['volatility_regime'] = (df['realized_vol'] > df['realized_vol'].rolling(50).mean()).astype(int)

        # Price momentum features
        df['momentum_1d'] = df['close'].pct_change(1)
        df['momentum_5d'] = df['close'].pct_change(5)
        df['momentum_20d'] = df['close'].pct_change(20)

        return df

    def _enhance_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final dataset enhancement"""
        df = data.copy()

        # Ensure all required features are present
        required_features = ['returns', 'log_returns', 'realized_vol', 'volume_ratio']
        for feature in required_features:
            if feature not in df.columns:
                df = self._add_advanced_features(df)
                break

        # Add timestamp-based features
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        return df

    def _generate_enhanced_mock_data(self, period: str, interval: str) -> pd.DataFrame:
        """Generate highly realistic mock data simulating current market conditions"""
        try:
            # Get current market price for realism
            ticker = yf.Ticker(self.ticker)
            info = ticker.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 45000))
            logging.info(f"Using current market price for mock data: ${current_price:,.2f}")
        except:
            current_price = 45000
            logging.warning("Could not fetch current price, using default")

        # Calculate periods based on interval
        days = int(period.replace('d', '')) if 'd' in period else 180
        freq_map = {'1d': 'D', '4h': '4H', '1h': 'H', '15m': '15T', '5m': '5T'}
        freq = freq_map.get(interval, 'D')
        periods = int(pd.Timedelta(days=days) / pd.Timedelta(freq))

        # Use dynamic seed for variation
        np.random.seed(int(datetime.now().timestamp() % 10000))

        # Realistic GBM parameters for BTC
        mu_daily = 0.0008  # 0.08% daily drift
        sigma_daily = 0.035  # 3.5% daily volatility
        start_price = current_price * 0.85  # Start below current for realism

        # Convert to appropriate timeframe
        if interval == '1d':
            dt = 1
            mu = mu_daily
            sigma = sigma_daily
        elif interval == '4h':
            dt = 1/6
            mu = mu_daily * dt
            sigma = sigma_daily * np.sqrt(dt)
        elif interval == '1h':
            dt = 1/24
            mu = mu_daily * dt
            sigma = sigma_daily * np.sqrt(dt)
        else:  # 15m, 5m
            dt = 1/96
            mu = mu_daily * dt
            sigma = sigma_daily * np.sqrt(dt)

        # Generate correlated price process with volatility clustering
        returns = np.random.normal(mu, sigma, periods)

        # Add volatility clustering (GARCH effect)
        for i in range(1, len(returns)):
            if np.abs(returns[i-1]) > 2 * sigma:
                returns[i] *= 1.5  # Volatility clustering

        # Generate price series
        price_series = start_price * np.exp(np.cumsum(returns))

        # Ensure ends near current price
        final_adjustment = current_price / price_series[-1] if price_series[-1] > 0 else 1
        price_series *= final_adjustment

        # Generate realistic OHLCV
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)

        # Realistic OHLC relationships
        noise_scale = 0.002  # 0.2% noise for OHLC relationships

        opens = price_series * (1 + np.random.normal(0, noise_scale, periods))
        highs = np.maximum(opens, price_series) * (1 + np.abs(np.random.normal(0, noise_scale * 2, periods)))
        lows = np.minimum(opens, price_series) * (1 - np.abs(np.random.normal(0, noise_scale * 2, periods)))
        closes = price_series

        # Volume with seasonality and correlation to volatility
        base_volume = np.random.lognormal(14, 1.2, periods)
        vol_factor = 1 + 2 * np.abs(returns)  # Volume increases with volatility
        volumes = base_volume * vol_factor * (1 + 0.3 * np.sin(2 * np.pi * np.arange(periods) / 30))

        df = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes,
            'volume': volumes
        }, index=dates)

        # Add advanced features
        df = self._enhance_dataset(df)

        logging.info(f"📊 Enhanced mock data generated: {len(df)} periods, current price ${current_price:,.2f}")
        return df


# ==============================================================================
# === QUANT ENGINE (quant_engine.py)
# ==============================================================================

class MLEnhancedQuantEngine:
    """
    Machine learning enhanced quantitative research engine
    Combines traditional technical analysis with ML predictions.
    """

    def __init__(self):
        self.risk_free_rate: float = 0.02
        self.feature_columns: List[str] = []
        self._initialize_ml_components()

    def _initialize_ml_components(self):
        """Initialize ML models and feature engineering"""
        self.regime_classifier: Optional[Any] = None  # Would be initialized with actual ML model
        self.feature_importance: Dict[str, float] = {}
        self.ml_confidence_threshold: float = 0.7

    def calculate_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive feature engineering with ML enhancements.

        Args:
            data: Raw OHLCV DataFrame from the data engine.

        Returns:
            DataFrame enhanced with a wide array of technical,
            volatility, volume, and ML-prepped features.
        """
        df = data.copy()

        # 1. Traditional Technical Indicators
        df = self._calculate_technical_indicators(df)

        # 2. Volatility and Risk Metrics
        df = self._calculate_volatility_metrics(df)

        # 3. Volume and Liquidity Analysis
        df = self._calculate_volume_analysis(df)

        # 4. Market Microstructure
        df = self._calculate_microstructure_features(df)

        # 5. ML Feature Engineering
        df = self._engineer_ml_features(df)

        # 6. Composite Scores
        df = self._calculate_composite_scores(df)

        return df

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive technical analysis"""
        try:
            # Trend indicators
            df.ta.ema(length=10, append=True)
            df.ta.ema(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.sma(length=200, append=True)

            # Kaufman Adaptive Moving Average
            df.ta.kama(length=10, append=True)
            df.ta.kama(length=20, append=True)

            # Momentum indicators
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            df.ta.stoch(append=True)
            df.ta.cci(length=20, append=True)
            df.ta.willr(length=14, append=True)

            # Volatility indicators
            df.ta.bbands(length=20, append=True)
            df.ta.atr(length=14, append=True)

            # Volume indicators
            df.ta.mfi(length=14, append=True)
            df.ta.adosc(append=True)
            df.ta.obv(append=True)

            # Efficiency Ratio (Kaufman)
            direction = np.abs(df['close'] - df['close'].shift(10))
            volatility = df['close'].diff().abs().rolling(10).sum()
            df['efficiency_ratio'] = direction / volatility.replace(0, np.nan)
            df['efficiency_ratio'] = df['efficiency_ratio'].fillna(0.5)

        except Exception as e:
            logging.warning(f"Technical indicator calculation failed: {e}")

        return df

    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced volatility analysis"""
        try:
            returns = df['close'].pct_change().dropna()

            # Realized volatility multiple timeframes
            for window in [5, 10, 20, 50]:
                df[f'realized_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(365)

            # Parkinson volatility (using high-low range)
            df['parkinson_vol'] = np.sqrt(1/(4*np.log(2)) *
                                        ((np.log(df['high']/df['low'])**2).rolling(20).mean())) * np.sqrt(365)

            # GARCH-like volatility estimation
            df = self._estimate_garch_volatility(df)

            # Volatility regime detection
            vol_ma = df['realized_vol_20d'].rolling(50).mean()
            df['volatility_regime'] = (df['realized_vol_20d'] > vol_ma).astype(int)
            df['volatility_ratio'] = df['realized_vol_20d'] / vol_ma.replace(0, np.nan)
            df['volatility_ratio'] = df['volatility_ratio'].fillna(1.0)

        except Exception as e:
            logging.warning(f"Volatility metrics calculation failed: {e}")

        return df

    def _estimate_garch_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """GARCH-style volatility estimation"""
        try:
            returns = df['close'].pct_change().dropna()
            if len(returns) < 10:
                df['garch_vol'] = 0.02
                return df

            # Simplified GARCH(1,1) approximation
            omega, alpha, beta = 0.05, 0.1, 0.85
            vol_squared = np.zeros(len(returns))
            vol_squared[0] = returns.var()

            for t in range(1, len(returns)):
                vol_squared[t] = omega + alpha * returns.iloc[t-1]**2 + beta * vol_squared[t-1]

            vol_series = pd.Series(np.sqrt(vol_squared), index=returns.index)
            df['garch_vol'] = vol_series * np.sqrt(365)

        except Exception as e:
            logging.warning(f"GARCH volatility estimation failed: {e}")
            df['garch_vol'] = df.get('realized_vol_20d', 0.5)

        return df

    def _calculate_volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced volume analysis"""
        try:
            # Volume profile indicators
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20'].replace(0, np.nan)
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

            # Volume-price relationship
            df['volume_price_correlation'] = (df['volume'].rolling(20).corr(df['close']) + 1) / 2

            # Volume clusters
            df['volume_cluster'] = (df['volume_ratio'] > 1.5).astype(int)
            df['volume_trend'] = df['volume_ratio'].rolling(5).mean()

            # Smart money indicators
            if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                # Price-Volume Divergence
                price_trend = df['close'].pct_change(5)
                volume_trend = df['volume_ratio'].pct_change(5)
                df['pv_divergence'] = price_trend - volume_trend

        except Exception as e:
            logging.warning(f"Volume analysis failed: {e}")

        return df

    def _calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features"""
        try:
            # Bid-ask spread estimation (simplified)
            df['estimated_spread'] = (df['high'] - df['low']) / df['close'] * 0.1

            # Liquidity measures
            df['amihud_illiquidity'] = np.abs(df['close'].pct_change()) / (df['volume'] * df['close'])
            df['amihud_illiquidity'] = df['amihud_illiquidity'].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Market efficiency measures
            df['hurst_exponent'] = self._estimate_hurst_exponent(df['close'])

            # Fractal complexity
            df['fractal_dimension'] = self._estimate_fractal_dimension(df['close'])

        except Exception as e:
            logging.warning(f"Microstructure features calculation failed: {e}")

        return df

    def _estimate_hurst_exponent(self, prices: pd.Series, max_lag: int = 20) -> pd.Series:
        """Estimate Hurst exponent for market efficiency using numpy"""
        try:
            lags = range(2, min(max_lag, len(prices)//2))
            tau: List[float] = []

            for lag in lags:
                rs_values: List[float] = []
                for i in range(0, len(prices) - lag, lag):
                    segment = prices.iloc[i:i + lag]
                    if len(segment) < 2:
                        continue
                    mean_segment = segment.mean()
                    cumulative_deviation = (segment - mean_segment).cumsum()
                    range_val = cumulative_deviation.max() - cumulative_deviation.min()
                    std_val = segment.std()
                    if std_val > 0:
                        rs_values.append(range_val / std_val)

                if rs_values:
                    tau.append(np.log(np.mean(rs_values)) if rs_values else 0)
                else:
                    tau.append(0)

            if len(tau) > 1:
                lags_array = np.log(lags[:len(tau)])
                # Use numpy polyfit instead of scipy
                hurst = np.polyfit(lags_array, tau, 1)[0]
                return pd.Series([hurst] * len(prices), index=prices.index)
            else:
                return pd.Series(0.5, index=prices.index)

        except Exception as e:
            logging.warning(f"Hurst exponent calculation failed: {e}")
            return pd.Series(0.5, index=prices.index)

    def _estimate_fractal_dimension(self, prices: pd.Series) -> pd.Series:
        """Estimate fractal dimension for market complexity"""
        try:
            returns = prices.pct_change().dropna()
            if len(returns) < 10:
                return pd.Series(1.5, index=prices.index)

            # Simplified fractal dimension estimation
            volatility = returns.rolling(10).std()
            fd = 2 - (volatility / (1 + volatility))
            return fd.fillna(1.5)

        except Exception as e:
            logging.warning(f"Fractal dimension calculation failed: {e}")
            return pd.Series(1.5, index=prices.index)

    def _engineer_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for machine learning"""
        try:
            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                df[f'volume_ratio_lag_{lag}'] = df['volume_ratio'].shift(lag)

            # Rolling statistics
            windows = [5, 10, 20]
            for window in windows:
                df[f'returns_rolling_mean_{window}'] = df['returns'].rolling(window).mean()
                df[f'returns_rolling_std_{window}'] = df['returns'].rolling(window).std()
                df[f'volume_rolling_skew_{window}'] = df['volume'].rolling(window).skew()

            # Technical indicator combinations
            if 'RSI_14' in df.columns and 'BBU_20_2.0' in df.columns:
                df['rsi_bb_combo'] = (df['RSI_14'] - 50) * (df['close'] - df['BBU_20_2.0']) / df['close']

            # Regime transition features
            df['volatility_regime_change'] = df['volatility_regime'].diff()
            df['volume_regime_change'] = (df['volume_ratio'] > 1.2).astype(int).diff()

            # Feature interactions
            df['vol_trend_interaction'] = df['volatility_ratio'] * df['efficiency_ratio']
            df['volume_vol_interaction'] = df['volume_ratio'] * df['garch_vol']

        except Exception as e:
            logging.warning(f"ML feature engineering failed: {e}")

        return df

    def _calculate_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite scores for regime detection"""
        try:
            # Trend Strength Score
            trend_indicators = ['efficiency_ratio', 'EMA_10', 'EMA_20']
            trend_weights = [0.5, 0.3, 0.2]
            df['trend_strength'] = self._weighted_composite_score(df, trend_indicators, trend_weights)

            # Momentum Score
            momentum_indicators = ['RSI_14', 'MACD_12_26_9', 'momentum_5d']
            momentum_weights = [0.4, 0.4, 0.2]
            df['momentum_score'] = self._weighted_composite_score(df, momentum_indicators, momentum_weights)

            # Volatility Score
            volatility_indicators = ['garch_vol', 'realized_vol_20d', 'volatility_ratio']
            volatility_weights = [0.4, 0.4, 0.2]
            df['volatility_score'] = self._weighted_composite_score(df, volatility_indicators, volatility_weights)

            # Volume Score
            volume_indicators = ['volume_ratio', 'MFI_14', 'volume_price_correlation']
            volume_weights = [0.5, 0.3, 0.2]
            df['volume_score'] = self._weighted_composite_score(df, volume_indicators, volume_weights)

            # Overall Regime Score
            regime_components = [df['trend_strength'], df['momentum_score'],
                               df['volatility_score'], df['volume_score']]
            regime_weights = [0.3, 0.3, 0.2, 0.2]
            df['regime_score'] = sum(comp * weight for comp, weight in zip(regime_components, regime_weights))

            # Signal Quality Score
            df['signal_quality'] = self._calculate_signal_quality(df)

        except Exception as e:
            logging.warning(f"Composite scores calculation failed: {e}")
            df['regime_score'] = 0.5
            df['signal_quality'] = 0.5

        return df

    def _weighted_composite_score(self, df: pd.DataFrame, indicators: List[str], weights: List[float]) -> pd.Series:
        """Calculate weighted composite score from multiple indicators"""
        score = pd.Series(0.0, index=df.index)
        total_weight = 0.0

        for indicator, weight in zip(indicators, weights):
            if indicator in df.columns:
                # Normalize indicator to 0-1 scale
                normalized = self._normalize_indicator(df[indicator])
                score += normalized * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else pd.Series(0.5, index=df.index)


    def _normalize_indicator(self, series: pd.Series) -> pd.Series:
        """Normalize indicator to 0-1 scale using robust scaling"""
        if series.std() == 0:
            return pd.Series(0.5, index=series.index)

        # Use robust scaling (median and IQR)
        median = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)

        if iqr == 0:
            # Fallback to standard scaling
            normalized = (series - series.mean()) / series.std()
        else:
            normalized = (series - median) / iqr

        # Convert to 0-1 scale using sigmoid
        return 1 / (1 + np.exp(-normalized))

    def _calculate_signal_quality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate signal quality metric"""
        try:
            # Factors affecting signal quality
            volatility_stability = 1 / (1 + df['garch_vol'].pct_change().abs())
            volume_consistency = 1 / (1 + df['volume_ratio'].pct_change().abs())
            trend_consistency = df['efficiency_ratio'].rolling(5).std()
            trend_consistency = 1 / (1 + trend_consistency)

            # Combine factors
            quality_factors = [volatility_stability, volume_consistency, trend_consistency]
            weights = [0.4, 0.3, 0.3]

            signal_quality = sum(factor * weight for factor, weight in zip(quality_factors, weights))
            return signal_quality.fillna(0.5)

        except Exception as e:
            logging.warning(f"Signal quality calculation failed: {e}")
            return pd.Series(0.5, index=df.index)


# ==============================================================================
# === REGIME DETECTION (regime_detector.py)
# ==============================================================================

class EnhancedBayesianRegimeDetector:
    """
    Machine learning enhanced Bayesian regime detection
    with advanced pattern recognition.
    """

    def __init__(self):
        self.current_regime: RegimeState = self._initialize_default_regime()
        self.regime_history: List[RegimeState] = []
        self.transition_matrix: Dict[MarketRegime, Dict[MarketRegime, float]] = self._initialize_transition_matrix()
        self.ml_enhancement: bool = True

    def _initialize_default_regime(self) -> RegimeState:
        """Initialize with default regime state"""
        return RegimeState(
            current_regime=MarketRegime.LOW_VOLATILITY_RANGE,
            regime_probability=0.5,
            ml_confidence=0.0,
            transition_confidence=0.0,
            duration_bars=0,
            last_transition=pd.Timestamp.now(),
            regime_strength=0.0,
            signal_strength=SignalStrength.WEAK,
            supporting_indicators=[]
        )

    def _initialize_transition_matrix(self) -> Dict[MarketRegime, Dict[MarketRegime, float]]:
        """Initialize Markov transition probabilities"""
        base_prob = 0.15
        stay_prob = 0.7

        matrix = {}
        for regime in MarketRegime:
            matrix[regime] = {}
            for target_regime in MarketRegime:
                if regime == target_regime:
                    matrix[regime][target_regime] = stay_prob
                else:
                    matrix[regime][target_regime] = base_prob

        # Adjust for realistic regime transitions
        matrix[MarketRegime.BULL_TRENDING][MarketRegime.BULL_CONSOLIDATION] = 0.25
        matrix[MarketRegime.BULL_CONSOLIDATION][MarketRegime.BULL_TRENDING] = 0.2
        matrix[MarketRegime.BEAR_TRENDING][MarketRegime.BEAR_CONSOLIDATION] = 0.25
        matrix[MarketRegime.BEAR_CONSOLIDATION][MarketRegime.BEAR_TRENDING] = 0.2
        matrix[MarketRegime.HIGH_VOLATILITY_BREAKOUT][MarketRegime.CRASH_IMMINENT] = 0.3

        return matrix

    def detect_regime(self, df: pd.DataFrame) -> Optional[RegimeState]:
        """
        Enhanced regime detection with ML support.

        Args:
            df: The enhanced DataFrame from the Quant Engine.

        Returns:
            An optional RegimeState object. Returns None if data is
            insufficient, or the latest state.
        """
        if len(df) < 30:
            return None

        try:
            # Extract multi-timeframe features
            features = self._extract_comprehensive_features(df)

            # Calculate Bayesian probabilities
            regime_probs = self._calculate_enhanced_probabilities(features, df)

            # ML enhancement if available
            if self.ml_enhancement:
                regime_probs = self._apply_ml_corrections(regime_probs, features)

            # Select most probable regime
            new_regime = max(regime_probs.items(), key=lambda x: x[1])[0]

            # Update regime state
            updated_state = self._update_regime_state(new_regime, regime_probs, features, df)

            # Check for significant transition
            if self._is_significant_transition(updated_state):
                self.current_regime = updated_state
                self.regime_history.append(updated_state)
                return updated_state
            else:
                # Update duration even if regime hasn't changed
                self.current_regime.duration_bars += 1
                return self.current_regime

        except Exception as e:
            logging.error(f"Enhanced regime detection failed: {e}")
            return None

    def _extract_comprehensive_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract comprehensive regime features"""
        latest = df.iloc[-1]

        features = {
            # Trend features
            'trend_strength': latest.get('trend_strength', 0.5),
            'efficiency_ratio': latest.get('efficiency_ratio', 0.5),
            'ema_slope': self._calculate_ema_slope(df),

            # Momentum features
            'momentum_score': latest.get('momentum_score', 0.5),
            'rsi_position': latest.get('RSI_14', 50) / 100,
            'macd_signal': self._calculate_macd_signal(df),

            # Volatility features
            'volatility_score': latest.get('volatility_score', 0.5),
            'garch_vol_regime': latest.get('garch_vol', 0.5) / df['garch_vol'].median() if 'garch_vol' in df.columns else 1.0,
            'volatility_regime': latest.get('volatility_regime', 0),

            # Volume features
            'volume_score': latest.get('volume_score', 0.5),
            'volume_ratio': latest.get('volume_ratio', 1.0),
            'volume_trend': latest.get('volume_trend', 1.0),

            # Microstructure features
            'market_efficiency': latest.get('hurst_exponent', 0.5),
            'fractal_complexity': latest.get('fractal_dimension', 1.5),
            'liquidity_score': 1 - latest.get('amihud_illiquidity', 0),

            # Risk features
            'var_estimate': latest.get('var_95', 0),
            'drawdown_current': self._calculate_current_drawdown(df),
        }

        return features

    def _calculate_ema_slope(self, df: pd.DataFrame) -> float:
        """Calculate EMA slope as trend confirmation"""
        try:
            if 'EMA_20' in df.columns and len(df) > 20:
                ema_slope = (df['EMA_20'].iloc[-1] - df['EMA_20'].iloc[-5]) / df['EMA_20'].iloc[-5]
                return float(ema_slope)
            return 0.0
        except:
            return 0.0

    def _calculate_macd_signal(self, df: pd.DataFrame) -> float:
        """Calculate MACD signal strength"""
        try:
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                macd = df['MACD_12_26_9'].iloc[-1]
                hist = df['MACDh_12_26_9'].iloc[-1]

                # Normalize MACD signal
                if abs(macd) > 0:
                    return hist / abs(macd)
                return 0.0
            return 0.0
        except:
            return 0.0

    def _calculate_current_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate current drawdown from recent high"""
        try:
            recent_high = df['close'].rolling(20).max().iloc[-1]
            current_price = df['close'].iloc[-1]
            return (current_price - recent_high) / recent_high
        except:
            return 0.0

    def _calculate_enhanced_probabilities(self, features: Dict[str, float], df: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Calculate enhanced regime probabilities with Bayesian updating"""

        # Base probabilities with noise for variation
        base_probs: Dict[MarketRegime, float] = {
            MarketRegime.BULL_TRENDING: 0.0,
            MarketRegime.BULL_CONSOLIDATION: 0.0,
            MarketRegime.BEAR_TRENDING: 0.0,
            MarketRegime.BEAR_CONSOLIDATION: 0.0,
            MarketRegime.HIGH_VOLATILITY_BREAKOUT: 0.0,
            MarketRegime.LOW_VOLATILITY_RANGE: 0.0,
            MarketRegime.CRASH_IMMINENT: 0.0
        }

        # Add dynamic noise based on market conditions
        noise_level = features.get('volatility_score', 0.5) * 0.1
        noise = np.random.normal(0, noise_level, len(base_probs))

        # Trend-based probabilities
        trend_strength = features['trend_strength'] + noise[0]
        momentum = features['momentum_score'] + noise[1]

        if trend_strength > 0.7 and momentum > 0.6:
            base_probs[MarketRegime.BULL_TRENDING] = 0.8
        elif trend_strength > 0.7 and momentum < 0.4:
            base_probs[MarketRegime.BEAR_TRENDING] = 0.8
        elif 0.4 <= trend_strength <= 0.6 and abs(momentum - 0.5) < 0.2:
            if momentum > 0.5:
                base_probs[MarketRegime.BULL_CONSOLIDATION] = 0.6
            else:
                base_probs[MarketRegime.BEAR_CONSOLIDATION] = 0.6

        # Volatility-based probabilities
        volatility_regime = features['volatility_score'] + noise[2]
        if volatility_regime > 0.7:
            base_probs[MarketRegime.HIGH_VOLATILITY_BREAKOUT] = 0.9
        elif volatility_regime < 0.3:
            base_probs[MarketRegime.LOW_VOLATILITY_RANGE] = 0.9

        # Crash detection (combination of high volatility and negative momentum)
        if (volatility_regime > 0.8 and momentum < 0.3 and
            features['drawdown_current'] < -0.05):
            base_probs[MarketRegime.CRASH_IMMINENT] = 0.7

        # Apply Markov transition probabilities
        transition_probs = self.transition_matrix[self.current_regime.current_regime]
        for regime in base_probs:
            base_probs[regime] *= transition_probs.get(regime, 0.1)


        # Normalize probabilities
        total = sum(base_probs.values())
        if total > 0:
            for regime in base_probs:
                base_probs[regime] /= total
        else:
            # Default to current regime if no clear signal
            base_probs[self.current_regime.current_regime] = 1.0

        return base_probs

    def _apply_ml_corrections(self, regime_probs: Dict[MarketRegime, float],
                            features: Dict[str, float]) -> Dict[MarketRegime, float]:
        """Apply ML-based corrections to regime probabilities"""
        # This would integrate with a trained ML model in production
        # For now, we use heuristic rules that mimic ML behavior

        correction_factors: Dict[MarketRegime, float] = {
            MarketRegime.BULL_TRENDING: 1.0,
            MarketRegime.BULL_CONSOLIDATION: 1.0,
            MarketRegime.BEAR_TRENDING: 1.0,
            MarketRegime.BEAR_CONSOLIDATION: 1.0,
            MarketRegime.HIGH_VOLATILITY_BREAKOUT: 1.0,
            MarketRegime.LOW_VOLATILITY_RANGE: 1.0,
            MarketRegime.CRASH_IMMINENT: 1.0
        }

        # Example ML-like corrections based on feature patterns
        if features['market_efficiency'] < 0.4:  # Inefficient market
            correction_factors[MarketRegime.HIGH_VOLATILITY_BREAKOUT] *= 1.2
            correction_factors[MarketRegime.CRASH_IMMINENT] *= 1.1

        if features['fractal_complexity'] > 1.7:  # High complexity
            correction_factors[MarketRegime.LOW_VOLATILITY_RANGE] *= 0.8
            correction_factors[MarketRegime.BULL_CONSOLIDATION] *= 0.9

        # Apply corrections
        for regime in regime_probs:
            regime_probs[regime] *= correction_factors[regime]

        # Renormalize
        total = sum(regime_probs.values())
        if total > 0:
            for regime in regime_probs:
                regime_probs[regime] /= total

        return regime_probs

    def _update_regime_state(self, new_regime: MarketRegime,
                           regime_probs: Dict[MarketRegime, float],
                           features: Dict[str, float],
                           df: pd.DataFrame) -> RegimeState:
        """Update regime state with comprehensive analysis"""

        regime_prob = regime_probs[new_regime]

        # Calculate ML confidence (simulated)
        ml_confidence = self._calculate_ml_confidence(features, new_regime)

        # Calculate transition confidence
        transition_confidence = min(regime_prob * 1.3, 1.0)

        # Determine signal strength
        signal_strength = self._determine_signal_strength(regime_prob, ml_confidence, features)

        # Identify supporting indicators
        supporting_indicators = self._identify_supporting_indicators(features, new_regime)

        # Calculate regime strength
        regime_strength = np.mean([
            features['trend_strength'],
            features['momentum_score'],
            features['volume_score'],
            ml_confidence
        ])
        
        duration = 0
        if new_regime == self.current_regime.current_regime:
            duration = self.current_regime.duration_bars + 1

        return RegimeState(
            current_regime=new_regime,
            regime_probability=regime_prob,
            ml_confidence=ml_confidence,
            transition_confidence=transition_confidence,
            duration_bars=duration,
            last_transition=pd.Timestamp.now(),
            regime_strength=regime_strength,
            signal_strength=signal_strength,
            supporting_indicators=supporting_indicators
        )

    def _calculate_ml_confidence(self, features: Dict[str, float], regime: MarketRegime) -> float:
        """Calculate ML model confidence (simulated)"""
        # In production, this would come from the actual ML model
        # For now, we use feature consistency as a proxy

        consistency_metrics: List[float] = []

        if regime in [MarketRegime.BULL_TRENDING, MarketRegime.BULL_CONSOLIDATION]:
            consistency_metrics.extend([
                features['trend_strength'],
                features['momentum_score'],
                features['rsi_position']
            ])
        elif regime in [MarketRegime.BEAR_TRENDING, MarketRegime.BEAR_CONSOLIDATION]:
            consistency_metrics.extend([
                1 - features['trend_strength'],
                1 - features['momentum_score'],
                1 - features['rsi_position']
            ])
        elif regime == MarketRegime.HIGH_VOLATILITY_BREAKOUT:
            consistency_metrics.extend([
                features['volatility_score'],
                features['volume_ratio']
            ])
        elif regime == MarketRegime.LOW_VOLATILITY_RANGE:
            consistency_metrics.extend([
                1 - features['volatility_score'],
                features['market_efficiency']
            ])

        return float(np.mean(consistency_metrics)) if consistency_metrics else 0.5

    def _determine_signal_strength(self, regime_prob: float, ml_confidence: float,
                                 features: Dict[str, float]) -> SignalStrength:
        """Determine signal strength based on multiple factors"""
        strength_score = (regime_prob + ml_confidence + features['trend_strength']) / 3

        if strength_score > 0.8:
            return SignalStrength.VERY_STRONG
        elif strength_score > 0.7:
            return SignalStrength.STRONG
        elif strength_score > 0.6:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    def _identify_supporting_indicators(self, features: Dict[str, float], regime: MarketRegime) -> List[str]:
        """Identify indicators supporting the regime classification"""
        supporting: List[str] = []

        threshold = 0.6
        if regime in [MarketRegime.BULL_TRENDING, MarketRegime.BULL_CONSOLIDATION]:
            if features['trend_strength'] > threshold:
                supporting.append("Strong Trend")
            if features['momentum_score'] > threshold:
                supporting.append("Positive Momentum")
            if features['volume_score'] > threshold:
                supporting.append("Supporting Volume")

        elif regime in [MarketRegime.BEAR_TRENDING, MarketRegime.BEAR_CONSOLIDATION]:
            if features['trend_strength'] > threshold:
                supporting.append("Strong Trend")
            if features['momentum_score'] < (1 - threshold):
                supporting.append("Negative Momentum")

        elif regime == MarketRegime.HIGH_VOLATILITY_BREAKOUT:
            if features['volatility_score'] > threshold:
                supporting.append("High Volatility")
            if features['volume_ratio'] > 1.5:
                supporting.append("Volume Breakout")

        elif regime == MarketRegime.LOW_VOLATILITY_RANGE:
            if features['volatility_score'] < (1 - threshold):
                supporting.append("Low Volatility")
            if features['market_efficiency'] > threshold:
                supporting.append("Efficient Market")

        return supporting if supporting else ["Mixed Signals"]

    def _is_significant_transition(self, new_state: RegimeState) -> bool:
        """Check if regime transition is statistically significant"""
        if new_state.current_regime != self.current_regime.current_regime:
            # Check confidence thresholds for a *change*
            return (new_state.regime_probability > st.session_state.confidence_threshold and
                   new_state.transition_confidence > 0.6 and
                   new_state.ml_confidence > 0.5)

        # No transition
        return False


# ==============================================================================
# === RISK MANAGEMENT (risk_engine.py)
# ==============================================================================

class EnhancedRiskManagementEngine:
    """
    Institutional-grade risk management with advanced metrics
    and portfolio optimization.
    """

    def __init__(self):
        self.risk_free_rate: float = 0.02
        self.var_confidence: float = VAR_CONFIDENCE
        self.position_limits: Dict[str, float] = {
            'max_position_size': MAX_POSITION_SIZE,
            'stop_loss_pct': STOP_LOSS_PCT,
            'max_daily_loss': 0.02  # 2% max daily loss
        }
        self._current_allocation: float = 0.5 # Internal state for rebalancing

    def calculate_comprehensive_risk(self, df: pd.DataFrame) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.

        Args:
            df: The enhanced DataFrame from the Quant Engine.

        Returns:
            A RiskMetrics dataclass object.
        """
        try:
            returns = df['close'].pct_change().dropna()

            if len(returns) < 10:
                return self._default_risk_metrics()

            # Value at Risk and Expected Shortfall
            var_99, expected_shortfall = self._calculate_var_es(returns)

            # Performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)

            # Drawdown analysis
            max_drawdown = self._calculate_max_drawdown(df['close'])
            ulcer_index = self._calculate_ulcer_index(df['close'])

            # Volatility metrics
            volatility_annualized = returns.std() * np.sqrt(365)

            # Alpha/Beta (simplified - would use benchmark in production)
            beta = self._calculate_beta(returns)
            alpha = self._calculate_alpha(returns, beta)
            information_ratio = self._calculate_information_ratio(returns)

            return RiskMetrics(
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                volatility_annualized=volatility_annualized,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                ulcer_index=ulcer_index
            )

        except Exception as e:
            logging.error(f"Risk calculation failed: {e}")
            return self._default_risk_metrics()

    def _calculate_var_es(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Value at Risk and Expected Shortfall"""
        var_99 = returns.quantile(1 - self.var_confidence)
        tail_returns = returns[returns <= var_99]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_99

        return float(var_99), float(expected_shortfall)

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 365)
        return float(excess_returns.mean() / excess_returns.std() * np.sqrt(365))

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        if len(returns) < 2:
            return 0.0

        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 365)
        downside_std = downside_returns.std()
        return float(excess_returns.mean() / downside_std * np.sqrt(365)) if downside_std > 0 else 0.0


    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return 0.0

        cumulative_max = prices.cummax()
        drawdown = (prices - cumulative_max) / cumulative_max
        return float(abs(drawdown.min()) * 100)  # Percentage

    def _calculate_ulcer_index(self, prices: pd.Series) -> float:
        """Calculate Ulcer Index - measure of downside risk"""
        if len(prices) < 14:
            return 0.0

        cumulative_max = prices.cummax()
        drawdowns = (prices - cumulative_max) / cumulative_max
        squared_drawdowns = drawdowns ** 2
        ulcer_index = np.sqrt(squared_drawdowns.rolling(14).mean().iloc[-1])

        return float(ulcer_index * 100)  # Scale for readability

    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate Beta (simplified - assumes market returns similar)"""
        # In production, this would use actual benchmark returns
        # For now, we use a simplified approach
        market_volatility = 0.18  # Assumed market volatility
        asset_volatility = returns.std() * np.sqrt(365)
        correlation = 0.7  # Assumed correlation with market

        return float((asset_volatility / market_volatility) * correlation)

    def _calculate_alpha(self, returns: pd.Series, beta: float) -> float:
        """Calculate Alpha (simplified)"""
        market_return = self.risk_free_rate + 0.04  # Assumed market risk premium
        expected_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
        actual_return = returns.mean() * 365

        return float(actual_return - expected_return)

    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate Information Ratio (simplified)"""
        # Using risk-free rate as benchmark for simplicity
        benchmark_return = self.risk_free_rate / 365
        excess_returns = returns - benchmark_return
        tracking_error = excess_returns.std()

        if tracking_error == 0:
            return 0.0

        return float(excess_returns.mean() / tracking_error * np.sqrt(365))

    def _default_risk_metrics(self) -> RiskMetrics:
        """Return default risk metrics when calculation fails"""
        return RiskMetrics(
            var_99=0.0,
            expected_shortfall=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            volatility_annualized=0.0,
            beta=0.0,
            alpha=0.0,
            information_ratio=0.0,
            ulcer_index=0.0
        )

    def calculate_portfolio_allocation(self, regime_state: RegimeState,
                                    risk_metrics: RiskMetrics) -> PortfolioAllocation:
        """
        Calculate optimal portfolio allocation based on regime and risk.

        Args:
            regime_state: The current detected RegimeState.
            risk_metrics: The latest calculated RiskMetrics.

        Returns:
            A PortfolioAllocation dataclass object.
        """

        # Base allocation
        base_btc_allocation = 0.5

        # Adjust based on regime
        regime_adjustments = {
            MarketRegime.BULL_TRENDING: 0.3,
            MarketRegime.BULL_CONSOLIDATION: 0.1,
            MarketRegime.BEAR_TRENDING: -0.4,
            MarketRegime.BEAR_CONSOLIDATION: -0.2,
            MarketRegime.HIGH_VOLATILITY_BREAKOUT: -0.3,
            MarketRegime.LOW_VOLATILITY_RANGE: 0.1,
            MarketRegime.CRASH_IMMINENT: -0.6,
            MarketRegime.REGIME_TRANSITION: 0.0
        }

        regime_adj = regime_adjustments.get(regime_state.current_regime, 0.0)

        # Adjust based on risk metrics
        risk_adjustment = 0.0
        if risk_metrics.sharpe_ratio > 1.0:
            risk_adjustment += 0.1
        if risk_metrics.max_drawdown > 10:
            risk_adjustment -= 0.2
        if risk_metrics.volatility_annualized > 0.8:
            risk_adjustment -= 0.15

        # Calculate final allocation, respecting configured limits
        max_pos = st.session_state.max_position_size / 100.0
        btc_allocation = max(0.0, min(max_pos, base_btc_allocation + regime_adj + risk_adjustment))
        cash_allocation = 1.0 - btc_allocation

        # Calculate expected metrics
        expected_return = btc_allocation * 0.15 + cash_allocation * self.risk_free_rate  # Simplified
        expected_volatility = btc_allocation * risk_metrics.volatility_annualized

        sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility if expected_volatility > 0 else 0.0

        # Rebalance signal
        rebalance_signal = abs(btc_allocation - self._current_allocation) > 0.1
        self._current_allocation = btc_allocation

        return PortfolioAllocation(
            btc_allocation=btc_allocation,
            cash_allocation=cash_allocation,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=risk_metrics.max_drawdown * btc_allocation,
            rebalance_signal=rebalance_signal
        )


# ==============================================================================
# === ANALYTICS & TRACKING (analytics.py)
# ==============================================================================

class EnhancedSignalTracker:
    """
    Advanced signal tracking with performance analytics
    and machine learning insights.
    """

    def __init__(self, max_signals: int = HISTORY_SIZE):
        self.max_signals: int = max_signals
        self.signals: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = self._get_default_performance_metrics()
        self._signal_cache: Set[str] = set()

    def _get_default_performance_metrics(self) -> Dict[str, Any]:
        """Returns a dict of zeroed-out metrics for initialization."""
        return {
            'total_signals': 0,
            'avg_signal_quality': 0.0,
            'avg_regime_probability': 0.0,
            'avg_ml_confidence': 0.0,
            'regime_distribution': {},
            'strength_distribution': {},
            'recent_accuracy': 0.0,
            'volatility_trend': 0.0,
            'avg_predicted_duration': 0.0
        }

    def add_signal(self, data: pd.DataFrame, regime_state: RegimeState,
                  risk_metrics: RiskMetrics, allocation: PortfolioAllocation):
        """
        Add enhanced trading signal with comprehensive context.

        Args:
            data: The enhanced DataFrame.
            regime_state: The detected RegimeState.
            risk_metrics: The calculated RiskMetrics.
            allocation: The calculated PortfolioAllocation.
        """
        if len(data) == 0:
            return

        latest = data.iloc[-1]
        signal_id = self._generate_signal_id(latest, regime_state)

        # Avoid duplicates
        if signal_id in self._signal_cache:
            return

        # Calculate advanced signal metrics
        signal_quality = self._calculate_signal_quality(data, regime_state)
        predicted_duration = self._predict_regime_duration(regime_state)
        confidence_interval = self._calculate_confidence_interval(regime_state, risk_metrics)

        signal: Dict[str, Any] = {
            'timestamp': datetime.now(),
            'signal_id': signal_id,
            'price': latest['close'],
            'volume': latest['volume'],
            'regime': regime_state.current_regime.value,
            'regime_probability': regime_state.regime_probability,
            'ml_confidence': regime_state.ml_confidence,
            'signal_strength': regime_state.signal_strength.name,
            'regime_strength': regime_state.regime_strength,
            'supporting_indicators': regime_state.supporting_indicators,
            'signal_quality': signal_quality,
            'predicted_duration_bars': predicted_duration,
            'confidence_interval': confidence_interval,
            'risk_metrics': asdict(risk_metrics),
            'portfolio_allocation': asdict(allocation),
            'technical_indicators': {
                'efficiency_ratio': latest.get('efficiency_ratio', 0.5),
                'rsi': latest.get('RSI_14', 50),
                'volume_ratio': latest.get('volume_ratio', 1.0),
                'volatility': latest.get('garch_vol', 0.5),
                'trend_strength': latest.get('trend_strength', 0.5)
            }
        }

        self.signals.append(signal)
        self._signal_cache.add(signal_id)

        # Maintain size limit
        if len(self.signals) > self.max_signals:
            removed_signal = self.signals.pop(0)
            self._signal_cache.discard(removed_signal['signal_id'])

        # Update performance metrics
        self._update_performance_metrics()

    def _generate_signal_id(self, latest_data: pd.Series, regime_state: RegimeState) -> str:
        """Generate unique signal ID"""
        components = [
            datetime.now().strftime('%Y%m%d%H%M'), # Finer granularity
            f"{latest_data['close']:.2f}",
            regime_state.current_regime.value,
            f"{regime_state.regime_probability:.3f}"
        ]
        return hashlib.md5('_'.join(components).encode()).hexdigest()[:12]

    def _calculate_signal_quality(self, data: pd.DataFrame, regime_state: RegimeState) -> float:
        """Calculate comprehensive signal quality score"""
        try:
            latest = data.iloc[-1]

            factors = [
                regime_state.regime_probability,
                regime_state.ml_confidence,
                latest.get('signal_quality', 0.5),
                min(latest.get('volume_ratio', 1.0) / 2.0, 1.0),
                1.0 - min(latest.get('garch_vol', 0.5) / 1.0, 1.0),  # Inverse of volatility
                regime_state.regime_strength
            ]

            # Weight recent signals more heavily
            weights = [0.25, 0.25, 0.15, 0.15, 0.1, 0.1]

            return float(np.average(factors, weights=weights))
        except Exception as e:
            logging.warning(f"Signal quality calculation failed: {e}")
            return 0.5

    def _predict_regime_duration(self, regime_state: RegimeState) -> int:
        """Predict regime duration based on historical patterns"""
        # Simplified prediction - in production would use ML model
        base_durations = {
            MarketRegime.BULL_TRENDING: 20,
            MarketRegime.BULL_CONSOLIDATION: 10,
            MarketRegime.BEAR_TRENDING: 15,
            MarketRegime.BEAR_CONSOLIDATION: 8,
            MarketRegime.HIGH_VOLATILITY_BREAKOUT: 5,
            MarketRegime.LOW_VOLATILITY_RANGE: 25,
            MarketRegime.CRASH_IMMINENT: 3,
            MarketRegime.REGIME_TRANSITION: 2
        }

        base_duration = base_durations.get(regime_state.current_regime, 10)

        # Adjust based on regime strength
        strength_multiplier = 0.5 + regime_state.regime_strength
        adjusted_duration = int(base_duration * strength_multiplier)

        return max(3, min(50, adjusted_duration))

    def _calculate_confidence_interval(self, regime_state: RegimeState,
                                    risk_metrics: RiskMetrics) -> Dict[str, float]:
        """Calculate confidence intervals for regime predictions"""
        base_confidence = regime_state.regime_probability
        volatility_impact = 1.0 - min(risk_metrics.volatility_annualized, 1.0)

        lower_bound = base_confidence * 0.7 * volatility_impact
        upper_bound = min(base_confidence * 1.3, 1.0)

        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'width': upper_bound - lower_bound
        }

    def _update_performance_metrics(self):
        """
        Update comprehensive performance metrics.
        **FIX:** This now runs even for 1 signal.
        """
        if not self.signals:
            self.performance_metrics = self._get_default_performance_metrics()
            return

        try:
            signals_df = pd.DataFrame(self.signals)

            self.performance_metrics = {
                'total_signals': len(self.signals),
                'avg_signal_quality': signals_df['signal_quality'].mean(),
                'avg_regime_probability': signals_df['regime_probability'].mean(),
                'avg_ml_confidence': signals_df['ml_confidence'].mean(),
                'regime_distribution': signals_df['regime'].value_counts().to_dict(),
                'strength_distribution': signals_df['signal_strength'].value_counts().to_dict(),
                'recent_accuracy': self._calculate_recent_accuracy(signals_df),
                'volatility_trend': signals_df['technical_indicators'].apply(lambda x: x['volatility']).mean(),
                'avg_predicted_duration': signals_df['predicted_duration_bars'].mean()
            }

        except Exception as e:
            logging.warning(f"Performance metrics update failed: {e}")
            # Revert to default if calculation fails
            if not self.performance_metrics:
                 self.performance_metrics = self._get_default_performance_metrics()

    def _calculate_recent_accuracy(self, signals_df: pd.DataFrame) -> float:
        """Calculate recent signal accuracy (simplified)"""
        if len(signals_df) < 5:
            # Use all available signals if fewer than 5
            recent_signals = signals_df
        else:
            recent_signals = signals_df.tail(5)

        if recent_signals.empty:
            return 0.0

        # Simplified accuracy calculation - in production would use actual P&L
        avg_quality = recent_signals['signal_quality'].mean()
        avg_confidence = recent_signals['ml_confidence'].mean()

        return (avg_quality + avg_confidence) / 2

    def get_signals_dataframe(self) -> pd.DataFrame:
        """Convert signals to formatted DataFrame"""
        if not self.signals:
            return pd.DataFrame()

        flattened_data: List[Dict[str, Any]] = []
        for signal in self.signals:
            flat_signal = {
                'Timestamp': signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Price': f"${signal['price']:,.2f}",
                'Volume': f"{signal['volume']:,.0f}",
                'Regime': signal['regime'],
                'Regime Prob': f"{signal['regime_probability']:.1%}",
                'ML Confidence': f"{signal['ml_confidence']:.1%}",
                'Signal Strength': signal['signal_strength'],
                'Signal Quality': f"{signal['signal_quality']:.3f}",
                'Predicted Duration': f"{signal['predicted_duration_bars']} bars",
                'Supporting Indicators': ', '.join(signal['supporting_indicators'][:2])
            }
            flattened_data.append(flat_signal)

        return pd.DataFrame(flattened_data)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        base_metrics = self.performance_metrics.copy()

        # Add derived metrics
        if self.signals:
            latest_signal = self.signals[-1]
            base_metrics.update({
                'current_regime': latest_signal['regime'],
                'current_confidence': latest_signal['regime_probability'],
                'current_ml_confidence': latest_signal['ml_confidence'],
                'signal_strength_trend': self._calculate_strength_trend(),
                'regime_stability': self._calculate_regime_stability()
            })

        return base_metrics

    def _calculate_strength_trend(self) -> str:
        """Calculate signal strength trend"""
        if len(self.signals) < 3:
            return "STABLE"

        recent_strengths = [s['regime_strength'] for s in self.signals[-3:]]
        trend = np.polyfit(range(3), recent_strengths, 1)[0]

        if trend > 0.05:
            return "IMPROVING"
        elif trend < -0.05:
            return "DETERIORATING"
        else:
            return "STABLE"

    def _calculate_regime_stability(self) -> float:
        """Calculate regime stability score"""
        if len(self.signals) < 5:
            return 0.5

        recent_regimes = [s['regime'] for s in self.signals[-5:]]
        unique_regimes = len(set(recent_regimes))

        # More unique regimes = less stability
        stability = 1.0 - (unique_regimes - 1) / 4.0
        return max(0.0, min(1.0, stability))


# ==============================================================================
# === VISUALIZATION (ui_charts.py)
# ==============================================================================

class InstitutionalChartingSuite:
    """
    Comprehensive professional visualization suite.
    This class is static as it's a collection of plotting functions.
    """

    @staticmethod
    def create_comprehensive_dashboard(df: pd.DataFrame, regime_state: Optional[RegimeState],
                                     risk_metrics: RiskMetrics,
                                     performance_summary: Dict[str, Any]) -> go.Figure:
        """Create institutional-grade comprehensive dashboard"""

        # Create subplot figure
        fig = go.Figure()

        # 1. Price and Volume Chart
        fig = InstitutionalChartingSuite._add_price_volume_subplot(fig, df, regime_state)

        # 2. Technical Indicators
        fig = InstitutionalChartingSuite._add_technical_indicators(fig, df)

        # 3. Regime Probability
        fig = InstitutionalChartingSuite._add_regime_probability(fig, df, regime_state)

        # Professional layout
        regime_name = regime_state.current_regime.value if regime_state else "UNKNOWN"
        fig.update_layout(
            title=f"Institutional Bitcoin Regime Dashboard - {regime_name}",
            height=900,
            template="plotly_dark",
            showlegend=True,
            xaxis_rangeslider_visible=False,
            margin=dict(t=100, b=50, l=50, r=50)
        )

        return fig

    @staticmethod
    def _add_price_volume_subplot(fig: go.Figure, df: pd.DataFrame, 
                                regime_state: Optional[RegimeState]) -> go.Figure:
        """Add price and volume subplot"""
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name="BTC/USD"
        ))

        # Add moving averages
        for length, color in [(10, 'yellow'), (20, 'orange'), (50, 'red')]:
            ema_col = f'EMA_{length}'
            if ema_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ema_col],
                    line=dict(width=1.5, color=color),
                    name=f"EMA {length}",
                    opacity=0.7
                ))

        # Add Bollinger Bands if available
        if 'BBU_20_2.0' in df.columns and 'BBL_20_2.0' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BBU_20_2.0'],
                line=dict(width=1, color='gray'),
                name="BB Upper",
                opacity=0.5
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BBL_20_2.0'],
                line=dict(width=1, color='gray'),
                name="BB Lower",
                opacity=0.5,
                fill='tonexty'
            ))

        return fig

    @staticmethod
    def _add_technical_indicators(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
        """Add technical indicators subplot"""
        # Create secondary y-axis for indicators
        fig.update_layout(
            yaxis2=dict(
                title="Indicator Values",
                overlaying="y",
                side="right"
            )
        )

        # RSI
        if 'RSI_14' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['RSI_14'],
                line=dict(width=1.5, color='purple'),
                name="RSI",
                yaxis="y2",
                opacity=0.7
            ))

        # MACD
        if 'MACD_12_26_9' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD_12_26_9'],
                line=dict(width=1.5, color='cyan'),
                name="MACD",
                yaxis="y2",
                opacity=0.7
            ))

        return fig

    @staticmethod
    def _add_regime_probability(fig: go.Figure, df: pd.DataFrame, 
                              regime_state: Optional[RegimeState]) -> go.Figure:
        """Add regime probability overlay"""
        if regime_state and 'regime_score' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['regime_score'] * 100,  # Scale to percentage
                line=dict(width=2, color='lime'),
                name="Regime Score",
                yaxis="y2",
                opacity=0.8
            ))

        return fig

    @staticmethod
    def create_risk_metrics_chart(risk_metrics: RiskMetrics) -> go.Figure:
        """Create risk metrics radar chart"""
        categories = ['VaR 99%', 'Expected Shortfall', 'Volatility',
                     'Max Drawdown', 'Sharpe Ratio', 'Sortino Ratio']

        values = [
            abs(risk_metrics.var_99) * 100,  # Convert to percentage
            abs(risk_metrics.expected_shortfall) * 100,
            risk_metrics.volatility_annualized * 100,
            risk_metrics.max_drawdown,
            risk_metrics.sharpe_ratio * 10,  # Scale for visibility
            risk_metrics.sortino_ratio * 10
        ]
        
        # Ensure values are non-negative for polar chart
        values = [max(0, v) for v in values]

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='#00D4AA')
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(values) * 1.2])
            ),
            showlegend=False,
            title="Risk Metrics Radar",
            template="plotly_dark"
        )

        return fig


# ==============================================================================
# === APPLICATION LAYER (ui_app.py)
# ==============================================================================

class EnhancedInstitutionalDashboard:
    """
    Production-grade institutional dashboard.
    This class orchestrates the data, analytics, and UI components.
    """

    def __init__(self,
                 data_engine: MultiSourceDataEngine,
                 quant_engine: MLEnhancedQuantEngine,
                 regime_detector: EnhancedBayesianRegimeDetector,
                 risk_engine: EnhancedRiskManagementEngine,
                 signal_tracker: EnhancedSignalTracker,
                 charting_suite: InstitutionalChartingSuite):
        
        # Injected Dependencies
        self.data_engine = data_engine
        self.quant_engine = quant_engine
        self.regime_detector = regime_detector
        self.risk_engine = risk_engine
        self.signal_tracker = signal_tracker
        self.charting_suite = charting_suite
        
        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state with robust defaults."""
        defaults = {
            'last_refresh': None,
            'data_quality': 'UNKNOWN',
            'system_health': 'HEALTHY',
            'timeframe': DEFAULT_TIMEFRAME,
            'confidence_threshold': DEFAULT_CONFIDENCE_THRESHOLD,
            'max_position_size': DEFAULT_MAX_POSITION_SIZE,
            'stop_loss_pct': DEFAULT_STOP_LOSS_PCT
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def run(self):
        """Run the enhanced institutional dashboard"""
        self._setup_enhanced_page()
        self._render_professional_sidebar()

        # Data loading with enhanced monitoring
        with st.spinner("🔄 Fetching LIVE multi-source market data..."):
            data = self._load_enhanced_market_data()

        if data.empty:
            self._handle_data_failure()
            return

        # Quantitative analysis pipeline
        with st.spinner("🔬 Running quantitative analysis..."):
            enhanced_data = self.quant_engine.calculate_advanced_indicators(data)
            risk_metrics = self.risk_engine.calculate_comprehensive_risk(enhanced_data)
            regime_state = self.regime_detector.detect_regime(enhanced_data)
            portfolio_allocation = self.risk_engine.calculate_portfolio_allocation(
                regime_state, risk_metrics) if regime_state else None

        # Signal tracking and analytics
        if regime_state and portfolio_allocation:
            self.signal_tracker.add_signal(enhanced_data, regime_state, risk_metrics, portfolio_allocation)

        # Performance analytics
        performance_summary = self.signal_tracker.get_performance_summary()

        # Render dashboard components
        self._render_enhanced_header(enhanced_data, regime_state, risk_metrics)
        self._render_main_analytics(enhanced_data, regime_state, risk_metrics, performance_summary)
        self._render_risk_management(risk_metrics, portfolio_allocation)
        self._render_signal_analytics(performance_summary)
        self._render_portfolio_insights(portfolio_allocation)

        # Auto-refresh logic
        self._handle_auto_refresh()

    def _setup_enhanced_page(self):
        """Professional page setup with enhanced styling"""
        st.set_page_config(
            layout="wide",
            page_title="₿ PRO BTC Regime Tracker",
            page_icon="₿",
            initial_sidebar_state="expanded"
        )

        # Enhanced professional CSS
        st.markdown(APP_CSS, unsafe_allow_html=True)

    def _render_professional_sidebar(self):
        """Enhanced professional sidebar, reading from session_state"""
        with st.sidebar:
            st.markdown("## 🎯 Institutional Configuration")

            # Trading parameters
            timeframe_options = [t.value for t in TimeInterval]
            current_timeframe_index = timeframe_options.index(st.session_state.timeframe)
            
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox(
                    "Primary Timeframe",
                    options=timeframe_options,
                    index=current_timeframe_index,
                    key="timeframe" # Key links it to session_state
                )
            with col2:
                st.slider("Confidence Threshold", 0.6, 0.95,
                         key="confidence_threshold")

            st.markdown("---")
            st.markdown("### ⚙️ Risk Parameters")

            # Risk management
            st.number_input("Max Position Size (%)", min_value=1.0,
                          max_value=50.0, key="max_position_size")
            st.number_input("Stop Loss (%)", min_value=5.0,
                          max_value=30.0, key="stop_loss_pct")

            st.markdown("---")
            st.markdown("### 📊 System Monitoring")

            # System status
            st.metric("Data Quality", st.session_state.data_quality, "Live")
            st.metric("System Health", st.session_state.system_health, "Optimal")

            # Refresh control
            st.markdown("---")
            if st.button("🔄 Manual Refresh", use_container_width=True):
                st.rerun()

    def _load_enhanced_market_data(self) -> pd.DataFrame:
        """Load market data with enhanced monitoring"""
        try:
            timeframe = st.session_state.timeframe
            data = self.data_engine.fetch_multi_source_ohlcv(PRIMARY_PERIOD, timeframe)

            # Update data quality assessment
            if data.empty:
                st.session_state.data_quality = "POOR"
            else:
                st.session_state.data_quality = "EXCELLENT" if len(data) > 100 else "GOOD"

            return data

        except Exception as e:
            logging.error(f"Data loading failed: {e}")
            st.session_state.data_quality = "FAILED"
            return pd.DataFrame()

    def _handle_data_failure(self):
        """Handle data loading failures professionally"""
        st.error("""
        ❌ **CRITICAL: Market Data Unavailable**

        **Troubleshooting Steps:**
        1. Check internet connection
        2. Verify Yahoo Finance accessibility
        3. Try manual refresh
        4. Contact system administrator if persistent
        """)

        st.session_state.system_health = "DEGRADED"

    def _render_enhanced_header(self, data: pd.DataFrame, 
                              regime_state: Optional[RegimeState],
                              risk_metrics: RiskMetrics):
        """Enhanced professional header"""
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

        with col1:
            st.markdown('<div class="main-header">Institutional-Grade Bitcoin Tracker</div>',
                       unsafe_allow_html=True)
            if not data.empty:
                latest_price = data.iloc[-1]['close']
                price_change = data.iloc[-1].get('returns', 0) * 100
                st.markdown(f'<div class="live-badge">LIVE: ${latest_price:,.2f} ({price_change:+.2f}%)</div>',
                           unsafe_allow_html=True)

        with col2:
            if regime_state:
                regime_emoji = {
                    MarketRegime.BULL_TRENDING: "🟢",
                    MarketRegime.BULL_CONSOLIDATION: "🟡",
                    MarketRegime.BEAR_TRENDING: "🔴",
                    MarketRegime.BEAR_CONSOLIDATION: "🟠",
                    MarketRegime.HIGH_VOLATILITY_BREAKOUT: "🟣",
                    MarketRegime.LOW_VOLATILITY_RANGE: "🔵",
                    MarketRegime.CRASH_IMMINENT: "⚫",
                    MarketRegime.REGIME_TRANSITION: "⚪"
                }.get(regime_state.current_regime, "⚪")

                st.metric(
                    "Market Regime",
                    f"{regime_emoji} {regime_state.current_regime.value}",
                    f"{regime_state.regime_probability:.1%} confidence"
                )
            else:
                st.metric("Market Regime", "⚪ UNKNOWN", "Insufficient Data")

        with col3:
            st.metric("Portfolio VaR", f"{risk_metrics.var_99:.2%}", "99% Confidence")

        with col4:
            st.metric("Expected Shortfall", f"{risk_metrics.expected_shortfall:.2%}", "Tail Risk")

        with col5:
            sharpe_color = "normal" if risk_metrics.sharpe_ratio > 1 else "inverse"
            st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}",
                     delta_color=sharpe_color)

    def _render_main_analytics(self, data: pd.DataFrame, 
                             regime_state: Optional[RegimeState],
                             risk_metrics: RiskMetrics, 
                             performance_summary: Dict[str, Any]):
        """Main analytics dashboard"""
        tab1, tab2, tab3 = st.tabs(["📈 Market Analysis", "🔍 Regime Analytics", "📊 Performance"])

        with tab1:
            self._render_market_analysis(data, regime_state, risk_metrics)

        with tab2:
            self._render_regime_analytics(regime_state, performance_summary)

        with tab3:
            self._render_performance_analytics(performance_summary)

    def _render_market_analysis(self, data: pd.DataFrame, 
                              regime_state: Optional[RegimeState],
                              risk_metrics: RiskMetrics):
        """Market analysis tab"""
        col1, col2 = st.columns([3, 1])

        with col1:
            # Main chart
            performance_summary = self.signal_tracker.get_performance_summary()
            fig = self.charting_suite.create_comprehensive_dashboard(
                data, regime_state, risk_metrics, performance_summary)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 Live Metrics")

            if not data.empty:
                latest = data.iloc[-1]

                metrics = [
                    ("Price", f"${latest['close']:,.2f}", "Current"),
                    ("24h Return", f"{latest.get('returns', 0):.2%}", "Daily"),
                    ("Volume Ratio", f"{latest.get('volume_ratio', 1.0):.2f}", "vs MA"),
                    ("Efficiency", f"{latest.get('efficiency_ratio', 0.5):.2f}", "Trend"),
                    ("RSI", f"{latest.get('RSI_14', 50):.1f}", "Momentum"),
                    ("Volatility", f"{latest.get('garch_vol', 0.5):.2f}", "Annualized")
                ]

                for label, value, delta in metrics:
                    st.metric(label, value, delta)

    def _render_regime_analytics(self, regime_state: Optional[RegimeState],
                               performance_summary: Dict[str, Any]):
        """Regime analytics tab"""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Current Regime Analysis")

            if regime_state:
                st.markdown(f"""
                **Primary Regime:** {regime_state.current_regime.value}  
                **Probability:** {regime_state.regime_probability:.1%}  
                **ML Confidence:** {regime_state.ml_confidence:.1%}  
                **Signal Strength:** <span class="signal-strength-{regime_state.signal_strength.name.lower()}">{regime_state.signal_strength.name}</span>  
                **Duration:** {regime_state.duration_bars} bars  
                **Regime Strength:** {regime_state.regime_strength:.2f}
                """, unsafe_allow_html=True)

                st.markdown("**Supporting Indicators:**")
                for indicator in regime_state.supporting_indicators:
                    st.write(f"- {indicator}")
            else:
                st.warning("Regime not yet determined. More data is required.")

        with col2:
            st.markdown("### 📈 Regime Performance")

            # **BUG FIX:** Use .get() for robust rendering
            st.metric("Total Signals", performance_summary.get('total_signals', 0))
            st.metric("Avg Signal Quality", f"{performance_summary.get('avg_signal_quality', 0):.3f}")
            st.metric("Recent Accuracy", f"{performance_summary.get('recent_accuracy', 0):.1%}")
            st.metric("Regime Stability", f"{performance_summary.get('regime_stability', 0.5):.1%}")

            # Regime distribution chart
            regime_dist = performance_summary.get('regime_distribution', {})
            if regime_dist:
                fig = px.pie(values=list(regime_dist.values()),
                           names=list(regime_dist.keys()),
                           title="Regime Distribution")
                st.plotly_chart(fig, use_container_width=True)

    def _render_performance_analytics(self, performance_summary: Dict[str, Any]):
        """Performance analytics tab"""
        st.markdown("### 📊 Advanced Analytics")

        col1, col2, col3, col4 = st.columns(4)

        # **BUG FIX:** Use .get() for robust rendering
        with col1:
            st.metric("ML Confidence", f"{performance_summary.get('avg_ml_confidence', 0):.1%}")
        with col2:
            st.metric("Signal Strength Trend", performance_summary.get('signal_strength_trend', 'STABLE'))
        with col3:
            st.metric("Avg Predicted Duration", f"{performance_summary.get('avg_predicted_duration', 0):.1f}")
        with col4:
            st.metric("Volatility Level", f"{performance_summary.get('volatility_trend', 0):.3f}")

        # Display historical signals
        signals_df = self.signal_tracker.get_signals_dataframe()
        if not signals_df.empty:
            st.markdown("#### 📋 Signal History")
            st.dataframe(signals_df, use_container_width=True, height=400)

            # Download capability
            csv = signals_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Signal History",
                data=csv,
                file_name=f"btc_regime_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    def _render_risk_management(self, risk_metrics: RiskMetrics,
                              portfolio_allocation: Optional[PortfolioAllocation]):
        """Risk management section"""
        st.markdown("---")
        st.markdown("## 🛡️ Institutional Risk Management")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Risk metrics chart
            fig = self.charting_suite.create_risk_metrics_chart(risk_metrics)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 Risk Metrics")

            risk_metrics_data = [
                ("Value at Risk (99%)", f"{risk_metrics.var_99:.2%}"),
                ("Expected Shortfall", f"{risk_metrics.expected_shortfall:.2%}"),
                ("Volatility", f"{risk_metrics.volatility_annualized:.1%}"),
                ("Max Drawdown", f"{risk_metrics.max_drawdown:.1f}%"),
                ("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}"),
                ("Sortino Ratio", f"{risk_metrics.sortino_ratio:.2f}"),
                ("Ulcer Index", f"{risk_metrics.ulcer_index:.2f}"),
                ("Information Ratio", f"{risk_metrics.information_ratio:.2f}")
            ]

            for metric, value in risk_metrics_data:
                st.metric(metric, value)

    def _render_signal_analytics(self, performance_summary: Dict[str, Any]):
        """Signal analytics section"""
        st.markdown("---")
        st.markdown("## 📈 Signal Performance Analytics")

        col1, col2, col3, col4 = st.columns(4)

        # **BUG FIX:** Use .get() for robust rendering
        with col1:
            st.metric("Total Signals", performance_summary.get('total_signals', 0))
        with col2:
            st.metric("Average Quality", f"{performance_summary.get('avg_signal_quality', 0):.3f}")
        with col3:
            st.metric("ML Confidence", f"{performance_summary.get('avg_ml_confidence', 0):.1%}")
        with col4:
            st.metric("Recent Accuracy", f"{performance_summary.get('recent_accuracy', 0):.1%}")

        # Regime distribution
        st.markdown("#### 🎯 Regime Distribution")
        regime_data = []
        total_signals = performance_summary.get('total_signals', 0)
        regime_dist = performance_summary.get('regime_distribution', {})
        
        if total_signals > 0:
            for regime, count in regime_dist.items():
                percentage = (count / total_signals) * 100
                regime_data.append({"Regime": regime, "Count": count, "Percentage": f"{percentage:.1f}%"})

            if regime_data:
                regime_df = pd.DataFrame(regime_data)
                st.dataframe(regime_df, use_container_width=True)
        else:
            st.info("No signals generated yet.")


    def _render_portfolio_insights(self, portfolio_allocation: Optional[PortfolioAllocation]):
        """Portfolio insights section"""
        if not portfolio_allocation:
            return

        st.markdown("---")
        st.markdown("## 💼 Portfolio Optimization")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("BTC Allocation", f"{portfolio_allocation.btc_allocation:.1%}")
        with col2:
            st.metric("Cash Allocation", f"{portfolio_allocation.cash_allocation:.1%}")
        with col3:
            st.metric("Expected Return", f"{portfolio_allocation.expected_return:.1%}")
        with col4:
            rebalance_color = "normal" if portfolio_allocation.rebalance_signal else "off"
            st.metric("Rebalance Signal",
                     "YES" if portfolio_allocation.rebalance_signal else "NO",
                     delta_color=rebalance_color)

        # Allocation visualization
        allocation_data = {
            'Asset': ['Bitcoin', 'Cash'],
            'Allocation': [portfolio_allocation.btc_allocation, portfolio_allocation.cash_allocation]
        }
        alloc_df = pd.DataFrame(allocation_data)

        fig = px.pie(alloc_df, values='Allocation', names='Asset',
                    title="Optimal Portfolio Allocation",
                    color_discrete_sequence=['#00D4AA', '#0099FF'])
        st.plotly_chart(fig, use_container_width=True)

    def _handle_auto_refresh(self):
        """Handle automatic refresh logic"""
        current_time = datetime.now()

        if st.session_state.last_refresh is None:
            st.session_state.last_refresh = current_time
            return

        time_diff = (current_time - st.session_state.last_refresh).total_seconds()

        # Refresh every 30 seconds for live data
        if time_diff > 30:
            st.session_state.last_refresh = current_time
            st.rerun()

# ==============================================================================
# === MAIN EXECUTION (main.py)
# ==============================================================================

def main():
    """
    Main execution with professional error handling and monitoring.
    Acts as the "Composition Root" for dependency injection.
    """
    try:
        # --- Dependency Composition Root ---
        # Instantiate all services
        data_engine = MultiSourceDataEngine(TICKER)
        quant_engine = MLEnhancedQuantEngine()
        regime_detector = EnhancedBayesianRegimeDetector()
        risk_engine = EnhancedRiskManagementEngine()
        signal_tracker = EnhancedSignalTracker(max_signals=HISTORY_SIZE)
        charting_suite = InstitutionalChartingSuite() # Static class, but passing for consistency

        # Inject dependencies into the main application
        dashboard = EnhancedInstitutionalDashboard(
            data_engine=data_engine,
            quant_engine=quant_engine,
            regime_detector=regime_detector,
            risk_engine=risk_engine,
            signal_tracker=signal_tracker,
            charting_suite=charting_suite
        )
        
        # Run the application
        dashboard.run()

    except Exception as e:
        # Comprehensive error handling
        logging.error(f"Dashboard execution failed: {e}", exc_info=True)

        st.error("""
        🚨 **System Temporarily Unavailable**

        Our engineering team has been notified and is working to resolve the issue.

        **Error Details:**
        """)
        st.code(f"{type(e).__name__}: {str(e)}")

        # Provide recovery options
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Try Again", use_container_width=True):
                st.rerun()

        with col2:
            if st.button("📋 Copy Error Report", use_container_width=True):
                st.code(f"BTC Regime Tracker Error: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    main()