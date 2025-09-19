import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Constants
RISK_FREE_RATE = 0.06  # 6% annual risk-free rate
TRADING_DAYS = 252  # Number of trading days in a year

class FundAnalyzer:
    def __init__(self, mf_file, nifty_file):
        self.mf_file = mf_file
        self.nifty_file = nifty_file
        self.mf_data = None
        self.nifty_data = None
        self.returns_data = None  # mutual fund returns only (DataFrame)
        self.nifty_returns = None  # NIFTY returns only (Series)
        self.risk_free_rate_daily = (1 + RISK_FREE_RATE) ** (1/TRADING_DAYS) - 1
        
    def load_data(self):
        """Load and preprocess mutual fund and NIFTY 100 data."""
        # Load mutual fund data
        mf_df = pd.read_csv(self.mf_file)
        # Robust date parsing for MF data: try both dayfirst True/False and pick the best
        mf_dates1 = pd.to_datetime(mf_df['Date'], dayfirst=True, errors='coerce')
        mf_dates2 = pd.to_datetime(mf_df['Date'], dayfirst=False, errors='coerce')
        def span_and_unique(s):
            s2 = s.dropna()
            return (s2.max() - s2.min()).days if len(s2) else -1, s2.nunique()
        span1, uniq1 = span_and_unique(mf_dates1)
        span2, uniq2 = span_and_unique(mf_dates2)
        if (span2, uniq2) > (span1, uniq1):
            mf_df['Date'] = mf_dates2
        else:
            mf_df['Date'] = mf_dates1
        mf_df = mf_df.dropna(subset=['Date'])
        mf_df['Date'] = mf_df['Date'].dt.tz_localize(None)
        mf_df = mf_df.sort_values('Date')
        # Ensure NAV is numeric
        mf_df['NAV'] = pd.to_numeric(mf_df['NAV'], errors='coerce')
        mf_df = mf_df.dropna(subset=['NAV'])
        
        # Load NIFTY 100 data
        nifty_df = pd.read_csv(self.nifty_file)
        # Robust date parsing for NIFTY data: try both
        n_dates1 = pd.to_datetime(nifty_df['Date'], dayfirst=True, errors='coerce')
        n_dates2 = pd.to_datetime(nifty_df['Date'], dayfirst=False, errors='coerce')
        n_span1, n_uniq1 = span_and_unique(n_dates1)
        n_span2, n_uniq2 = span_and_unique(n_dates2)
        if (n_span2, n_uniq2) > (n_span1, n_uniq1):
            nifty_df['Date'] = n_dates2
        else:
            nifty_df['Date'] = n_dates1
        nifty_df = nifty_df.dropna(subset=['Date'])
        nifty_df['Date'] = nifty_df['Date'].dt.tz_localize(None)
        nifty_df = nifty_df[['Date', 'Close']].rename(columns={'Close': 'NIFTY100'})
        nifty_df = nifty_df.sort_values('Date').drop_duplicates('Date')
        
        # Pivot MF data to get NAVs by date and scheme
        mf_navs = mf_df.pivot(index='Date', columns='Scheme Name', values='NAV')
        mf_navs = mf_navs[~mf_navs.index.duplicated(keep='last')]
        mf_navs = mf_navs.sort_index()
        
        # Ensure both indices are timezone-naive
        mf_navs.index = mf_navs.index.tz_localize(None)
        nifty_idx = nifty_df.set_index('Date').index.tz_localize(None)
        
        # Calculate returns separately (do not join)
        mf_returns = mf_navs.sort_index().pct_change()
        nifty_series = nifty_df.set_index('Date')['NIFTY100'].sort_index().pct_change()
        
        self.mf_data = mf_df
        self.nifty_data = nifty_df
        self.returns_data = mf_returns
        self.nifty_returns = nifty_series
        
        return mf_returns
    
    def calculate_quarterly_outperformance(self, returns, nifty_series: pd.Series):
        """Calculate how many quarters each fund has beaten the market (NIFTY100)."""
        # Use full history, compute quarters separately and compare to NIFTY quarterly
        quarterly_returns = (1 + returns).resample('Q').prod() - 1
        nifty_quarterly = (1 + nifty_series).resample('Q').prod() - 1
        
        # Calculate number of quarters each fund beat the market (robustly)
        fund_cols = list(returns.columns)
        if len(quarterly_returns) > 0 and len(nifty_quarterly) > 0:
            aligned = quarterly_returns.join(nifty_quarterly.rename('NIFTY100'), how='inner')
            compare_df = aligned[fund_cols].gt(aligned['NIFTY100'], axis=0)
            outperformance = compare_df.sum(axis=0)
            total_quarters = int(len(aligned.index))
        else:
            outperformance = pd.Series(0, index=fund_cols)
            total_quarters = 0
        
        # Convert to percentage of quarters beaten
        outperformance_pct = (outperformance / total_quarters * 100) if total_quarters > 0 else outperformance.astype(float)
        
        return outperformance, total_quarters, outperformance_pct
        
    def calculate_metrics(self, returns, nifty_series: pd.Series):
        """Calculate various performance metrics for each fund."""
        metrics = {}
        
        # Calculate quarterly outperformance
        outperformance, total_quarters, outperformance_pct = self.calculate_quarterly_outperformance(returns, nifty_series)
        
        for fund in returns.columns:
            fund_returns = returns[fund].dropna()
            nifty_returns = nifty_series.dropna()
            
            # Basic metrics
            total_return = (1 + fund_returns).prod() - 1 if len(fund_returns) else np.nan
            # Annualized return (internal for Sharpe), use average daily return
            mean_daily = fund_returns.mean() if len(fund_returns) else np.nan
            annual_return = ((1 + mean_daily) ** TRADING_DAYS - 1) if pd.notna(mean_daily) else np.nan
            volatility = (fund_returns.std() * np.sqrt(TRADING_DAYS)) if len(fund_returns) else np.nan
            
            # Weekly and Monthly average returns
            weekly_ret_series = (1 + fund_returns).resample('W-FRI').prod() - 1 if len(fund_returns) else pd.Series(dtype=float)
            monthly_ret_series = (1 + fund_returns).resample('M').prod() - 1 if len(fund_returns) else pd.Series(dtype=float)
            weekly_return = weekly_ret_series.mean() if len(weekly_ret_series) > 0 else np.nan
            monthly_return = monthly_ret_series.mean() if len(monthly_ret_series) > 0 else np.nan
            
            # 2Y CAGR over last 2 calendar years using actual time span
            end_date = returns.index.max()
            start_date = end_date - pd.DateOffset(years=2)
            fund_slice = fund_returns.loc[fund_returns.index >= start_date]
            if len(fund_slice) > 1:
                growth = (1 + fund_slice).prod()
                years = (fund_slice.index.max() - fund_slice.index.min()).days / 365.25
                # Require at least 0.5 years of data to report 2Y-window CAGR; otherwise set NaN
                if years >= 0.5 and growth > 0:
                    cagr_2y = growth ** (1/years) - 1
                else:
                    cagr_2y = np.nan
            else:
                cagr_2y = np.nan

            # CAGR over available full history (fallback)
            if len(fund_returns) > 1:
                growth_all = (1 + fund_returns).prod()
                years_all = (fund_returns.index.max() - fund_returns.index.min()).days / 365.25
                if years_all > 0 and growth_all > 0:
                    cagr_available = growth_all ** (1/years_all) - 1
                else:
                    cagr_available = np.nan
            else:
                cagr_available = np.nan

            # As a final fallback, use annualized mean daily return if available CAGR is NaN
            if pd.isna(cagr_available) and pd.notna(mean_daily):
                cagr_available = (1 + mean_daily) ** TRADING_DAYS - 1
            
            # Risk-adjusted metrics
            sharpe_ratio = (annual_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
            
            # Sortino ratio (only downside deviation)
            downside_returns = fund_returns[fund_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(TRADING_DAYS) if len(downside_returns) > 0 else 0
            sortino_ratio = (annual_return - RISK_FREE_RATE) / downside_volatility if downside_volatility > 0 else 0
            
            # Beta and Alpha computed on overlapping non-NaN dates only
            if len(nifty_returns) > 1 and len(fund_returns) > 1:
                aligned = pd.concat([fund_returns, nifty_returns], axis=1, join='inner').dropna()
                if len(aligned) > 1 and aligned.iloc[:,1].var() > 0:
                    cov_matrix = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])
                    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                    market_ann = aligned.iloc[:,1].mean() * TRADING_DAYS
                    alpha = (annual_return if pd.notna(annual_return) else 0) - (RISK_FREE_RATE + beta * (market_ann - RISK_FREE_RATE))
                else:
                    beta = np.nan
                    alpha = np.nan
            else:
                beta = np.nan
                alpha = np.nan
            
            # Maximum Drawdown
            cum_returns = (1 + fund_slice if 'fund_slice' in locals() and len(fund_slice)>0 else (1 + fund_returns)).cumprod()
            rolling_max = cum_returns.cummax()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Store metrics
            metrics[fund] = {
                'Total Return': total_return * 100,  # as percentage
                'Weekly Return': weekly_return * 100 if pd.notnull(weekly_return) else np.nan,  # percentage
                'Monthly Return': monthly_return * 100 if pd.notnull(monthly_return) else np.nan,  # percentage
                'CAGR 2Y': cagr_2y * 100 if pd.notnull(cagr_2y) else np.nan,  # percentage
                'CAGR Available': cagr_available * 100 if pd.notnull(cagr_available) else np.nan,  # percentage
                'Volatility': volatility * 100,  # as percentage
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Beta': beta,
                'Alpha': alpha * 100,  # as percentage
                'Max Drawdown': max_drawdown * 100,  # as percentage
                'Downside Volatility': downside_volatility * 100,  # as percentage
                'Quarters Beat Market': outperformance.get(fund, 0),
                'Total Quarters': total_quarters,
                'Market Outperformance %': outperformance_pct.get(fund, 0)
            }
            
        # Also compute and append NIFTY100 metrics independently
        if nifty_series is not None and len(nifty_series.dropna()) > 1:
            nret = nifty_series.dropna()
            total_return = (1 + nret).prod() - 1
            mean_daily = nret.mean()
            annual_return = (1 + mean_daily) ** TRADING_DAYS - 1
            volatility = nret.std() * np.sqrt(TRADING_DAYS)
            weekly_ret_series = (1 + nret).resample('W-FRI').prod() - 1
            monthly_ret_series = (1 + nret).resample('M').prod() - 1
            weekly_return = weekly_ret_series.mean() if len(weekly_ret_series) > 0 else np.nan
            monthly_return = monthly_ret_series.mean() if len(monthly_ret_series) > 0 else np.nan
            end_date = nret.index.max()
            start_date = end_date - pd.DateOffset(years=2)
            n_slice = nret.loc[nret.index >= start_date]
            if len(n_slice) > 1:
                growth = (1 + n_slice).prod()
                years = (n_slice.index.max() - n_slice.index.min()).days / 365.25
                cagr_2y = growth ** (1/years) - 1 if years >= 0.5 and growth > 0 else np.nan
            else:
                cagr_2y = np.nan
            # Available period CAGR
            growth_all = (1 + nret).prod()
            years_all = (nret.index.max() - nret.index.min()).days / 365.25
            cagr_available = growth_all ** (1/years_all) - 1 if years_all > 0 else np.nan
            if pd.isna(cagr_available) and pd.notna(mean_daily):
                cagr_available = (1 + mean_daily) ** TRADING_DAYS - 1
            downside_returns = nret[nret < 0]
            downside_volatility = downside_returns.std() * np.sqrt(TRADING_DAYS) if len(downside_returns) > 0 else 0
            sharpe_ratio = (annual_return - RISK_FREE_RATE) / volatility if volatility and volatility > 0 else 0
            sortino_ratio = (annual_return - RISK_FREE_RATE) / downside_volatility if downside_volatility and downside_volatility > 0 else 0
            # Beta and Alpha for NIFTY are 1 and 0 by definition relative to itself
            beta = 1.0
            alpha = 0.0
            cum_returns = (1 + nret).cumprod()
            rolling_max = cum_returns.cummax()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            metrics['NIFTY100'] = {
                'Total Return': total_return * 100,
                'Weekly Return': weekly_return * 100 if pd.notnull(weekly_return) else np.nan,
                'Monthly Return': monthly_return * 100 if pd.notnull(monthly_return) else np.nan,
                'CAGR 2Y': cagr_2y * 100 if pd.notnull(cagr_2y) else np.nan,
                'CAGR Available': cagr_available * 100 if pd.notnull(cagr_available) else np.nan,
                'Volatility': volatility * 100,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Beta': beta,
                'Alpha': alpha * 100,
                'Max Drawdown': max_drawdown * 100,
                'Downside Volatility': downside_volatility * 100,
                'Quarters Beat Market': np.nan,
                'Total Quarters': total_quarters,
                'Market Outperformance %': np.nan
            }

        return pd.DataFrame(metrics).T

        

    def compute_window_metrics(self, returns: pd.DataFrame, window_days: int) -> pd.DataFrame:
        """Compute non-annualized window return and volatility over the given trailing window.
        - Return = cumulative return over window (product of daily returns) - 1
        - Volatility = standard deviation of daily returns within window (not annualized)
        Returns a DataFrame with columns: ['Return_window', 'Volatility_window'] in percent.
        """
        # Use available rows if fewer than requested; require a very small minimum only
        if len(returns) < 5:
            return pd.DataFrame()
        use_rows = min(window_days, len(returns))
        window = returns.tail(use_rows)
        # Compute per-series metrics with NaNs dropped within the window
        vol = window.apply(lambda s: s.dropna().std())
        growth = window.apply(lambda s: (1 + s.dropna()).prod() - 1)
        df = pd.DataFrame({'Return_window': growth * 100, 'Volatility_window': vol * 100})
        return df

    def compute_max_drawdown(self, returns: pd.DataFrame) -> pd.Series:
        """Compute Max Drawdown (%) for each column of daily returns.
        Uses cumulative product of (1+r), then rolling max, then (cum/rollmax - 1)."""
        if returns is None or returns.empty:
            return pd.Series(dtype=float)
        def mdd(series: pd.Series) -> float:
            # Clean and sanitize returns
            s = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) == 0:
                return np.nan
            # If data seems to be in percent units (e.g., 2 -> 200%), convert to fraction
            try:
                if s.abs().median() > 1.5:
                    s = s / 100.0
            except Exception:
                pass
            # Cap extreme negatives to avoid cumprod flipping sign due to bad data
            s = s.clip(lower=-0.99)
            if len(s) == 0:
                return np.nan
            cum = (1 + s).cumprod()
            roll_max = cum.cummax()
            dd = (cum / roll_max - 1).min()
            return dd * 100.0  # percent
        return returns.apply(mdd, axis=0)

    def compute_volatility(self, returns: pd.DataFrame, annualized: bool = True) -> pd.Series:
        """Compute volatility (%) for each column of daily returns.
        If annualized is True, std * sqrt(252), else raw std of daily returns."""
        if returns is None or returns.empty:
            return pd.Series(dtype=float)
        std = returns.std(skipna=True)
        if annualized:
            std = std * np.sqrt(TRADING_DAYS)
        return std * 100.0

    def compute_downside_volatility(self, returns: pd.DataFrame, annualized: bool = True) -> pd.Series:
        """Compute downside volatility (%) for each column of daily returns.
        Std of negative daily returns; annualized optionally by sqrt(252)."""
        if returns is None or returns.empty:
            return pd.Series(dtype=float)
        def dsv(s: pd.Series) -> float:
            neg = s[s < 0].dropna()
            if len(neg) == 0:
                return 0.0
            val = neg.std()
            return float(val)
        ds = returns.apply(dsv, axis=0)
        if annualized:
            ds = ds * np.sqrt(TRADING_DAYS)
        return ds * 100.0

    def plot_risk_return_window(self, returns: pd.DataFrame, window_days: int, title_suffix: str) -> go.Figure:
        """Plot Risk (Volatility) vs Return (CAGR) for a selected trailing window.
        Applies jitter, WebGL rendering, and opacity to reduce overlap.
        """
        metrics_w = self.compute_window_metrics(returns, window_days)
        if metrics_w.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No sufficient data for selected window ({title_suffix})")
            return fig

    def compute_composite_score(self, metrics: pd.DataFrame, weights: dict, exclude_nifty: bool = True) -> pd.DataFrame:
        """Compute a composite score (0-1) per fund using percentile ranks of multiple metrics.
        weights: dict mapping metric name -> weight (non-negative). Higher is better metrics will be ranked ascending=False; lower is better metrics ascending=True.
        """
        if metrics is None or metrics.empty:
            return pd.DataFrame()
        df = metrics.copy()
        if exclude_nifty and 'NIFTY100' in df.index:
            df = df.drop(index='NIFTY100')
        # Define direction for metrics
        higher_better = {
            'Annualized Return': True,
            'CAGR Available': True,
            'CAGR 2Y': True,
            'Sharpe Ratio': True,
            'Sortino Ratio': True,
            'Market Outperformance %': True,
            'Quarters Beat Market': True
        }
        lower_better = {
            'Volatility': True,
            'Downside Volatility': True,
            'Max Drawdown': True,
            'Beta': True  # closer to 1 may be desired, but we treat lower beta as less market risk by default
        }
        # Build rank columns
        rank_cols = {}
        total_w = 0.0
        for m, w in weights.items():
            if w is None or w <= 0 or m not in df.columns:
                continue
            total_w += w
            ser = df[m]
            if m in lower_better:
                rank = ser.rank(pct=True, ascending=True)
            else:
                rank = ser.rank(pct=True, ascending=False)
            rank_cols[m] = rank
        if total_w == 0 or not rank_cols:
            df['Composite Score'] = np.nan
            return df
        score = None
        for m, rank in rank_cols.items():
            w = weights[m]
            part = rank.fillna(0) * w
            score = part if score is None else score.add(part, fill_value=0)
        df['Composite Score'] = score / total_w
        return df.sort_values('Composite Score', ascending=False)

    def compute_pairwise_wins(self, metrics: pd.DataFrame, weights: dict, exclude_nifty: bool = True) -> pd.DataFrame:
        """Rank funds using pairwise comparison across metrics.
        For each metric with positive weight, compare every pair of funds:
        - If metric is higher-better: fund A wins if A_m > B_m
        - If lower-better: fund A wins if A_m < B_m
        - Ties: split points (0.5 each) if both values are equal and not NaN
        Skip comparison for a metric if either fund's value is NaN.
        Sum wins across metrics using weights (wins count multiplied by metric weight).
        Normalize by total_possible = weight_sum * (N-1) for each fund to produce a 0-1 score.
        """
        if metrics is None or metrics.empty:
            return pd.DataFrame()
        df = metrics.copy()
        if exclude_nifty and 'NIFTY100' in df.index:
            df = df.drop(index='NIFTY100')
        funds = df.index.tolist()
        if len(funds) < 2:
            return df.assign(PairwiseScore=np.nan)

        # Directions
        higher_better = {'Annualized Return','CAGR Available','CAGR 2Y','Sharpe Ratio','Sortino Ratio','Market Outperformance %','Quarters Beat Market'}
        lower_better = {'Volatility','Downside Volatility','Max Drawdown','Beta'}

        # Keep only metrics present and with positive weights
        use_metrics = [(m, w) for m, w in weights.items() if w is not None and w > 0 and m in df.columns]
        if not use_metrics:
            return df.assign(PairwiseScore=np.nan)

        wins = pd.Series(0.0, index=funds)
        weight_sum = sum(w for _, w in use_metrics)
        # Pairwise loop
        for i, a in enumerate(funds):
            for j, b in enumerate(funds):
                if j == i:
                    continue
                score_a = 0.0
                for m, w in use_metrics:
                    va = df.at[a, m]
                    vb = df.at[b, m]
                    if pd.isna(va) or pd.isna(vb):
                        continue
                    if m in lower_better:
                        if va < vb:
                            score_a += w
                        elif va == vb:
                            score_a += w * 0.5
                    else:  # higher is better
                        if va > vb:
                            score_a += w
                        elif va == vb:
                            score_a += w * 0.5
                wins[a] += score_a

        # Normalize per fund by maximum possible points vs others
        max_points_per_opponent = weight_sum
        denom = max((len(funds) - 1) * max_points_per_opponent, 1e-9)
        pair_score = wins / denom
        out = df.copy()
        out['Pairwise Score'] = pair_score
        return out.sort_values('Pairwise Score', ascending=False)

    def compute_annualized_metrics(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute annualized return as CAGR over the effective window and annualized volatility.
        - Annualized Return (CAGR-window): growth = prod(1+r) - 1; n = non-NaN trading days; (1+growth) ** (252/n) - 1
        - Annualized Volatility = std_daily * sqrt(252)
        Returns a DataFrame with columns ['Annualized Return', 'Annualized Volatility'] in percent.
        """
        if returns is None or returns.empty:
            return pd.DataFrame()
        def cagr_from_series(s: pd.Series) -> float:
            s2 = s.dropna()
            n = len(s2)
            if n == 0:
                return np.nan
            growth = (1.0 + s2).prod() - 1.0
            try:
                ann = (1.0 + growth) ** (TRADING_DAYS / n) - 1.0
            except Exception:
                ann = np.nan
            return ann
        ann_return = returns.apply(cagr_from_series, axis=0)
        std_daily = returns.std(skipna=True)
        ann_vol = std_daily * np.sqrt(TRADING_DAYS)
        df = pd.DataFrame({
            'Annualized Return': ann_return * 100,
            'Annualized Volatility': ann_vol * 100
        })
        return df

    def plot_annualized_risk_return(self, returns: pd.DataFrame, nifty_series: pd.Series) -> go.Figure:
        """Plot annualized return vs annualized volatility for all funds; add NIFTY point if available."""
        ann_df = self.compute_annualized_metrics(returns)
        if ann_df.empty:
            fig = go.Figure()
            fig.update_layout(title="Annualized Risk-Return (no data)")
            return fig
        plot_df = ann_df.reset_index()
        if 'index' in plot_df.columns:
            plot_df = plot_df.rename(columns={'index': 'Fund'})
        elif 'Scheme Name' in plot_df.columns:
            plot_df = plot_df.rename(columns={'Scheme Name': 'Fund'})
        elif 'Fund' not in plot_df.columns:
            plot_df['Fund'] = ann_df.index.astype(str)
        plot_df['Vol_frac'] = plot_df['Annualized Volatility'] / 100.0
        plot_df['Ret_frac'] = plot_df['Annualized Return'] / 100.0
        # Scale marker size by absolute annualized return
        plot_df['Size'] = plot_df['Annualized Return'].abs().clip(upper=100) + 5
        fig = px.scatter(
            plot_df,
            x='Vol_frac',
            y='Ret_frac',
            hover_name='Fund',
            size='Size',
            size_max=18,
            title='Annualized (CAGR) Risk vs Return (Funds)',
            labels={'Vol_frac': 'Annualized Volatility', 'Ret_frac': 'Annualized Return (CAGR)'}
        )
        # Add NIFTY point if available
        if nifty_series is not None and len(nifty_series.dropna()) > 1:
            n = nifty_series.dropna()
            n_mean = n.mean()
            n_std = n.std()
            n_ret = (1 + n_mean) ** TRADING_DAYS - 1
            n_vol = n_std * np.sqrt(TRADING_DAYS)
            fig.add_trace(
                go.Scatter(
                    x=[n_vol],
                    y=[n_ret],
                    mode='markers',
                    marker=dict(size=18, symbol='star', color='red', line=dict(width=1, color='black')),
                    name='NIFTY 100',
                    hovertemplate='NIFTY 100<br>Ann Return: %{y:.2%}<br>Ann Vol: %{x:.2%}<extra></extra>'
                )
            )
        fig.update_layout(xaxis=dict(tickformat='.1%'), yaxis=dict(tickformat='.1%'))
        return fig

    def compute_annualized_metrics(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute annualized mean return and annualized volatility from daily returns.
        - Annualized Return = (1 + mean_daily) ** TRADING_DAYS - 1
        - Annualized Volatility = std_daily * sqrt(TRADING_DAYS)
        Returns a DataFrame with columns ['Annualized Return', 'Annualized Volatility'] in percent.
        """
        if returns is None or returns.empty:
            return pd.DataFrame()
        mean_daily = returns.mean(skipna=True)
        std_daily = returns.std(skipna=True)
        ann_return = (1 + mean_daily) ** TRADING_DAYS - 1
        ann_vol = std_daily * np.sqrt(TRADING_DAYS)
        df = pd.DataFrame({
            'Annualized Return': ann_return * 100,
            'Annualized Volatility': ann_vol * 100
        })
        return df

    
    def plot_returns_heatmap(self, metrics):
        """Create a heatmap of returns by period."""
        # Select top 10 funds by available-period CAGR
        top_funds = metrics.nlargest(10, 'CAGR Available').index.tolist()
        # Start with MF returns only
        returns_data = self.returns_data[top_funds].copy()
        # Add NIFTY as a separate column if available
        if getattr(self, 'nifty_returns', None) is not None and len(self.nifty_returns) > 0:
            returns_data = returns_data.join(self.nifty_returns.rename('NIFTY100'), how='left')
        
        # Calculate returns for different periods
        periods = {
            '1M': 21,    # 1 month
            '3M': 63,    # 3 months
            '6M': 126,   # 6 months
            '1Y': 252,   # 1 year
            '3Y': 756    # 3 years
        }
        
        period_returns = []
        for fund in returns_data.columns:
            for period_name, days in periods.items():
                if len(returns_data) >= days:
                    period_return = (1 + returns_data[fund].tail(days)).prod() - 1
                    period_returns.append({
                        'Fund': fund,
                        'Period': period_name,
                        'Return': period_return
                    })
        
        df = pd.DataFrame(period_returns)
        heatmap_data = df.pivot(index='Fund', columns='Period', values='Return')
        
        fig = px.imshow(
            heatmap_data * 100,
            labels=dict(x="Period", y="Fund", color="Return (%)"),
            title="Returns by Period (%)",
            color_continuous_scale='RdYlGn',
            aspect="auto"
        )
        
        return fig
    
    def plot_risk_return(self, metrics, use_available: bool = False):
        """Create a risk-return scatter plot.
        If use_available is True, use 'CAGR Available' for Y-axis; otherwise prefer 'CAGR 2Y' with fallback to available.
        """
        # Create a copy of metrics and ensure size is positive and properly scaled
        plot_data = metrics.reset_index().copy()
        # Convert percent columns to fractions for plotting with percent tickformat
        plot_data['Volatility_frac'] = plot_data['Volatility'] / 100.0
        # Choose source for CAGR axis
        if use_available:
            plot_data['CAGR_plot'] = plot_data['CAGR Available']
            y_label = 'CAGR (Available)'
            title = 'Risk vs CAGR (Available Period)'
        else:
            # Prefer 2Y window; fallback to available if missing
            plot_data['CAGR_plot'] = plot_data['CAGR 2Y'].fillna(plot_data['CAGR Available'])
            y_label = 'CAGR (2Y or Available)'
            title = 'Risk vs 2Y CAGR'
        plot_data['CAGR_frac'] = plot_data['CAGR_plot'] / 100.0
        # Total Return is stored as percent; clip and shift to ensure positive marker sizes
        plot_data['Size'] = plot_data['Total Return'].fillna(0).clip(lower=-90) + 100
        
        fig = px.scatter(
            plot_data,
            x='Volatility_frac',
            y='CAGR_frac',
            hover_name='index',
            size='Size',
            size_max=30,
            color='Sharpe Ratio',
            title=title,
            labels={'CAGR_frac': y_label, 'Volatility_frac': 'Annual Volatility', 'Size': 'Total Return (scaled)'}
        )
        
        if 'NIFTY100' in metrics.index:
            nifty_metrics = metrics.loc['NIFTY100']
            fig.add_trace(
                go.Scatter(
                    x=[nifty_metrics['Volatility'] / 100.0],
                    y=[(nifty_metrics['CAGR Available'] if use_available else (nifty_metrics['CAGR 2Y'] if pd.notnull(nifty_metrics['CAGR 2Y']) else nifty_metrics['CAGR Available'])) / 100.0],
                    mode='markers',
                    marker=dict(size=15, symbol='star', color='red'),
                    name='NIFTY 100',
                    hoverinfo='text',
                    text='NIFTY 100',
                    showlegend=True
                )
            )
        
        fig.update_layout(
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(tickformat=".1%")
        )
        
        return fig

def main():
    st.set_page_config(
        page_title="Advanced Mutual Fund Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Advanced Mutual Fund Analysis Dashboard")
    st.markdown("---")
    
    # Initialize analyzer
    analyzer = FundAnalyzer(
        'largecap_regular_growth_raw_3yrs (1).csv',
        'nifty100_historical_data (1).csv'
    )
    
    with st.spinner('Loading and analyzing data...'):
        # Load data
        returns = analyzer.load_data()

        # Global calculation window selector
        st.sidebar.header("Calculation Window")
        calc_horizon_map = {
            'Full history': None,
            '2Y': 504,
            '1Y': 252,
            '6M': 126,
            '3M': 63,
            '2M': 42,
            '1M': 21
        }
        calc_choice = st.sidebar.selectbox("Apply calculations over:", list(calc_horizon_map.keys()), index=0)
        calc_days = calc_horizon_map[calc_choice]
        returns_eff = returns if calc_days is None else returns.tail(min(calc_days, len(returns)))
        # Prepare effective NIFTY returns series for same window size
        nifty_full = analyzer.nifty_returns if analyzer.nifty_returns is not None else pd.Series(dtype=float)
        if calc_days is None:
            nifty_eff = nifty_full
        else:
            nifty_eff = nifty_full.tail(min(calc_days, len(nifty_full)))
        st.sidebar.caption(f"Using {len(returns_eff)} trading days for all calculations")

        # Diagnostics: show how many data points exist with MF-only dates vs others
        st.sidebar.markdown("---")
        st.sidebar.subheader("Diagnostics")
        try:
            mf_days = analyzer.mf_data['Date'].nunique()
            mf_min = analyzer.mf_data['Date'].min().date()
            mf_max = analyzer.mf_data['Date'].max().date()
        except Exception:
            mf_days, mf_min, mf_max = None, None, None
        try:
            n_days = analyzer.nifty_data['Date'].nunique()
            n_min = analyzer.nifty_data['Date'].min().date()
            n_max = analyzer.nifty_data['Date'].max().date()
        except Exception:
            n_days, n_min, n_max = None, None, None
        try:
            ret_days = analyzer.returns_data.index.nunique()
            ret_min = analyzer.returns_data.index.min().date()
            ret_max = analyzer.returns_data.index.max().date()
        except Exception:
            ret_days, ret_min, ret_max = None, None, None
        eff_days = returns_eff.index.nunique()
        eff_min = returns_eff.index.min().date()
        eff_max = returns_eff.index.max().date()

        st.sidebar.markdown(f"- MF-only trading days: **{mf_days}**\n  - Range: {mf_min} → {mf_max}")
        st.sidebar.markdown(f"- NIFTY trading days: **{n_days}**\n  - Range: {n_min} → {n_max}")
        st.sidebar.markdown(f"- Combined (returns) index days: **{ret_days}**\n  - Range: {ret_min} → {ret_max}")
        st.sidebar.markdown(f"- Effective window days: **{eff_days}**\n  - Range: {eff_min} → {eff_max}")

        # Calculate metrics on effective returns slice
        metrics = analyzer.calculate_metrics(returns_eff, nifty_eff)
        
        # Sort by Sharpe Ratio by default
        metrics = metrics.sort_values('Sharpe Ratio', ascending=False)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview",
            "📊 Performance Analysis",
            "📉 Risk Analysis",
            "🔍 Fund Comparison"
        ])
        
        with tab1:
            st.header("Fund Performance Overview")
            
            # Top funds by Market Outperformance
            st.subheader("🏆 Top Funds by Market Outperformance")
            top_outperformers = metrics.sort_values('Quarters Beat Market', ascending=False).head(5)
            st.dataframe(top_outperformers[['Quarters Beat Market', 'Total Quarters', 'Market Outperformance %', 'CAGR 2Y', 'Sharpe Ratio']].style
                .format({
                    'CAGR 2Y': '{:.1f}%',
                    'Market Outperformance %': '{:.1f}%',
                    'Sharpe Ratio': '{:.2f}'
                }))
            st.caption(f"Shows funds that have beaten NIFTY100 most frequently over {int(top_outperformers['Total Quarters'].iloc[0])} quarters (full history)")
            
            # Quarterly returns table (last 3 years) for top outperformers vs NIFTY100
            st.subheader("Quarterly Returns (last 3 years)")
            end_date = returns_eff.index.max()
            start_date = end_date - pd.DateOffset(years=3)
            qr = (1 + returns_eff.loc[returns_eff.index >= start_date]).resample('Q').prod() - 1
            # Add NIFTY quarterly column from independent series if available
            if analyzer.nifty_returns is not None and len(analyzer.nifty_returns) > 0:
                nifty_q = (1 + analyzer.nifty_returns.loc[analyzer.nifty_returns.index >= start_date]).resample('Q').prod() - 1
                qr = qr.join(nifty_q.rename('NIFTY100'), how='left')
            # Label quarters as YYYYQn
            qr.index = qr.index.to_period('Q').astype(str)
            display_funds = top_outperformers.index.tolist()
            cols_to_show = [c for c in display_funds if c in qr.columns]
            if 'NIFTY100' in qr.columns:
                cols_to_show += ['NIFTY100']
            if cols_to_show:
                qr_show = (qr[cols_to_show] * 100).round(2)
                st.dataframe(qr_show)
            else:
                st.info("No overlapping quarterly data available to display.")
            
            # Top 5 funds by Sharpe Ratio
            st.subheader("📈 Top Performing Funds (by Sharpe Ratio)")
            top_funds = metrics.sort_values('Sharpe Ratio', ascending=False).head(5)
            st.dataframe(top_funds[['CAGR 2Y', 'Volatility', 'Sharpe Ratio', 'Monthly Return', 'Weekly Return']].style
                .format({
                    'CAGR 2Y': '{:.1f}%',
                    'Volatility': '{:.1f}%',
                    'Monthly Return': '{:.2f}%',
                    'Weekly Return': '{:.2f}%',
                    'Sharpe Ratio': '{:.2f}'
                }))
            
            # Returns Heatmap
            st.subheader("Returns by Period")
            # Ensure heatmap uses the same returns slice
            analyzer.returns_data = returns_eff
            heatmap_fig = analyzer.plot_returns_heatmap(metrics)
            st.plotly_chart(heatmap_fig, use_container_width=True)

            # Composite / Pairwise Ranking
            st.subheader("🏅 Fund Ranking (Multi-Metric)")
            st.caption("Rank funds using either percentile-based composite score or pairwise wins across metrics.")
            # Presets and custom weights
            with st.expander("Ranking Settings", expanded=False):
                preset = st.selectbox("Preset", ["Balanced", "Return-focused", "Risk-focused", "Custom"], index=0)
                if preset == "Balanced":
                    weights = {
                        'Annualized Return': 2,
                        'CAGR Available': 2,
                        'Sharpe Ratio': 2,
                        'Sortino Ratio': 1,
                        'Volatility': 1,
                        'Downside Volatility': 1,
                        'Max Drawdown': 1,
                        'Market Outperformance %': 1
                    }
                elif preset == "Return-focused":
                    weights = {
                        'Annualized Return': 3,
                        'CAGR Available': 3,
                        'Sharpe Ratio': 2,
                        'Sortino Ratio': 1,
                        'Volatility': 1,
                        'Max Drawdown': 0.5,
                        'Market Outperformance %': 1
                    }
                elif preset == "Risk-focused":
                    weights = {
                        'Annualized Return': 1,
                        'CAGR Available': 1,
                        'Sharpe Ratio': 2,
                        'Sortino Ratio': 2,
                        'Volatility': 2,
                        'Downside Volatility': 2,
                        'Max Drawdown': 2,
                        'Market Outperformance %': 1
                    }
                else:
                    # Custom sliders
                    weights = {}
                    metrics_for_weights = ['Annualized Return','CAGR Available','CAGR 2Y','Sharpe Ratio','Sortino Ratio','Volatility','Downside Volatility','Max Drawdown','Market Outperformance %']
                    cols = st.columns(2)
                    for i, m in enumerate(metrics_for_weights):
                        with cols[i % 2]:
                            weights[m] = st.slider(m, min_value=0.0, max_value=5.0, value=2.0 if m in ['Annualized Return','Sharpe Ratio'] else 1.0, step=0.5)
                ranking_method = st.radio("Ranking method:", ["Percentile (Composite Score)", "Pairwise (Wins Score)"], index=1, horizontal=False)
                exclude_nifty = st.checkbox("Exclude NIFTY 100 from ranking", value=True)

            # Ensure Annualized Return exists for ranking; compute from returns_eff if missing
            if 'Annualized Return' not in metrics.columns:
                ann_tab = analyzer.compute_annualized_metrics(returns_eff)
                for col in ['Annualized Return']:
                    if col in ann_tab.columns:
                        metrics[col] = ann_tab[col]

            if ranking_method.startswith("Percentile"):
                ranked = analyzer.compute_composite_score(metrics, weights, exclude_nifty=exclude_nifty)
                if not ranked.empty:
                    ranked = ranked.copy()
                    ranked['Rank'] = np.arange(1, len(ranked) + 1)
                    show_cols = ['Rank','Composite Score','Annualized Return','CAGR Available','Sharpe Ratio','Sortino Ratio','Volatility','Downside Volatility','Max Drawdown','Market Outperformance %']
                    show_cols = [c for c in show_cols if c in ranked.columns]
                    st.dataframe(ranked[show_cols].round(2).style.format({
                        'Composite Score': '{:.3f}',
                        'Annualized Return': '{:.2f}%','CAGR Available': '{:.2f}%','Sharpe Ratio': '{:.2f}','Sortino Ratio': '{:.2f}','Volatility': '{:.2f}%','Downside Volatility': '{:.2f}%','Max Drawdown': '{:.1f}%','Market Outperformance %': '{:.1f}%'
                    }))
                    # Bar chart for top 10 composite scores
                    topN = ranked.head(10).reset_index().rename(columns={'index': 'Fund'})
                    fig_comp = px.bar(
                        topN,
                        x='Fund',
                        y='Composite Score',
                        title='Top 10 Funds by Composite Score',
                        labels={'Fund': 'Fund', 'Composite Score': 'Score (0-1)'}
                    )
                    fig_comp.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_comp, use_container_width=True)
            else:
                ranked = analyzer.compute_pairwise_wins(metrics, weights, exclude_nifty=exclude_nifty)
                if not ranked.empty:
                    ranked = ranked.copy()
                    ranked['Rank'] = np.arange(1, len(ranked) + 1)
                    show_cols = ['Rank','Pairwise Score','Annualized Return','CAGR Available','Sharpe Ratio','Sortino Ratio','Volatility','Downside Volatility','Max Drawdown','Market Outperformance %']
                    show_cols = [c for c in show_cols if c in ranked.columns]
                    st.dataframe(ranked[show_cols].round(2).style.format({
                        'Pairwise Score': '{:.3f}',
                        'Annualized Return': '{:.2f}%','CAGR Available': '{:.2f}%','Sharpe Ratio': '{:.2f}','Sortino Ratio': '{:.2f}','Volatility': '{:.2f}%','Downside Volatility': '{:.2f}%','Max Drawdown': '{:.1f}%','Market Outperformance %': '{:.1f}%'
                    }))
                    # Bar chart for top 10 pairwise scores
                    topN = ranked.head(10).reset_index().rename(columns={'index': 'Fund'})
                    fig_comp = px.bar(
                        topN,
                        x='Fund',
                        y='Pairwise Score',
                        title='Top 10 Funds by Pairwise Wins Score',
                        labels={'Fund': 'Fund', 'Pairwise Score': 'Score (0-1)'}
                    )
                    fig_comp.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_comp, use_container_width=True)
            
        with tab2:
            st.header("Performance Analysis")
            
            # Risk-Return Profile (dynamic horizon)
            st.subheader("Risk-Return Profile (select horizon)")
            horizon_map = {
                '1M': 21,
                '2M': 42,
                '3M': 63,
                '6M': 126,
                '1Y': 252,
                '2Y': 504
            }
            choice = st.selectbox("Select horizon for Return and Volatility:", list(horizon_map.keys()), index=4)
            window_days = horizon_map[choice]
            use_full = st.checkbox("Use full history for risk-return (override horizon)", value=False)
            title_suffix = f"Full history" if use_full else f"{choice} window"
            if use_full:
                window_days = len(returns)
            rr_fig = analyzer.plot_risk_return_window(returns_eff, window_days if not use_full else len(returns_eff), title_suffix=title_suffix)
            # Guard: ensure rr_fig is a Plotly Figure
            if isinstance(rr_fig, dict):
                rr_fig = go.Figure(rr_fig)
            elif not isinstance(rr_fig, go.Figure):
                # Last resort fallback
                tmp_fig = go.Figure()
                tmp_fig.update_layout(title="Risk-Return: unexpected return type; showing empty figure")
                rr_fig = tmp_fig
            st.plotly_chart(rr_fig, use_container_width=True)
            # Show info about rows actually used
            available_rows = min(window_days if not use_full else len(returns_eff), len(returns_eff))
            if available_rows > 0:
                end_dt = returns_eff.index.max()
                start_dt = returns_eff.index[-available_rows]
                requested_text = "full history" if use_full else str(window_days)
                st.caption(f"Computed using {available_rows} trading days (requested {requested_text}). Window: {start_dt.date()} → {end_dt.date()}")
            
            # Validation panel: inspect precise return calculation for a fund
            with st.expander("Validate window return calculation for a fund"):
                valid_funds = [f for f in metrics.index.tolist() if f != 'NIFTY100']
                if valid_funds:
                    vf = st.selectbox("Select fund to validate:", valid_funds, index=0, key="validate_fund")
                    series = returns_eff[vf].tail(available_rows)
                    series_nonan = series.dropna()
                    days_used = len(series_nonan)
                    cum_return = (1 + series_nonan).prod() - 1 if days_used > 0 else np.nan
                    vol_window = series_nonan.std() if days_used > 1 else np.nan
                    st.write(f"Days used (non-NaN): {days_used}")
                    st.write(f"Cumulative window return (non-annualized): {cum_return:.4%}")
                    st.write(f"Volatility (daily std in window, non-annualized): {vol_window:.4%}")
                    # Show sample of returns used
                    if days_used > 0:
                        head_tail = pd.concat([series_nonan.head(3), series_nonan.tail(3)])
                        st.dataframe((head_tail * 100).round(3).rename("Daily Return (%)"))

            # Annualized Risk-Return (Return vs Volatility)
            st.subheader("Annualized Risk-Return (Return vs Volatility)")
            ann_fig = analyzer.plot_annualized_risk_return(returns_eff, nifty_eff)
            st.plotly_chart(ann_fig, use_container_width=True)
            # Also show a table for clarity
            ann_table = analyzer.compute_annualized_metrics(returns_eff)
            if not ann_table.empty:
                # Append NIFTY row to table if available
                if nifty_eff is not None and len(nifty_eff.dropna()) > 1:
                    n = nifty_eff.dropna()
                    n_ret = (1 + n.mean()) ** TRADING_DAYS - 1
                    n_vol = n.std() * np.sqrt(TRADING_DAYS)
                    ann_table.loc['NIFTY100', 'Annualized Return'] = n_ret * 100
                    ann_table.loc['NIFTY100', 'Annualized Volatility'] = n_vol * 100
                st.dataframe(ann_table.sort_values('Annualized Return', ascending=False).round(2))
            
            # Performance Metrics Table
            st.subheader("Performance Metrics")
            st.dataframe(metrics.style.format({
                'Total Return': '{:.1f}%',
                'Weekly Return': '{:.2f}%',
                'Monthly Return': '{:.2f}%',
                'CAGR Available': '{:.1f}%',
                'Volatility': '{:.1f}%',
                'Sharpe Ratio': '{:.2f}',
                'Sortino Ratio': '{:.2f}',
                'Beta': '{:.2f}',
                'Alpha': '{:.2f}%','Max Drawdown': '{:.1f}%','Downside Volatility': '{:.1f}%','Quarters Beat Market': '{:.0f}','Market Outperformance %': '{:.1f}%'
            }))
            
        with tab3:
            st.header("Risk Analysis")
            
            # Controls
            include_nifty = st.checkbox("Include NIFTY 100 in charts", value=True)
            vol_type = st.radio("Volatility type:", ["Annualized", "Window std (daily)", "Downside (annualized)"], index=0, horizontal=True)
            annualized_flag = (vol_type == "Annualized")

            # Build effective returns set for risk analysis (MF + optional NIFTY)
            risk_df = returns_eff.copy()
            if include_nifty and analyzer.nifty_returns is not None and len(analyzer.nifty_returns) > 0:
                risk_df = risk_df.join(analyzer.nifty_returns.rename('NIFTY100'), how='left')

            # Max Drawdown (compute directly from effective slice)
            st.subheader("Maximum Drawdown (computed over selected calculation window)")
            mdd_series = analyzer.compute_max_drawdown(risk_df)
            mdd_df = mdd_series.sort_values().to_frame(name='Max Drawdown (%)')  # most negative first
            fig_mdd = px.bar(
                mdd_df.head(15).reset_index().rename(columns={'index': 'Fund'}),
                x='Fund',
                y='Max Drawdown (%)',
                title="Worst Drawdowns (Top 15)",
                labels={'Fund': 'Fund', 'Max Drawdown (%)': 'Max Drawdown (%)'},
                color='Max Drawdown (%)',
                color_continuous_scale='RdYlGn_r'
            )
            fig_mdd.update_layout(xaxis_tickangle=-45)
            fig_mdd.update_traces(hovertemplate='%{x}<br>Max Drawdown: %{y:.1f}%<extra></extra>')
            st.plotly_chart(fig_mdd, use_container_width=True)

            # Volatility (compute directly and allow annualized/window std)
            st.subheader(f"Volatility ({'Annualized' if annualized_flag else 'Window std (daily)'})")
            if vol_type == "Downside (annualized)":
                vol_series = analyzer.compute_downside_volatility(risk_df, annualized=True)
            else:
                vol_series = analyzer.compute_volatility(risk_df, annualized=annualized_flag)
            vol_df = vol_series.sort_values(ascending=False).to_frame(name='Volatility (%)')
            fig_vol = px.bar(
                vol_df.head(15).reset_index().rename(columns={'index': 'Fund'}),
                x='Fund',
                y='Volatility (%)',
                title="Most Volatile (Top 15)",
                labels={'Fund': 'Fund', 'Volatility (%)': 'Volatility (%)'},
                color='Volatility (%)',
                color_continuous_scale='Viridis'
            )
            fig_vol.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_vol, use_container_width=True)
            
        with tab4:
            st.header("Fund Comparison")
            
            # Fund Selector
            selected_funds = st.multiselect(
                "Select up to 5 funds to compare:",
                options=metrics.index.tolist(),
                default=metrics.index[:3].tolist()
            )
            
            if len(selected_funds) > 5:
                st.warning("Please select a maximum of 5 funds.")
            elif len(selected_funds) >= 1:
                # Cumulative Returns Comparison
                st.subheader("Cumulative Returns Comparison")
                plot_df = returns_eff[selected_funds].copy()
                if analyzer.nifty_returns is not None and len(analyzer.nifty_returns) > 0:
                    plot_df = plot_df.join(analyzer.nifty_returns.rename('NIFTY100'), how='left')
                # Compute growth of 1; treat missing returns as 0 for continuity
                growth = (1 + plot_df.fillna(0)).cumprod()
                fig = px.line(
                    growth,
                    title="Growth of ₹1 Investment",
                    labels={'value': 'Growth of ₹1', 'variable': 'Fund'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics Comparison
                st.subheader("Metrics Comparison")
                st.dataframe(
                    metrics.loc[selected_funds + ['NIFTY100']].style
                    .format({
                        'Weekly Return': '{:.2f}%',
                        'Monthly Return': '{:.2f}%',
                        'CAGR Available': '{:.1f}%',
                        'Volatility': '{:.1f}%',
                        'Sharpe Ratio': '{:.2f}',
                        'Sortino Ratio': '{:.2f}',
                        'Beta': '{:.2f}',
                        'Alpha': '{:.2f}%','Max Drawdown': '{:.1f}%'
                    })
                )
            
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - **Sharpe Ratio**: Higher is better (risk-adjusted return)
    - **Sortino Ratio**: Higher is better (downside risk-adjusted return)
    - **Beta**: Measures market risk (1 = market, <1 = less volatile, >1 = more volatile)
    - **Alpha**: Excess return compared to benchmark (positive is good)
    - **Max Drawdown**: Maximum observed loss from peak (lower is better)
    """)

if __name__ == "__main__":
    main()
