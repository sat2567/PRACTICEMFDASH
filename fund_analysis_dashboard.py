import streamlit as st
import pandas as pd
import numpy as np

# ------- Constants -------
MOMENTUM_WINDOWS = {'1M': 21, '3M': 63, '6M': 126, '1Y': 252, '3Y': 756}

# ------- Data Cleaning Utility -------
def clean_mf_data(mf_file):
    mf_df = pd.read_csv(mf_file)
    mf_dates1 = pd.to_datetime(mf_df['Date'], dayfirst=True, errors='coerce')
    mf_dates2 = pd.to_datetime(mf_df['Date'], dayfirst=False, errors='coerce')
    
    def span_and_unique(s):
        s2 = s.dropna()
        return (s2.max() - s2.min()).days if len(s2) else -1, s2.nunique()
    span1, uniq1 = span_and_unique(mf_dates1)
    span2, uniq2 = span_and_unique(mf_dates2)
    mf_df['Date'] = mf_dates2 if (span2, uniq2) > (span1, uniq1) else mf_dates1
    mf_df = mf_df.dropna(subset=['Date'])
    mf_df['Date'] = mf_df['Date'].dt.tz_localize(None)
    mf_df = mf_df.sort_values('Date')
    mf_df['NAV'] = pd.to_numeric(mf_df['NAV'], errors='coerce')
    mf_df = mf_df.dropna(subset=['NAV'])
    return mf_df

def clean_nifty_data(nifty_file):
    nifty_df = pd.read_csv(nifty_file)
    n_dates1 = pd.to_datetime(nifty_df['Date'], dayfirst=True, errors='coerce')
    n_dates2 = pd.to_datetime(nifty_df['Date'], dayfirst=False, errors='coerce')
    def span_and_unique(s):
        s2 = s.dropna()
        return (s2.max() - s2.min()).days if len(s2) else -1, s2.nunique()
    n_span1, n_uniq1 = span_and_unique(n_dates1)
    n_span2, n_uniq2 = span_and_unique(n_dates2)
    nifty_df['Date'] = n_dates2 if (n_span2, n_uniq2) > (n_span1, n_uniq1) else n_dates1
    nifty_df = nifty_df.dropna(subset=['Date'])
    nifty_df['Date'] = nifty_df['Date'].dt.tz_localize(None)
    # Ensure correct columns
    nifty_df = nifty_df[['Date', 'Close']].rename(columns={'Close': 'NIFTY100'})
    nifty_df = nifty_df.sort_values('Date').drop_duplicates('Date')
    return nifty_df

# ------- Analysis Functions -------
def get_nav_pivot(mf_df):
    mf_navs = mf_df.pivot(index='Date', columns='Scheme Name', values='NAV')
    mf_navs = mf_navs[~mf_navs.index.duplicated(keep='last')]
    mf_navs.index = mf_navs.index.tz_localize(None)
    mf_navs = mf_navs.sort_index()
    return mf_navs

def get_returns(mf_navs, nifty_df):
    mf_returns = mf_navs.pct_change()
    nifty_series = nifty_df.set_index('Date')['NIFTY100'].sort_index().pct_change()
    return mf_returns, nifty_series

def get_momentum(mf_returns, window_days):
    recent_returns = mf_returns.tail(window_days)
    momentum = (1 + recent_returns).prod() - 1
    momentum = (momentum * 100).round(2)
    return momentum

def calculate_quarterly_outperformance(mf_returns, nifty_series):
    quarterly_returns = (1 + mf_returns).resample('Q').prod() - 1
    nifty_quarterly = (1 + nifty_series).resample('Q').prod() - 1
    fund_cols = quarterly_returns.columns
    aligned = quarterly_returns.join(nifty_quarterly.rename('NIFTY100'), how="inner")
    outperformance = (aligned[fund_cols].gt(aligned['NIFTY100'], axis=0)).sum(axis=0)
    total_quarters = int(len(aligned.index))
    outperf_pct = (outperformance / total_quarters * 100) if total_quarters else outperformance.astype(float)
    return outperformance, total_quarters, outperf_pct

# ------- Streamlit App -------
def main():
    st.title("Mutual Fund Dashboard: Momentum & Market Outperformance")
    st.caption("Analyzing using robust data cleaning (date parsing, sorting, pivoting)")

    # --- File Names ---
    mf_file = "largecap_regular_growth_raw_10yrs.csv"
    nifty_file = "nifty100_filtered_data.csv"

    # --- Cleaning & Preparation ---
    mf_df = clean_mf_data(mf_file)
    nifty_df = clean_nifty_data(nifty_file)
    mf_navs = get_nav_pivot(mf_df)
    mf_returns, nifty_series = get_returns(mf_navs, nifty_df)

    # --- Sidebar Setting ---
    st.sidebar.header("Momentum Settings")
    window_label = st.sidebar.selectbox("Select Momentum Window", list(MOMENTUM_WINDOWS.keys()), index=1)
    window_days = MOMENTUM_WINDOWS[window_label]

    # --- Momentum Analysis ---
    momentum = get_momentum(mf_returns, window_days)
    momentum_df = pd.DataFrame({f"Momentum {window_label} (%)": momentum})
    momentum_df = momentum_df.sort_values(f"Momentum {window_label} (%)", ascending=False)

    # --- Quarter Outperformance Analysis ---
    outperf, total_quarters, outperf_pct = calculate_quarterly_outperformance(mf_returns, nifty_series)
    qtr_df = pd.DataFrame({
        "Quarters Beat Market": outperf,
        "Total Quarters": total_quarters,
        "Beat %": outperf_pct.round(1)
    }).sort_values("Quarters Beat Market", ascending=False)

    # --- Display Results ---
    st.header(f"Momentum Analysis ({window_label})")
    st.dataframe(momentum_df)

    st.header("Number of Quarters Fund Beat Market")
    st.dataframe(qtr_df)

if __name__ == "__main__":
    main()
