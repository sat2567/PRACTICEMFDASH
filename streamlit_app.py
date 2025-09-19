import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    # Load mutual fund data
    mf_df = pd.read_csv('largecap_regular_growth_raw_3yrs (1).csv')
    mf_df['Date'] = pd.to_datetime(mf_df['Date']).dt.tz_localize(None)
    
    # Load NIFTY 100 data
    nifty_df = pd.read_csv('nifty100_historical_data (1).csv')
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date']).dt.tz_localize(None)
    nifty_df = nifty_df[['Date', 'Close']].rename(columns={'Close': 'NIFTY100'})
    
    # Sort both dataframes by date and remove duplicates
    mf_df = mf_df.sort_values('Date').drop_duplicates(['Date', 'Scheme Name'])
    nifty_df = nifty_df.sort_values('Date').drop_duplicates('Date')
    
    return mf_df, nifty_df

def calculate_returns(mf_df, nifty_df):
    # Pivot mutual fund data to get NAVs by date and fund
    mf_navs = mf_df.pivot(index='Date', columns='Scheme Name', values='NAV')
    
    # Calculate daily returns for each mutual fund
    mf_returns = mf_navs.pct_change()
    
    # Calculate NIFTY 100 returns
    nifty_returns = nifty_df.set_index('Date')['NIFTY100'].pct_change()
    
    # Align both dataframes on date
    combined_returns = pd.concat([
        mf_returns,
        nifty_returns.rename('NIFTY100')
    ], axis=1).dropna()
    
    return combined_returns

def calculate_period_returns(returns, periods):
    period_returns = []
    
    for scheme in returns.columns:
        # Calculate cumulative returns
        nav_series = (1 + returns[scheme]).cumprod()
        
        # Calculate 3-year return for all funds
        if len(nav_series) > 1:  # Need at least 2 data points
            start_nav = nav_series.iloc[0]
            end_nav = nav_series.iloc[-1]
            three_year_return = (end_nav / start_nav) ** (1/3) - 1  # Annualized return
            
            period_returns.append({
                'Scheme': scheme,
                'Period': '3Y',
                'Return': three_year_return
            })
            
            # Calculate other periods for the last year only
            for period_name, days in periods.items():
                if period_name == '3Y':
                    continue  # Already calculated
                    
                if len(nav_series) > days:
                    # Calculate return for the period
                    start_idx = -days - 1 if len(nav_series) > days else 0
                    period_return = (nav_series.iloc[-1] / nav_series.iloc[start_idx]) - 1
                    
                    period_returns.append({
                        'Scheme': scheme,
                        'Period': period_name,
                        'Return': period_return
                    })
    
    return pd.DataFrame(period_returns)

def calculate_max_drawdown(returns):
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max
    
    # Get maximum drawdown for each fund
    max_drawdown = drawdown.min()
    
    return max_drawdown

def main():
    st.set_page_config(page_title="Mutual Fund Analysis", layout="wide")
    st.title("Mutual Fund Performance Analysis")
    
    with st.spinner('Loading and processing data...'):
        # Load and preprocess data
        mf_df, nifty_df = load_and_preprocess_data()
        
        # Calculate returns
        combined_returns = calculate_returns(mf_df, nifty_df)
        
        # Define periods in trading days
        periods = {
            '1M': 21,    # 1 month
            '3M': 63,    # 3 months
            '6M': 126,   # 6 months
            '12M': 252,  # 1 year
            '3Y': 756    # 3 years (used for filtering, not calculation)
        }
        period_returns = calculate_period_returns(combined_returns, periods)
        
        # Ensure we only keep periods with valid data
        period_returns = period_returns.dropna(subset=['Return'])
        period_returns = period_returns[period_returns['Return'].notna()]
        
        # Calculate max drawdown
        max_dd = calculate_max_drawdown(combined_returns)
        
        # Get all funds with 3Y returns
        funds_3y = period_returns[period_returns['Period'] == '3Y']
        
        # Sort by 3Y return descending and get top 10
        top_funds = funds_3y.nlargest(10, 'Return')['Scheme'].tolist()
        
        # Ensure we have returns for all periods for the top funds
        period_returns = period_returns[
            (period_returns['Scheme'].isin(top_funds + ['NIFTY100'])) | 
            (period_returns['Period'] == '3Y')
        ]
        
        # Prepare data for visualization
        heatmap_data = period_returns[period_returns['Scheme'].isin(top_funds + ['NIFTY100'])]
    
    # Display the heatmap
    st.header("Periodic Returns Comparison (Top 10 Funds by 3Y Return)")
    fig = px.imshow(
        heatmap_data.pivot(index='Scheme', columns='Period', values='Return') * 100,
        labels=dict(x="Period", y="Scheme", color="Return (%)"),
        title="Returns by Period (%)",
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display max drawdown chart
    st.header("Maximum Drawdown (Top 10 Funds)")
    max_dd_df = max_dd[max_dd.index.isin(top_funds + ['NIFTY100'])].reset_index()\
        .rename(columns={'index': 'Scheme', 0: 'Max Drawdown'})
    fig = px.bar(
        max_dd_df,
        x='Scheme',
        y='Max Drawdown',
        title='Maximum Drawdown',
        color='Max Drawdown',
        color_continuous_scale='RdYlGn_r',
        text_auto='.2%'
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display NIFTY 100 vs Funds Comparison
    st.header("NIFTY 100 vs Top Performing Funds")
    growth_data = (1 + combined_returns[['NIFTY100'] + top_funds]).cumprod()
    fig = px.line(
        growth_data,
        title="Growth of ₹1 Investment",
        labels={'value': 'Growth of ₹1', 'variable': 'Fund'}
    )
    fig.update_yaxes(tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display Performance Table
    st.header("Performance Metrics")
    st.dataframe(
        heatmap_data.pivot(index='Scheme', columns='Period', values='Return').style
            .format('{:.2%}')
            .applymap(lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '')
            .set_properties(**{'text-align': 'center'})
    )

if __name__ == "__main__":
    main()
