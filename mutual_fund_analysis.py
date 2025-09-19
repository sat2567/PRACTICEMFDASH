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
    mf_df['Date'] = pd.to_datetime(mf_df['Date']).dt.tz_localize(None)  # Remove timezone info
    
    # Load NIFTY 100 data
    nifty_df = pd.read_csv('nifty100_historical_data (1).csv')
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date']).dt.tz_localize(None)  # Remove timezone info
    nifty_df = nifty_df[['Date', 'Close']].rename(columns={'Close': 'NIFTY100'})
    
    # Sort both dataframes by date
    mf_df = mf_df.sort_values('Date')
    nifty_df = nifty_df.sort_values('Date')
    
    # Ensure no duplicate dates for each fund
    mf_df = mf_df.drop_duplicates(['Date', 'Scheme Name'])
    nifty_df = nifty_df.drop_duplicates('Date')
    
    return mf_df, nifty_df

def calculate_returns(mf_df, nifty_df):
    # Pivot mutual fund data to get NAVs by date and fund
    mf_navs = mf_df.pivot(index='Date', columns='Scheme Name', values='NAV')
    
    # Calculate daily returns for each mutual fund
    mf_returns = mf_navs.pct_change()
    
    # Calculate NIFTY 100 returns
    nifty_returns = nifty_df.set_index('Date')['NIFTY100'].pct_change()
    
    # Align both dataframes on date (inner join to ensure we only keep dates with data in both)
    combined_returns = pd.concat([
        mf_returns,
        nifty_returns.rename('NIFTY100')
    ], axis=1).dropna()
    
    return combined_returns

def calculate_period_returns(returns, periods):
    period_returns = {}
    
    for name, period in periods.items():
        if period == '1M':
            window = 21  # Approx 21 trading days in a month
        elif period == '3M':
            window = 63  # Approx 63 trading days in 3 months
        elif period == '6M':
            window = 126  # Approx 126 trading days in 6 months
        elif period == '12M':
            window = 252  # Approx 252 trading days in a year
        elif period == '3Y':
            window = 756  # Approx 756 trading days in 3 years
        
        # Calculate rolling returns
        period_return = (1 + returns).rolling(window=window).apply(np.prod) - 1
        
        # Get the most recent period return
        latest_return = period_return.iloc[-1].to_frame('Return')
        latest_return['Period'] = period
        period_returns[period] = latest_return
    
    # Combine all periods
    all_returns = pd.concat(period_returns.values())
    all_returns = all_returns.reset_index().rename(columns={'index': 'Scheme'})
    
    return all_returns

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

def create_dashboard(combined_returns):
    # Import dash components here to avoid import issues
    from dash import Dash, html, dcc, dash_table
    import plotly.express as px
    
    # Calculate metrics
    periods = {'1M': '1M', '3M': '3M', '6M': '6M', '12M': '12M', '3Y': '3Y'}
    period_returns = calculate_period_returns(combined_returns, periods)
    
    # Calculate max drawdown
    max_dd = calculate_max_drawdown(combined_returns)
    
    # Create dashboard
    app = Dash(__name__)
    
    # Get top 10 funds by 3Y return for cleaner visualization
    top_funds = period_returns[period_returns['Period'] == '3Y']\
        .nlargest(10, 'Return')['Scheme'].tolist()
    
    # Prepare data for visualization
    heatmap_data = period_returns[period_returns['Scheme'].isin(top_funds + ['NIFTY100'])]
    
    app.layout = html.Div([
        html.H1("Mutual Fund Performance Analysis", style={'textAlign': 'center'}),
        
        # Period Returns Heatmap
        html.H2("Periodic Returns Comparison (Top 10 Funds by 3Y Return)"),
        dcc.Graph(
            id='returns-heatmap',
            figure=px.imshow(
                heatmap_data.pivot(index='Scheme', columns='Period', values='Return') * 100,
                labels=dict(x="Period", y="Scheme", color="Return (%)"),
                title="Returns by Period (%)",
                color_continuous_scale='RdYlGn',
                aspect="auto"
            )
        ),
        
        # Max Drawdown Bar Chart
        html.H2("Maximum Drawdown (Top 10 Funds)"),
        dcc.Graph(
            id='max-drawdown',
            figure=px.bar(
                max_dd[max_dd.index.isin(top_funds + ['NIFTY100'])].reset_index()\
                    .rename(columns={'index': 'Scheme', 0: 'Max Drawdown'}),
                x='Scheme',
                y='Max Drawdown',
                title='Maximum Drawdown',
                color='Max Drawdown',
                color_continuous_scale='RdYlGn_r',
                text_auto='.2%'
            ).update_layout(yaxis_tickformat=".0%")
        ),
        
        # NIFTY 100 vs Funds Comparison
        html.H2("NIFTY 100 vs Top Performing Funds"),
        dcc.Graph(
            id='nifty-comparison',
            figure=px.line(
                (1 + combined_returns[['NIFTY100'] + top_funds]).cumprod(),
                title="Growth of ₹1 Investment",
                labels={'value': 'Growth of ₹1', 'variable': 'Fund'}
            ).update_yaxes(tickformat=".2f")
        ),
        
        # Performance Table
        html.H2("Performance Metrics"),
        dash_table.DataTable(
            id='performance-table',
            columns=[
                {"name": "Scheme", "id": "Scheme"},
                {"name": "Return", "id": "Return", "type": "numeric",
                 "format": {"specifier": ".2%"}},
                {"name": "Period", "id": "Period"}
            ],
            data=heatmap_data.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'minWidth': '100px', 'width': '100px', 'maxWidth': '180px',
                'whiteSpace': 'normal',
                'textAlign': 'left'
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Return} > 0',
                        'column_id': 'Return'
                    },
                    'color': 'green'
                },
                {
                    'if': {
                        'filter_query': '{Return} < 0',
                        'column_id': 'Return'
                    },
                    'color': 'red'
                }
            ],
            page_size=20,
            sort_action='native'
        )
    ])
    
    return app

def main():
    try:
        print("Loading and preprocessing data...")
        mf_df, nifty_df = load_and_preprocess_data()
        
        print("Calculating returns...")
        combined_returns = calculate_returns(mf_df, nifty_df)
        
        print("Creating dashboard...")
        app = create_dashboard(combined_returns)
        
        print("Dashboard is ready!")
        print("Open your browser and navigate to http://127.0.0.1:8050/")
        
        # Run the Dash app
        if __name__ == '__main__':
            app.run(debug=True, port=8050)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
