import pandas as pd
import yfinance as yf
import requests
import numpy as np
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_fixed
import schedule
import time
import psycopg2
from psycopg2.extras import execute_values
import pytz
import json

# PostgreSQL Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'dbname': 'trading_data',
    'user': 'postgres',
    'password': '1234'
}

# Establish a connection to the PostgreSQL database
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            dbname=DB_CONFIG['dbname'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# Fetch Historical Data from Yahoo Finance
@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def fetch_index_data_yf(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]  # Flatten MultiIndex
        data.rename(columns={"Adj Close": "Close"}, inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        raise

# Fetch Options Chain Data from NSE India API
@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def fetch_options_chain(symbol):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain"
    }
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)  # Initial request to set cookies
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'records' not in data or 'data' not in data['records']:
            print(f"Error: No valid data found for {symbol}")
            return pd.DataFrame()

        options_data = []
        for record in data['records']['data']:
            ce_data = record.get('CE', {})
            pe_data = record.get('PE', {})
            options_data.append({
                'strikePrice': record['strikePrice'],
                'CE_LTP': ce_data.get('lastPrice', None),
                'PE_LTP': pe_data.get('lastPrice', None),
                'CE_IV': ce_data.get('impliedVolatility', None),
                'PE_IV': pe_data.get('impliedVolatility', None),
                'CE_OI': ce_data.get('openInterest', None),
                'PE_OI': pe_data.get('openInterest', None),
            })
        options_df = pd.DataFrame(options_data)
        options_df = options_df.dropna(subset=['CE_LTP', 'PE_LTP'])  # Drop rows with NaN LTP
        return options_df
    except Exception as e:
        print(f"Error fetching options chain for {symbol}: {e}")
        return pd.DataFrame()

# Calculate RSI
def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()  # EMA-based smoothing
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate MACD
def calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, macd_signal

# Generate Combined Recommendations including Straddle Logic
def generate_combined_recommendations(index_data, options_data, live_spot_price):
    recommendations = []
    rsi = calculate_rsi(index_data['Close'])
    macd, macd_signal = calculate_macd(index_data['Close'])

    if rsi.iloc[-1] > 70 and macd.iloc[-1] < macd_signal.iloc[-1]:
        trend = "Bearish"
        buy_type = "PE"
        sell_type = "CE"
    elif rsi.iloc[-1] < 30 and macd.iloc[-1] > macd_signal.iloc[-1]:
        trend = "Bullish"
        buy_type = "CE"
        sell_type = "PE"
    else:
        trend = "Neutral"

    print(f"Market Trend: {trend}")

    # Find ATM Strike Price
    atm_strike = round(live_spot_price / 100) * 100
    atm_options = options_data[options_data['strikePrice'].between(atm_strike - 200, atm_strike + 200)]

    # Directional Recommendations
    if trend in ["Bullish", "Bearish"]:
        # Recommend Buy or Sell based on trend
        try:
            best_buy_option = atm_options.loc[atm_options[f"{buy_type}_LTP"].idxmax()]
            recommendations.append({
                "Strike Price": best_buy_option['strikePrice'],
                "Type": buy_type,
                "LTP": best_buy_option[f"{buy_type}_LTP"],
                "Recommendation": f"Buy {buy_type}",
                "Stop Loss": round(best_buy_option[f"{buy_type}_LTP"] * 0.9, 2),
                "Target": round(best_buy_option[f"{buy_type}_LTP"] * 1.2, 2)
            })
        except Exception as e:
            print(f"Error generating Buy recommendation for {trend}: {e}")

        try:
            best_sell_option = atm_options.loc[atm_options[f"{sell_type}_LTP"].idxmax()]
            recommendations.append({
                "Strike Price": best_sell_option['strikePrice'],
                "Type": sell_type,
                "LTP": best_sell_option[f"{sell_type}_LTP"],
                "Recommendation": f"Sell {sell_type}",
                "Stop Loss": round(best_sell_option[f"{sell_type}_LTP"] * 1.1, 2),
                "Target": round(best_sell_option[f"{sell_type}_LTP"] * 0.8, 2)
            })
        except Exception as e:
            print(f"Error generating Sell recommendation for {trend}: {e}")

    elif trend == "Neutral":
        # Neutral Market: Recommend based on higher premium
        if atm_options['CE_LTP'].max() > atm_options['PE_LTP'].max():
            try:
                best_option = atm_options.loc[atm_options['CE_LTP'].idxmax()]
                recommendations.append({
                    "Strike Price": best_option['strikePrice'],
                    "Type": "CE",
                    "LTP": best_option['CE_LTP'],
                    "Recommendation": "Buy CE",
                    "Stop Loss": round(best_option['CE_LTP'] * 0.9, 2),
                    "Target": round(best_option['CE_LTP'] * 1.2, 2)
                })
            except Exception as e:
                print(f"Error generating Buy CE recommendation: {e}")
        else:
            try:
                best_option = atm_options.loc[atm_options['PE_LTP'].idxmax()]
                recommendations.append({
                    "Strike Price": best_option['strikePrice'],
                    "Type": "PE",
                    "LTP": best_option['PE_LTP'],
                    "Recommendation": "Buy PE",
                    "Stop Loss": round(best_option['PE_LTP'] * 0.9, 2),
                    "Target": round(best_option['PE_LTP'] * 1.2, 2)
                })
            except Exception as e:
                print(f"Error generating Buy PE recommendation: {e}")

        # Straddle Recommendation: Buy CE and PE at ATM Strike
        try:
            ce_option = atm_options.loc[atm_options['CE_LTP'].idxmax()]
            pe_option = atm_options.loc[atm_options['PE_LTP'].idxmax()]
            recommendations.append({
                "Strike Price": atm_strike,
                "Type": "Straddle",
                "LTP": ce_option['CE_LTP'] + pe_option['PE_LTP'],
                "Recommendation": "Buy Straddle",
                "Stop Loss": round((ce_option['CE_LTP'] + pe_option['PE_LTP']) * 0.85, 2),
                "Target": round((ce_option['CE_LTP'] + pe_option['PE_LTP']) * 1.25, 2)
            })

        except Exception as e:
            print(f"Error generating Straddle recommendation: {e}")

        # **Additional Sell Recommendations in Neutral Trend**
        # Sell CE at higher strike
        try:
            sell_ce_option = atm_options[atm_options['strikePrice'] > atm_strike].loc[atm_options['strikePrice'] > atm_strike].sort_values('CE_LTP', ascending=False).iloc[0]
            recommendations.append({
                "Strike Price": sell_ce_option['strikePrice'],
                "Type": "CE",
                "LTP": sell_ce_option['CE_LTP'],
                "Recommendation": "Sell CE",
                "Stop Loss": round(sell_ce_option['CE_LTP'] * 1.1, 2),
                "Target": round(sell_ce_option['CE_LTP'] * 0.8, 2)
            })
        except Exception as e:
            print(f"Error generating Sell CE recommendation in Neutral trend: {e}")

        # Sell PE at lower strike
        try:
            sell_pe_option = atm_options[atm_options['strikePrice'] < atm_strike].loc[atm_options['strikePrice'] < atm_strike].sort_values('PE_LTP', ascending=False).iloc[0]
            recommendations.append({
                "Strike Price": sell_pe_option['strikePrice'],
                "Type": "PE",
                "LTP": sell_pe_option['PE_LTP'],
                "Recommendation": "Sell PE",
                "Stop Loss": round(sell_pe_option['PE_LTP'] * 1.1, 2),
                "Target": round(sell_pe_option['PE_LTP'] * 0.8, 2)
            })
        except Exception as e:
            print(f"Error generating Sell PE recommendation in Neutral trend: {e}")

    return pd.DataFrame(recommendations)

# Save Recommendations and Backtest Results to PostgreSQL
def save_to_postgresql(symbol_data):
    conn = get_db_connection()
    if conn is None:
        print("Skipping database save due to connection issues.")
        return

    try:
        cur = conn.cursor()
        for data in symbol_data:
            symbol = data['symbol']
            recommendations = data['recommendations']
            backtest = data['backtest_results']
            skipped_trades = data['skipped_trades']
            live_spot_price = data['live_spot_price']
            timestamp = data['timestamp']

            # Insert Recommendations
            rec_records = recommendations.to_dict(orient='records')
            rec_values = [
                (
                    symbol,
                    rec['Strike Price'],
                    rec['Type'],
                    rec['LTP'],
                    rec['Recommendation'],
                    rec['Stop Loss'],
                    rec['Target'],
                    live_spot_price,
                    timestamp
                )
                for rec in rec_records
            ]

            rec_query = """
                INSERT INTO recommendations 
                (symbol, strike_price, option_type, ltp, recommendation, stop_loss, target, live_spot_price, timestamp)
                VALUES %s
            """
            execute_values(cur, rec_query, rec_values)

            # Insert Backtest Results
            bt_records = backtest['trades']
            skipped = skipped_trades

            bt_query = """
                INSERT INTO backtest_results 
                (symbol, final_capital, trades, skipped_trades, timestamp)
                VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(bt_query, (
                symbol,
                backtest['final_capital'],
                json.dumps(bt_records),
                json.dumps(skipped),
                timestamp
            ))

        conn.commit()
        cur.close()
        print("Data successfully saved to PostgreSQL database.")
    except Exception as e:
        print(f"Error saving data to PostgreSQL: {e}")
        conn.rollback()
    finally:
        conn.close()

# Save recommendations to a PostgreSQL database
def save_recommendations_to_db(symbol_data):
    try:
        save_to_postgresql(symbol_data)
    except Exception as e:
        print(f"Failed to save recommendations to PostgreSQL: {e}")

# Restrict Execution to Market Hours (9:00 AM to 3:30 PM IST)
def is_market_open():
    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(india_tz)
        current_time = now.time()
        market_open = datetime.strptime("09:00", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        return market_open <= current_time <= market_close
    except Exception as e:
        print(f"Error checking market hours: {e}")
        return False

# Display All Recommendations (Current and Hourly) in a Single HTML File
def save_recommendations_to_html(symbol_data, filename):
    conn = get_db_connection()
    if conn is None:
        print("Cannot fetch data for HTML display due to database connection issues.")
        return

    try:
        cur = conn.cursor()

        # Calculate the time window for the past hour
        india_tz = pytz.timezone('Asia/Kolkata')
        end_time = datetime.now(india_tz)
        start_time = end_time - timedelta(hours=1)

        # Fetch recommendations within the last hour
        query = """
            SELECT symbol, strike_price, option_type, ltp, recommendation, stop_loss, target, live_spot_price, timestamp
            FROM recommendations
            WHERE timestamp BETWEEN %s AND %s
            ORDER BY timestamp DESC
        """
        cur.execute(query, (start_time, end_time))
        records = cur.fetchall()

        # Convert records to DataFrame
        columns = ['Symbol', 'Strike Price', 'Option Type', 'LTP', 'Recommendation', 'Stop Loss', 'Target', 'Live Spot Price', 'Timestamp']
        hourly_df = pd.DataFrame(records, columns=columns)

        # Start building HTML content
        html_content = ""

        # Add Current Recommendations and Backtest Results
        for data in symbol_data:
            # Recommendations
            recommendations_html = data["recommendations"].to_html(
                index=False,
                border=0,
                justify="center",
                classes="recommendations"
            )
            # Backtest Results
            trades_html = "".join(f"<li>Type: {trade['Type']}, Strike: {trade['Strike']}, Profit: ₹{trade['Profit']:.2f}</li>" for trade in data["backtest_results"]["trades"])
            skipped_html = "".join(f"<li>{reason}</li>" for reason in data["skipped_trades"])

            # Append the HTML section for the current symbol
            html_content += f"""
            <section>
                <h2>Recommendations and Backtesting for {data['symbol']}</h2>
                <p><strong>Live Spot Price:</strong> ₹{data['live_spot_price']:.2f}</p>
                <p><strong>Timestamp:</strong> {data['timestamp']}</p>

                <h3>Recommendations</h3>
                {recommendations_html}

                <h3>Backtesting Results</h3>
                <p><strong>Final Capital:</strong> ₹{data['backtest_results']['final_capital']:.2f}</p>
                <p><strong>Trades Executed:</strong> {len(data['backtest_results']['trades'])}</p>
                <ul>
                    {trades_html}
                </ul>

                <h3>Skipped Trades</h3>
                <ul>
                    {skipped_html}
                </ul>
            </section>
            <hr>
            """

        # Add Hourly Recommendations
        html_content += f"""
        <section>
            <h2>Hourly Recommendations (Last Hour)</h2>
        """

        if not hourly_df.empty:
            hourly_recommendations_html = hourly_df.to_html(
                index=False,
                border=0,
                justify="center",
                classes="recommendations"
            )
            html_content += f"""
                {hourly_recommendations_html}
            """
        else:
            html_content += f"""
                <p>No recommendations found in the past hour.</p>
            """

        html_content += """
        </section>
        <hr>
        """

        # Write the combined HTML file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Combined Recommendations and Backtesting</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #f9f9f9;
                        color: #333;
                    }}
                    h1, h2, h3 {{
                        text-align: center;
                        color: #555;
                    }}
                    .recommendations {{
                        border-collapse: collapse;
                        width: 90%;
                        margin: auto;
                        background-color: #fff;
                    }}
                    .recommendations th {{
                        background-color: #4CAF50;
                        color: white;
                        font-weight: bold;
                        padding: 10px;
                    }}
                    .recommendations td {{
                        border: 1px solid #ddd;
                        padding: 10px;
                        text-align: center;
                    }}
                    .recommendations tr:nth-child(even) {{
                        background-color: #f2f2f2;
                    }}
                    .recommendations tr:hover {{
                        background-color: #ddd;
                    }}
                    ul {{
                        width: 80%;
                        margin: auto;
                        list-style: none;
                        padding: 0;
                    }}
                    ul li {{
                        background-color: #f2f2f2;
                        margin: 5px;
                        padding: 10px;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                    }}
                    hr {{
                        margin: 40px 0;
                        border: none;
                        border-top: 1px solid #ccc;
                    }}
                </style>
            </head>
            <body>
                <h1>Combined Recommendations and Backtesting Results</h1>
                {html_content}
            </body>
            </html>
            """)
        print(f"Combined recommendations and backtest results saved to {filename}")
        cur.close()
    except Exception as e:
        print(f"Error generating combined HTML: {e}")
    finally:
        conn.close()

# Backtest Recommendations including Straddle Logic
def backtest_recommendations(recommendations, initial_capital=100000, transaction_cost=0.0005, slippage=0.001):
    """
    Backtest recommendations with realistic costs and slippage, including Straddle strategies.
    """
    capital = initial_capital
    pnl = 0
    trades = []
    reasons_skipped = []

    for _, rec in recommendations.iterrows():
        recommendation = rec['Recommendation']
        strike_price = rec['Strike Price']
        option_type = rec['Type']
        ltp = rec['LTP']
        target = rec['Target']
        stop_loss = rec['Stop Loss']

        if option_type == "Straddle":
            # Handle Straddle: Buy CE and PE
            # Entry: Buy CE and PE
            ce_ltp = rec.get('CE_LTP', None)
            pe_ltp = rec.get('PE_LTP', None)
            if pd.isna(ce_ltp) or pd.isna(pe_ltp):
                reasons_skipped.append(f"Skipped {recommendation} at {strike_price} due to missing CE or PE LTP.")
                continue

            entry_price_ce = ce_ltp * (1 + slippage)
            entry_price_pe = pe_ltp * (1 + slippage)
            total_entry_price = entry_price_ce + entry_price_pe

            lot_size_ce = min(capital // (entry_price_ce * 25), 1)
            lot_size_pe = min(capital // (entry_price_pe * 25), 1)

            if lot_size_ce == 0 or lot_size_pe == 0:
                reasons_skipped.append(f"Skipped {recommendation} at {strike_price} due to insufficient capital.")
                continue

            cost_ce = entry_price_ce * lot_size_ce * 25
            cost_pe = entry_price_pe * lot_size_pe * 25
            total_cost = (cost_ce + cost_pe) * (1 + transaction_cost)
            capital -= total_cost

            # Exit: Target and Stop Loss for Straddle
            # For simplicity, assume both legs reach target or stop loss
            # In reality, legs might have different outcomes
            exit_price_ce = ce_ltp * 1.2  # Target for CE
            exit_price_pe = pe_ltp * 1.2  # Target for PE
            total_exit_price = exit_price_ce + exit_price_pe
            profit_ce = (exit_price_ce - entry_price_ce) * lot_size_ce * 25
            profit_pe = (exit_price_pe - entry_price_pe) * lot_size_pe * 25
            total_profit = profit_ce + profit_pe

            pnl += total_profit
            capital += total_profit

            trades.append({
                "Type": recommendation,
                "Strike": strike_price,
                "Profit": total_profit
            })

        else:
            # Handle Directional Trades: Buy CE or Buy PE or Sell CE or Sell PE
            entry_price = ltp * (1 + slippage if "Buy" in recommendation else 1 - slippage)
            lot_size = min(capital // (entry_price * 25), 1)  # Adjust lot size
            if lot_size == 0:
                reasons_skipped.append(f"Skipped {recommendation} at {strike_price} due to insufficient capital.")
                continue

            cost = entry_price * lot_size * 25
            capital -= cost * (1 + transaction_cost)

            if "Buy" in recommendation:
                exit_price = target
                profit = (exit_price - entry_price) * lot_size * 25
            elif "Sell" in recommendation:
                exit_price = stop_loss
                profit = (entry_price - exit_price) * lot_size * 25
            else:
                # Undefined recommendation type
                reasons_skipped.append(f"Skipped {recommendation} at {strike_price} due to undefined recommendation type.")
                capital += cost * (1 + transaction_cost)  # Revert capital
                continue

            pnl += profit
            capital += profit
            trades.append({
                "Type": recommendation,
                "Strike": strike_price,
                "Profit": profit
            })

    return capital, trades, reasons_skipped

# Main Task to Run Periodically
def run_task():
    if not is_market_open():
        print("Market is closed. Task will not run.")
        return

    # Main logic
    today = datetime.today()
    start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    symbols = {
        "BANKNIFTY": "^NSEBANK",
        "NIFTY": "^NSEI"
    }

    combined_data = []

    for symbol, yahoo_ticker in symbols.items():
        print(f"\nProcessing {symbol}...")
        index_data = fetch_index_data_yf(yahoo_ticker, start_date, end_date)

        if index_data.empty:
            print(f"No index data available for {symbol}.")
            continue

        live_spot_price = index_data['Close'].iloc[-1]
        india_tz = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(india_tz).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Live Spot Price for {symbol}: {live_spot_price}")

        options_data = fetch_options_chain(symbol)
        if options_data.empty:
            print(f"No options data available for {symbol}.")
            continue

        recommendations = generate_combined_recommendations(index_data, options_data, live_spot_price)
        print(f"\nActionable Recommendations for {symbol}:")
        print(recommendations)

        if recommendations.empty:
            print(f"No actionable recommendations for {symbol}.")
            continue

        final_capital, trades, reasons_skipped = backtest_recommendations(recommendations)
        backtest_results = {
            "final_capital": final_capital,
            "trades": trades,
        }

        combined_data.append({
            "symbol": symbol,
            "recommendations": recommendations,
            "backtest_results": backtest_results,
            "skipped_trades": reasons_skipped,
            "live_spot_price": live_spot_price,
            "timestamp": timestamp
        })

    if combined_data:
        # Save to PostgreSQL
        save_recommendations_to_db(combined_data)
        # Save to HTML (including hourly recommendations)
        save_recommendations_to_html(combined_data, "combined_recommendations.html")
    else:
        print("No data to save.")

# Schedule the script to run every 1 minute within market hours
schedule.every(1).minutes.do(run_task)

# Run the scheduler
if __name__ == "__main__":
    print("Scheduler started. The task will run every 1 minute during market hours (9:00 AM to 3:30 PM IST).")
    run_task()  # Run immediately at start
    while True:
        schedule.run_pending()
        time.sleep(1)
