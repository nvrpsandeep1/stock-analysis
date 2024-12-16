from flask import Flask, render_template
import psycopg2
from stock_analysis import analyze_stocks

app = Flask(__name__)

# Database Configuration
DATABASE_CONFIG = {
    "host": "your-rds-endpoint.amazonaws.com",  # RDS Endpoint
    "database": "postgres",
    "user": "postgres",
    "password": "your-password",
    "port": "5432"
}

# Function to save results to PostgreSQL
def save_to_db(results):
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_analysis (
            stock_name TEXT,
            date DATE,
            close FLOAT,
            ma20 FLOAT,
            upper_band FLOAT,
            lower_band FLOAT,
            rsi FLOAT,
            macd FLOAT,
            signal_line FLOAT,
            suggestion TEXT
        )
        """)
        for stock, data in results.items():
            for i, row in data.iterrows():
                cursor.execute("""
                INSERT INTO stock_analysis (stock_name, date, close, ma20, upper_band, lower_band, rsi, macd, signal_line, suggestion)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (stock, i, row['Close'], row['MA20'], row['Upper Band'], row['Lower Band'], row['RSI'], row['MACD'], row['Signal Line'], row['Suggestion']))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Error:", e)

# Flask route for the homepage
@app.route('/')
def index():
    results = analyze_stocks()  # Analyze stocks
    save_to_db(results)        # Save results to PostgreSQL
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
