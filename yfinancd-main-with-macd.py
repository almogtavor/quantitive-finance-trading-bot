import yfinance as yf
import numpy as np
from datetime import datetime
from collections import deque  # For efficient rolling list
import pandas as pd

# Constants for strategy
INITIAL_MONEY = 10000
FEE_PER_TRADE = 7

# Constants for buying conditions
ONE_HOUR_MA_WINDOW = 60
FIVE_MINUTE_MA_WINDOW = 5
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 50

# Constants for selling conditions
SELL_THRESHOLD = 0.005  # Increased threshold for 5-minute MA trending down
RSI_SELL_THRESHOLD = 90

# Constants for stop loss management
STOP_LOSS_PERCENTAGE_OF_CURRENT_PRICE = 0.98
PROFIT_THRESHOLD_FOR_STOP_LOSS = 0.05  # Increase stop loss when profit reaches 5%
BASE_PROFIT_THRESHOLD = 0.05  # Base profit threshold to start increasing stop loss
PROFIT_INCREASE_INTERVAL = 0.02  # Additional profit percentage to trigger the next stop loss update
INCREASE_STOP_LOSS_PERCENTAGE = 0.02  # Percentage to increase stop loss from the current price

# Set up a rolling window for RSI values (e.g., the last 14 values)
RSI_ROLLING_PERIOD = 14
last_rsi_values = deque(maxlen=RSI_ROLLING_PERIOD)

# Constants for MACD calculation
SHORT_TERM_EMA = 12
LONG_TERM_EMA = 26
SIGNAL_LINE_EMA = 9

# Constants for Bollinger Bands calculation
BOLLINGER_BAND_WINDOW = 20
BOLLINGER_BAND_STD_MULTIPLIER = 2

def calculate_bollinger_bands(prices):
    prices_series = pd.Series(prices)
    sma = prices_series.rolling(window=BOLLINGER_BAND_WINDOW).mean()
    std = prices_series.rolling(window=BOLLINGER_BAND_WINDOW).std()
    upper_band = sma + (std * BOLLINGER_BAND_STD_MULTIPLIER)
    lower_band = sma - (std * BOLLINGER_BAND_STD_MULTIPLIER)
    return upper_band, lower_band

def calculate_ema(prices, window):
    prices_series = pd.Series(prices)
    return prices_series.ewm(span=window, adjust=False).mean()

def calculate_macd(prices):
    prices_series = pd.Series(prices)
    short_ema = calculate_ema(prices_series, SHORT_TERM_EMA)
    long_ema = calculate_ema(prices_series, LONG_TERM_EMA)
    macd = short_ema - long_ema
    signal = macd.ewm(span=SIGNAL_LINE_EMA, adjust=False).mean()
    return macd, signal

def calculate_moving_average(prices, window_size):
    return np.convolve(prices, np.ones(window_size), 'valid') / window_size

def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    avg_gain = np.convolve(gain, np.ones(period), 'valid') / period
    avg_loss = np.convolve(loss, np.ones(period), 'valid') / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([np.full(period, np.nan), rsi])

def should_buy(latest_price, one_hour_ma, five_minute_ma, rsi):
    if len(one_hour_ma) == 0 or len(five_minute_ma) < 2 or np.isnan(rsi[-1]):
        return False
    return latest_price > one_hour_ma[-1] and five_minute_ma[-1] > five_minute_ma[-2] and rsi[-1] < 50

def should_sell(five_minute_ma, stop_loss, current_price, rsi, last_rsi_values, macd, signal, upper_band, volume, index, threshold=0.005):
    if len(five_minute_ma) < 3 or np.isnan(rsi[-1]):  # Ensure sufficient data
        return False

    # MACD Histogram for momentum
    macd_histogram = macd - signal
    histogram_weakening = macd_histogram.iloc[-1] < macd_histogram.iloc[-2]

    # Bearish MACD crossover with additional criteria
    bearish_macd_crossover = macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] > signal.iloc[-2] and histogram_weakening

    # Price hitting upper Bollinger Band with volume confirmation
    current_volume = volumes.iloc[index]  # Access the current volume
    volume_mean = volumes.rolling(window=5).mean().iloc[index]  # Mean of the last 5 volumes
    price_hits_upper_band = current_price > upper_band.iloc[-1] and current_volume > volume_mean

    # Other sell conditions
    dynamic_rsi_threshold = max(RSI_SELL_THRESHOLD, np.percentile(last_rsi_values, 80)) if last_rsi_values else RSI_SELL_THRESHOLD
    price_below_stop_loss = current_price <= stop_loss
    ma_trending_down = five_minute_ma[-1] < five_minute_ma[-3] * (1 - threshold)
    rsi_overbought = rsi[-1] > dynamic_rsi_threshold

    if bearish_macd_crossover and price_hits_upper_band:
        print(f"SELL reason: Bearish MACD crossover detected & Price hitting the upper Bollinger Band.")
    if price_below_stop_loss:
        print(f"SELL reason: Price below stop loss.")
    elif ma_trending_down:
        print(f"SELL reason: 5-minute MA trending down significantly.")
    elif rsi_overbought:
        print(f"SELL reason: RSI overbought. RSI: {rsi[-1]}, Dynamic threshold: {dynamic_rsi_threshold}")

    return price_below_stop_loss or ma_trending_down or rsi_overbought or (bearish_macd_crossover and price_hits_upper_band)


# Function to log buying actions with detailed conditions
def log_buy(timestamp, current_price, one_hour_ma, five_minute_ma, rsi, balance, stock_worth):
    print(f"""{timestamp} - BUY Triggered (price {current_price}, new balance: {balance})
            - 1-hour MA: {one_hour_ma[-1]}
            - 5-minute MA: {five_minute_ma[-1]}
            - RSI: {rsi[-1]}
            - Stocks worth: {stock_worth}""")

# Function to log selling actions with detailed conditions
def log_sell(timestamp, current_price, five_minute_ma, rsi, last_rsi_values, balance, change):
    print(f"""{timestamp} - SELL Triggered: (price {current_price}, new balance: {balance})
        - 5-minute MA change: {(five_minute_ma[-1] - five_minute_ma[-3]) / five_minute_ma[-3]}
        - RSI: {rsi[-1]}
        - Dynamic RSI Threshold: {max(RSI_SELL_THRESHOLD, np.percentile(last_rsi_values, 80)) if last_rsi_values else RSI_SELL_THRESHOLD}
        - Change in balance: {change}""")

def print_balance_status(current_balance: float, timestamp: datetime):
    print(f"{timestamp} - Balance status is {current_balance}\n")

# Main function
if __name__ == "__main__":
    ticker = "TSLA"
    balance = INITIAL_MONEY
    data = yf.download(ticker, start="2023-12-16", end="2023-12-23", interval="1m")

    # Initialization
    prices = data['Close'].tolist()
    volumes = pd.Series(data['Volume'].tolist())
    timestamps = data.index.tolist()
    print_balance_status(balance, timestamps[0])
    in_position = False
    shares_held = 0
    stop_loss = 0
    purchase_price = 0
    # Variable for dynamic stop loss updates
    last_profit_update = 0  # Last profit percentage at which stop loss was updated

    # Iterate through each price point
    for i in range(60, len(prices)):
        current_timestamp = timestamps[i]
        current_price = prices[i]
        one_hour_ma = calculate_moving_average(prices[:i], 60)
        five_minute_ma = calculate_moving_average(prices[:i], 5)
        rsi = calculate_rsi(prices[:i])
        macd, signal = calculate_macd(prices[:i])
        upper_band, _ = calculate_bollinger_bands(prices[:i])

        if not in_position:
            cost_of_buying = current_price + (2 * FEE_PER_TRADE)  # Consider fees for both buying and selling
            if should_buy(current_price, one_hour_ma, five_minute_ma, rsi) and balance >= cost_of_buying:
                print(f"BUY reason: Price above 1-hour MA: {one_hour_ma[-1]}, 5-minute MA trending up, RSI: {rsi[-1]}")
                purchase_price = current_price
                stop_loss = purchase_price * STOP_LOSS_PERCENTAGE_OF_CURRENT_PRICE
                shares_to_buy = (balance - FEE_PER_TRADE) / current_price
                balance -= shares_to_buy * current_price + FEE_PER_TRADE
                shares_held += shares_to_buy
                in_position = True
                stock_worth = shares_held * current_price  # Worth of the stocks (not including fees)
                print(
                    f"{current_timestamp} - BUY: Buying at {current_price}, stock worth: {stock_worth}, new balance: {balance}")
                print_balance_status(balance, current_timestamp)
                log_buy(current_timestamp, current_price, one_hour_ma, five_minute_ma, rsi, balance, stock_worth)

        elif in_position:
            # Calculate current profit percentage
            current_profit_percentage = (current_price - purchase_price) / purchase_price

            # Check if the current profit percentage has reached the next update threshold
            if current_profit_percentage >= BASE_PROFIT_THRESHOLD + last_profit_update:
                new_stop_loss = current_price * (1 - INCREASE_STOP_LOSS_PERCENTAGE)
                if new_stop_loss > stop_loss:  # Ensure we only move the stop loss up
                    stop_loss = new_stop_loss
                    last_profit_update += PROFIT_INCREASE_INTERVAL  # Update the threshold for the next increase
                    print(f"{current_timestamp} - UPDATE: Stop loss increased to {stop_loss}")

            # Sell based on the updated stop loss or other selling conditions
            if should_sell(five_minute_ma, stop_loss, current_price, rsi, last_rsi_values, macd, signal, upper_band,
                           volumes, i, SELL_THRESHOLD):
                sell_value = shares_held * current_price
                change = sell_value - (shares_held * purchase_price) - 2 * FEE_PER_TRADE
                balance += sell_value - FEE_PER_TRADE
                print(
                    f"{current_timestamp} - SELL: Selling at {current_price}, change: {change}, new balance: {balance}")
                shares_held = 0
                in_position = False
                print_balance_status(balance, current_timestamp)
                log_sell(current_timestamp, current_price, five_minute_ma, rsi, last_rsi_values, balance, change)

    # End of data, sell any remaining shares
    if in_position:
        sell_value = shares_held * current_price
        change = sell_value - (shares_held * purchase_price) - 2 * FEE_PER_TRADE
        balance += sell_value - FEE_PER_TRADE
        print(f"{current_timestamp} - END: Selling at {current_price}, change: {change}, final balance: {balance}")
        print_balance_status(balance, current_timestamp)
    else:
        print(f"{current_timestamp} - END: Final balance without position: {balance}")
        print_balance_status(balance, current_timestamp)
