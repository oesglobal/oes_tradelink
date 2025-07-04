import streamlit as st
import pandas as pd
import numpy as np
import time
import ccxt
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
import os # Import os module
import pytz # Import pytz for timezone handling
import datetime # Import datetime for timestamp conversion
import plotly.graph_objects as go # Import Plotly for charting

# --- Configuration (from config.py) ---
try:
    from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_MODE
except ImportError:
    st.error("Error: config.py not found. Please create config.py with BINANCE_API_KEY, BINANCE_SECRET_KEY, and BINANCE_MODE.")
    st.stop() # Stop the app if config is missing

# --- Constants ---
MODEL_PATH = "btc_model.h5" # This model is trained on BTC/USDT. For other pairs, you'd need separate models.
SCALER_PATH = "btc_scaler.pkl" # This scaler is trained on BTC/USDT. For other pairs, you'd need separate scalers.
SEQUENCE_LENGTH = 20 # Must match the sequence length used during model training

# --- Streamlit UI Initialization ---
st.set_page_config(layout="wide", page_title="Crypto Auto-Trader", page_icon="🚀")

st.header("🚀 oes_tradelink Crypto Auto-Trader with LSTM Prediction")

# Initialize session state for trading status and messages
if 'auto_trade_started' not in st.session_state:
    st.session_state['auto_trade_started'] = False
if 'stop_flag' not in st.session_state:
    st.session_state['stop_flag'] = False
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = "N/A"
if 'last_prediction_proba' not in st.session_state:
    st.session_state['last_prediction_proba'] = "N/A"
if 'last_trade_type' not in st.session_state:
    st.session_state['last_trade_type'] = "N/A"
if 'last_trade_amount' not in st.session_state:
    st.session_state['last_trade_amount'] = "N/A"
if 'last_trade_time' not in st.session_state:
    st.session_state['last_trade_time'] = "N/A"
if 'open_position' not in st.session_state:
    st.session_state['open_position'] = False # True if a buy trade is open, False if no position or sold
if 'position_entry_price' not in st.session_state:
    st.session_state['position_entry_price'] = None
if 'position_type' not in st.session_state: # 'long' or 'short' (though model only predicts 'up')
    st.session_state['position_type'] = None
if 'trading_messages' not in st.session_state:
    st.session_state['trading_messages'] = []
if 'trade_mode' not in st.session_state:
    st.session_state['trade_mode'] = 'Futures' # Default to Futures

# --- Message Log Update Function (separated from rendering) ---
def update_log(message, level="info"):
    """Updates the Streamlit log with a timestamped message."""
    timestamp = datetime.datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.session_state.trading_messages.insert(0, f"{timestamp} [{level.upper()}] {message}")
    # Keep log short
    st.session_state.trading_messages = st.session_state.trading_messages[:20] # Keep last 20 messages


def stop_auto_trade(reason="user"):
    st.session_state['auto_trade_started'] = False
    st.session_state['stop_flag'] = True # Set a flag to signal the loop to stop
    update_log(f"Auto-trading is STOPPED by {reason}.", level="warning")

# --- Load Model and Scaler (Modified to not call st. functions directly) ---
@st.cache_resource
def load_ml_assets_cached(model_path, scaler_path):
    """Loads the pre-trained Keras model and MinMaxScaler. Returns (model, scaler, success_flag, error_message)."""
    try:
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, True, None
    except Exception as e:
        error_msg = f"Failed to load ML assets: {e}"
        return None, None, False, error_msg

model, scaler, ml_assets_loaded_success, ml_assets_error_msg = load_ml_assets_cached(MODEL_PATH, SCALER_PATH)

if ml_assets_loaded_success:
    update_log("Model and scaler loaded successfully!", level="info")
else:
    update_log(ml_assets_error_msg, level="error")
    update_log(f"Model or scaler could not be loaded. Please ensure '{os.path.basename(MODEL_PATH)}' and '{os.path.basename(SCALER_PATH)}' are in the same directory as app.py.", level="warning")
    st.stop() # Stop the app if model/scaler loading failed


# --- Binance Client Initialization (Modified to not call st. functions directly) ---
@st.cache_resource
def initialize_binance_client_cached(api_key, secret_key, mode, default_type):
    """Initializes and returns a Binance client for Spot or Futures. Returns (exchange_client, success_flag, error_message)."""
    options = {
        'defaultType': default_type, # 'spot' or 'future'
        'createMarketBuyOrderRequiresPrice': False # Needed for some Futures exchanges
    }
    try:
        if mode == "testnet":
            if default_type == 'future':
                urls = {
                    'api': 'https://testnet.binancefuture.com/fapi/v1',
                    'fetchMyTrades': 'https://testnet.binancefuture.com/fapi/v1',
                    'private': 'https://testnet.binancefuture.com/fapi/v1',
                }
            else: # spot testnet
                urls = {
                    'api': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api',
                }
            
            exchange_client = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
                'options': options,
                'urls': urls
            })
        elif mode == "mainnet":
            exchange_client = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
                'options': options,
            })
        else:
            return None, False, "Invalid BINANCE_MODE in config.py. Must be 'testnet' or 'mainnet'."
        
        return exchange_client, True, None
    except Exception as e:
        return None, False, f"Error initializing Binance client for {default_type} in {mode} mode: {e}"

exchange_spot_client, spot_client_success, spot_client_error_msg = initialize_binance_client_cached(BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_MODE, 'spot')
exchange_futures_client, futures_client_success, futures_client_error_msg = initialize_binance_client_cached(BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_MODE, 'future')

if not spot_client_success:
    update_log(spot_client_error_msg, level="error")
    st.stop()
if not futures_client_success:
    update_log(futures_client_error_msg, level="error")
    st.stop()

# --- Utility Functions ---
def get_current_exchange():
    """Returns the currently selected CCXT exchange instance."""
    if st.session_state['trade_mode'] == 'Spot':
        return exchange_spot_client
    else: # Futures
        return exchange_futures_client

def get_balance(coin='USDT'):
    """Fetches the available balance for a given coin based on current trade mode."""
    try:
        exchange = get_current_exchange()
        balance_info = exchange.fetch_balance()
        return balance_info['free'].get(coin, 0.0)
    except Exception as e:
        update_log(f"Error fetching balance for {st.session_state['trade_mode']} {coin}: {e}", level="error")
        return 0.0

def fetch_ohlcv(symbol, timeframe, limit):
    """Fetches OHLCV data from Binance using the appropriate exchange."""
    try:
        exchange = get_current_exchange()
        ccxt_symbol = symbol.replace('/', '') # e.g., BTCUSDT
        klines = exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float) # Ensure numerical columns are float
        return df
    except Exception as e:
        update_log(f"Error fetching OHLCV data for {symbol} ({st.session_state['trade_mode']}): {e}", level="error")
        return pd.DataFrame()

def preprocess_data(df, scaler, sequence_length):
    """
    Applies feature engineering and scaling to the DataFrame for model prediction.
    Features MUST match those used during model training.
    """
    if df.empty:
        return np.array([]), None

    # Calculate target and drop last row first
    df['next_close'] = df['close'].shift(-1)
    df['target'] = (df['next_close'] > df['close']).astype(int)
    df.dropna(inplace=True) # Drop the last row which has NaN for 'next_close' and 'target'

    # Technical Indicators - Must match the training notebook
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['close'])
    df['BBL'] = ta.volatility.bollinger_lband(df['close'])
    df['BBM'] = ta.volatility.bollinger_mavg(df['close'])
    df['BBH'] = ta.volatility.bollinger_hband(df['close'])
    df['BB_bandwidth'] = ta.volatility.bollinger_wband(df['close'])
    df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['price_change_pct'] = df['close'].pct_change() * 100
    df['volume_SMA_5'] = df['volume'].rolling(window=5).mean()

    # --- NEW INDICATORS - MUST MATCH THE COLAB NOTEBOOK ---
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['Stoch_K'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    # --- END NEW INDICATORS ---

    df.dropna(inplace=True) # Drop NaNs from indicator calculations

    # Ensure all features are numeric and handle any remaining infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fill any remaining NaNs with the mean of the column (should be few if any after dropna)
    df.fillna(df.mean(), inplace=True)

    # Define features - MUST match the training notebook EXACTLY, including order
    features = [
        'open', 'high', 'low', 'close', 'volume',
        'RSI', 'MACD', 'BBL', 'BBM', 'BBH', 'BB_bandwidth',
        'EMA_12', 'EMA_26', 'price_change_pct', 'volume_SMA_5',
        'ATR', 'Stoch_K', 'OBV' # Include new indicators
    ]

    # Get the latest row for prediction
    latest_data_point = df[features].iloc[-1:].values

    # Scale the latest data point
    latest_data_scaled = scaler.transform(latest_data_point)

    # Create sequence for the latest data point
    # For live prediction, we need the last `sequence_length` data points
    # If the df is shorter than sequence_length after dropna, we can't form a sequence
    if len(df) < sequence_length:
        update_log(f"Not enough historical data ({len(df)} points) to create a sequence of length {sequence_length}.", level="warning")
        return np.array([]), None

    # Take the last 'sequence_length' rows for prediction
    sequence = df[features].tail(sequence_length).values
    sequence_scaled = scaler.transform(sequence)
    # Reshape for LSTM input: (1, sequence_length, num_features)
    X_predict = sequence_scaled.reshape(1, sequence_length, len(features))

    return X_predict, df['target'].iloc[-1] # Return X for prediction and the actual target of the latest candle

def make_prediction(X_predict):
    """Makes a prediction using the loaded model."""
    if X_predict is None or X_predict.size == 0:
        return None, None
    prediction_proba = model.predict(X_predict, verbose=0)[0][0] # verbose=0 to suppress Keras output
    return (prediction_proba > 0.5).astype(int), prediction_proba # Return binary prediction and probability

def execute_trade(symbol, trade_type, order_type, amount_usdt, leverage=None, limit_price=None, stop_price=None, take_profit_price=None):
    """Executes a trade (buy/sell) with specified order type and leverage (if Futures)."""
    current_exchange = get_current_exchange()
    mode = st.session_state['trade_mode']

    try:
        if mode == 'Futures':
            if leverage is None:
                raise ValueError("Leverage must be provided for Futures trades.")
            # Set leverage first (Futures specific)
            ccxt_symbol_no_slash = symbol.replace('/', '')
            current_exchange.set_leverage(leverage, ccxt_symbol_no_slash)
            update_log(f"Leverage set to {leverage}x for {symbol} ({mode}).", level="info")

        current_price = get_current_price(symbol)
        if current_price is None:
            raise Exception("Could not get current price to calculate trade quantity.")

        if amount_usdt <= 0:
            raise ValueError("Trade amount must be greater than zero.")

        # Calculate quantity based on USDT amount and current price
        quantity = amount_usdt / current_price

        # Fetch market limits (min_amount, precision for quantity)
        market = current_exchange.market(symbol)
        if market and 'limits' in market and 'amount' in market['limits']:
            min_quantity = market['limits']['amount']['min']
            
            # Adjust quantity to market precision
            quantity = current_exchange.amount_to_precision(symbol, quantity)

            if quantity < min_quantity:
                raise ValueError(f"Calculated quantity {quantity:.8f} is less than minimum allowed {min_quantity:.8f} for {symbol}. Increase Trade Amount (USDT).")
        else:
            update_log(f"Warning: Could not fetch market limits for {symbol}. Proceeding without precision check.", level="warning")

        params = {}
        if take_profit_price:
            params['takeProfitPrice'] = current_exchange.price_to_precision(symbol, take_profit_price)
        if stop_price:
            params['stopLossPrice'] = current_exchange.price_to_precision(symbol, stop_price)

        order = None
        if order_type == 'market':
            order = current_exchange.create_market_order(symbol, trade_type, quantity, params)
        elif order_type == 'limit':
            if limit_price is None or limit_price <= 0:
                raise ValueError("Limit price must be provided and positive for limit orders.")
            
            # Adjust limit_price to market precision
            price_precision = market['precision']['price'] if market and 'precision' in market else None
            if price_precision:
                limit_price = current_exchange.price_to_precision(symbol, limit_price)

            order = current_exchange.create_limit_order(symbol, trade_type, quantity, limit_price, params)
        elif order_type == 'stop-loss': # This will create a Stop Market order
            if stop_price is None or stop_price <= 0:
                raise ValueError("Stop price must be provided and positive for Stop-Loss orders.")
            # For simplicity, using create_stop_market_order. Some exchanges might use create_order with params.
            order = current_exchange.create_stop_market_order(symbol, trade_type, quantity, stop_price, params)
        elif order_type == 'take-profit': # This will create a Take Profit Market order
            if take_profit_price is None or take_profit_price <= 0:
                raise ValueError("Take Profit price must be provided and positive for Take-Profit orders.")
            # For simplicity, using create_take_profit_market_order. Some exchanges might use create_order with params.
            order = current_exchange.create_take_profit_market_order(symbol, trade_type, quantity, take_profit_price, params)

        
        st.success(f"Trade executed: {mode} {order_type.upper()} {trade_type.upper()} {quantity:.4f} {symbol.split('/')[0]} ({amount_usdt:.2f} USDT equivalent)")
        st.json(order)
        st.session_state['last_trade_type'] = trade_type
        st.session_state['last_trade_amount'] = amount_usdt # Store USDT amount for consistency
        st.session_state['last_trade_time'] = datetime.datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Position tracking logic (simplified for auto-trader's single long position)
        # This part needs more robust handling for complex manual trades
        if trade_type == 'buy':
            st.session_state['open_position'] = True
            st.session_state['position_entry_price'] = current_price
            st.session_state['position_type'] = 'long'
        elif trade_type == 'sell' and st.session_state['open_position']:
            # Assume a sell closes the existing long position initiated by auto-trader or manual buy
            st.session_state['open_position'] = False
            st.session_state['position_entry_price'] = None
            st.session_state['position_type'] = None
        
        return True
    except ccxt.InsufficientFunds as e:
        update_log(f"Insufficient funds to execute {trade_type} trade: {e}", level="error")
        return False
    except ValueError as e:
        update_log(f"Trade parameter error: {e}", level="error")
        return False
    except Exception as e:
        update_log(f"Error executing trade: {e}", level="error")
        return False

def get_current_price(symbol):
    """Fetches the current market price of the symbol."""
    try:
        exchange = get_current_exchange()
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        update_log(f"Error fetching current price for {symbol} ({st.session_state['trade_mode']}): {e}", level="error")
        return None

# --- Sidebar for Configuration ---
st.sidebar.header("Trading Configuration")

# Moved USDT Balance to top of sidebar
current_balance = get_balance('USDT')
st.sidebar.metric(label=f"{st.session_state['trade_mode']} USDT Balance", value=f"{current_balance:.2f} USDT")

# Trade Mode Selector (Spot/Futures)
st.sidebar.markdown("---")
st.sidebar.subheader("Global Trade Mode")
st.session_state['trade_mode'] = st.sidebar.radio(
    "Select Trading Mode",
    ('Futures', 'Spot'),
    key='global_trade_mode',
    on_change=lambda: [
        st.session_state.update({
            'auto_trade_started': False,
            'stop_flag': True,
            'open_position': False,
            'position_entry_price': None,
            'position_type': None
        }),
        update_log(f"Trading mode switched to {st.session_state['trade_mode']}. Auto-trading stopped and position reset.", level="warning")
    ]
)
st.sidebar.markdown("---")

# Define selected_symbol here, globally in the sidebar before any trade logic uses it
with st.sidebar.expander("Coin Pairs"):
    selected_symbol = st.selectbox("Select Coin Pairs", ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', '1000PEPE/USDT'])

# Main Trade Expander for Auto and Manual Trade Sections
with st.sidebar.expander("Trade", expanded=True):

    st.subheader("Auto-Trade")
    # New input for auto-trade amount
    auto_trade_amount_usdt = st.number_input("Auto-Trade Amount (USDT)", min_value=10.0, value=50.0, step=10.0, format="%.2f", key='auto_trade_amount_usdt_input')

    # Moved Start/Stop Auto-Trade Buttons to sidebar
    col_start, col_stop = st.columns(2) # Using st.columns directly inside expander
    with col_start:
        if st.button("Start Auto-Trade", disabled=st.session_state['auto_trade_started']):
            if model is None or scaler is None:
                update_log("Cannot start auto-trade: Model or scaler not loaded.", level="error")
            else:
                st.session_state['auto_trade_started'] = True
                st.session_state['stop_flag'] = False
                st.session_state['trading_messages'] = [] # Clear previous log on start
                update_log("Auto-trading started!", level="info")

    with col_stop:
        if st.button("Stop Auto-Trade", disabled=not st.session_state['auto_trade_started']):
            stop_auto_trade()

    st.markdown("---")
    
    st.subheader("Manual Trade")
    manual_trade_type = st.radio("Trade Type", ('BUY', 'SELL'), horizontal=True, key='manual_trade_type_radio')
    manual_order_type = st.radio("Order Type", ('Market', 'Limit', 'Stop-Loss', 'Take-Profit'), horizontal=True, key='manual_order_type_radio')

    manual_limit_price = None
    manual_stop_price = None
    manual_take_profit_price = None

    if manual_order_type == 'Limit':
        manual_limit_price = st.number_input("Limit Price", min_value=0.0, format="%.4f", key='manual_limit_price_input')
    elif manual_order_type == 'Stop-Loss':
        manual_stop_price = st.number_input("Stop Price (Trigger)", min_value=0.0, format="%.4f", key='manual_stop_price_input')
    elif manual_order_type == 'Take-Profit':
        manual_take_profit_price = st.number_input("Take Profit Price (Trigger)", min_value=0.0, format="%.4f", key='manual_take_profit_price_input')


    manual_trade_amount_usdt = st.number_input("Trade Amount (USDT)", min_value=10.0, value=50.0, step=10.0, format="%.2f", key='manual_trade_amount_usdt_input')
    
    manual_leverage = None
    if st.session_state['trade_mode'] == 'Futures':
        manual_leverage = st.selectbox("Leverage", [1, 5, 10, 20, 50, 100], index=2, key='manual_leverage_select') # Default to 10x

    if st.button(f"Execute Manual {manual_trade_type} {manual_order_type} Order", key='execute_manual_trade_button'):
        # Basic validation for prices
        if manual_order_type == 'Limit' and (manual_limit_price is None or manual_limit_price <= 0):
            st.error("Please enter a valid Limit Price for Limit orders.")
        elif manual_order_type == 'Stop-Loss' and (manual_stop_price is None or manual_stop_price <= 0):
            st.error("Please enter a valid Stop Price for Stop-Loss orders.")
        elif manual_order_type == 'Take-Profit' and (manual_take_profit_price is None or manual_take_profit_price <= 0):
            st.error("Please enter a valid Take Profit Price for Take-Profit orders.")
        else:
            update_log(f"Attempting manual {manual_trade_type} {manual_order_type} order for {selected_symbol} ({st.session_state['trade_mode']})...", level="info")
            execute_trade(
                symbol=selected_symbol,
                trade_type=manual_trade_type.lower(),
                order_type=manual_order_type.lower(),
                amount_usdt=manual_trade_amount_usdt,
                leverage=manual_leverage,
                limit_price=manual_limit_price,
                stop_price=manual_stop_price,
                take_profit_price=manual_take_profit_price
            )

st.sidebar.markdown("---")

with st.sidebar.expander("Timeframe"):
    timeframe_option = st.selectbox("Select Timeframe", ['3m', '5m', '10m', '30m', '1h', '4h', '1d', '1w', '1M'])

with st.sidebar.expander("Historical Candles to Fetch"):
    limit_option = st.number_input("Historical Candles to Fetch", min_value=SEQUENCE_LENGTH + 10, value=500, step=10)

with st.sidebar.expander("Prediction Interval (seconds)"):
    prediction_interval = st.slider("Prediction Interval (seconds)", min_value=60, max_value=3600, value=300, step=60)

with st.sidebar.expander("Take Profit / Stop Loss"):
    take_profit_pct = st.slider("Take Profit (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    stop_loss_pct = st.slider("Stop Loss (%)", min_value=0.1, max_value=10.0, value=0.5, step=0.1, format="%.1f")

# --- Reset App State Button ---
st.sidebar.markdown("---")
if st.sidebar.button("Reset App State", help="Clears all session data and restarts the app."):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun() # Force a full rerun to reset the state


# --- Main Content Area ---

# Warning if not BTC/USDT is selected, as model is specific
if selected_symbol != 'BTC/USDT':
    st.warning(f"⚠️ Warning: The current model (`{MODEL_PATH}`) and scaler (`{SCALER_PATH}`) are trained specifically for **BTC/USDT**. Using them with **{selected_symbol}** will likely result in inaccurate predictions and trades.")

# --- Live Price Chart & Prediction (Moved Up and Formatted) ---
# Removed: st.subheader(f"Live Price Chart for {selected_symbol}") # Main chart title remains

# Placeholder for the prediction and current price details - MOVED HERE
prediction_details_placeholder = st.empty()

# Removed: chart_type = st.radio("Select Chart Type", ('Candlestick', 'Line'), horizontal=True)
chart_placeholder = st.empty() # Placeholder for the actual Plotly chart


st.subheader("Auto-Trading Status")
status_placeholder = st.empty()
position_info_placeholder = st.empty()


# --- Main Loop for UI Updates (Always Running) ---
while True:
    try:
        # 1. Update status and fetch data (always)
        if st.session_state['auto_trade_started']:
            status_placeholder.info(f"Auto-trading is ACTIVE for {selected_symbol} ({timeframe_option}) in **{st.session_state['trade_mode']}** mode. Next prediction in {prediction_interval} seconds.")
        else:
            status_placeholder.error("Auto-trading is STOPPED by user or due to critical error.")

        update_log(f"Fetching latest data for {selected_symbol} ({st.session_state['trade_mode']})...", level="info")
        chart_df = fetch_ohlcv(selected_symbol, timeframe_option, 200)
        latest_df = fetch_ohlcv(selected_symbol, timeframe_option, limit_option)
        current_price = get_current_price(selected_symbol)

        if latest_df.empty or chart_df.empty:
            update_log(f"No data fetched for {selected_symbol}. Retrying...", level="warning")
            chart_placeholder.warning("No chart data available. Retrying...") # Display message on UI
            prediction_details_placeholder.empty() # Clear prediction details
            position_info_placeholder.empty()
            time.sleep(10)
            st.rerun() # Force a rerun to retry fetching
            continue

        # 2. Make prediction (always, if data available)
        binary_prediction, prediction_proba = None, None
        X_predict = np.array([])
        if len(latest_df) >= SEQUENCE_LENGTH:
            X_predict, actual_target_direction = preprocess_data(latest_df.copy(), scaler, SEQUENCE_LENGTH)
            if X_predict.size > 0:
                binary_prediction, prediction_proba = make_prediction(X_predict)
                st.session_state['last_prediction'] = "BUY (UP)" if binary_prediction == 1 else "SELL/HOLD (DOWN/SIDEWAYS)"
                st.session_state['last_prediction_proba'] = f"{prediction_proba:.4f}"
            else:
                st.session_state['last_prediction'] = "N/A"
                st.session_state['last_prediction_proba'] = "N/A"
        else:
            st.session_state['last_prediction'] = "N/A"
            st.session_state['last_prediction_proba'] = "N/A"
            # No prediction info to display if not enough data

        # Update prediction info and current price (always) in the new placeholder
        # This will now be directly under "Live Price Chart for BTC/USDT" and above "Select Chart Type"
        prediction_details_placeholder.markdown(f"""
            Live Price Prediction for {selected_symbol}:
            **{st.session_state['last_prediction']}** (Confidence: {float(st.session_state['last_prediction_proba']):.2f}%)
            Current Price: **{current_price:.4f} USDT**
        """, unsafe_allow_html=True) # Added unsafe_allow_html for <br>


        # 3. Plot chart (always)
        def plot_candlestick_chart(df, symbol, timeframe, chart_placeholder, prediction=None):
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                open=df['open'],
                                                high=df['high'],
                                                low=df['low'],
                                                close=df['close'],
                                                name='Price')])
            if prediction is not None:
                last_candle_time = df.index[-1]
                last_candle_high = df['high'].iloc[-1]
                last_candle_low = df['low'].iloc[-1]
                
                marker_y = last_candle_high * 1.005 if prediction == 1 else last_candle_low * 0.995
                arrow_color = 'green' if prediction == 1 else 'red'
                arrow_symbol = "triangle-up" if prediction == 1 else "triangle-down"
                text_label = "UP" if prediction == 1 else "DOWN"

                fig.add_trace(go.Scatter(x=[last_candle_time], y=[marker_y],
                                        mode='markers',
                                        marker=dict(symbol=arrow_symbol, size=15, color=arrow_color),
                                        name=f'Predicted {text_label}',
                                        hovertemplate=f"Predicted {text_label}<br>Time: {last_candle_time}<br>Price: {df['close'].iloc[-1]:.2f}<extra></extra>"))

            fig.update_layout(title=f'{symbol} Price ({timeframe})',
                              xaxis_rangeslider_visible=False,
                              template="plotly_dark",
                              height=500)
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        plot_candlestick_chart(chart_df, selected_symbol, timeframe_option, chart_placeholder, prediction=binary_prediction)


        # 4. Auto-Trading Logic (conditional on auto_trade_started)
        if st.session_state['auto_trade_started'] and binary_prediction is not None:
            auto_trade_leverage = 1 if st.session_state['trade_mode'] == 'Futures' else None # No leverage for Spot

            if not st.session_state['open_position']: # No open position
                if binary_prediction == 1: # Predicted UP
                    update_log(f"Predicted UP for {selected_symbol}. Attempting to BUY ({st.session_state['trade_mode']})...", level="info")
                    if execute_trade(selected_symbol, 'buy', 'market', auto_trade_amount_usdt, auto_trade_leverage, None, None, None): # Pass None for limit/stop/tp prices
                        update_log(f"BUY order placed for {auto_trade_amount_usdt:.2f} USDT of {selected_symbol} at {current_price:.4f} USDT. Position opened.", level="info")
                else:
                    update_log(f"Predicted DOWN/SIDEWAYS for {selected_symbol}. Waiting for a BUY signal.", level="info")
            else: # Position is open (long position)
                update_log(f"Open position for {selected_symbol} ({st.session_state['trade_mode']}). Checking Take Profit/Stop Loss...", level="info")
                entry_price = st.session_state['position_entry_price']
                if entry_price is not None:
                    price_change = ((current_price - entry_price) / entry_price) * 100
                    position_info_placeholder.markdown(f"""
                        - Open Position: **{st.session_state['position_type'].upper()}** in **{st.session_state['trade_mode']}**
                        - Entry Price: **{entry_price:.4f} USDT**
                        - Current Price: **{current_price:.4f} USDT**
                        - P&L: **{price_change:.2f}%**
                    """)
                    if st.session_state['position_type'] == 'long': # If we bought (long position)
                        if price_change >= take_profit_pct:
                            update_log(f"Take Profit hit! ({price_change:.2f}%). Selling {selected_symbol}...", level="info")
                            if execute_trade(selected_symbol, 'sell', 'market', auto_trade_amount_usdt, auto_trade_leverage, None, None, None):
                                update_log(f"SELL order placed for {auto_trade_amount_usdt:.2f} USDT of {selected_symbol} at {current_price:.4f} USDT. Position closed.", level="info")
                                st.session_state['open_position'] = False
                                st.session_state['position_entry_price'] = None
                                st.session_state['position_type'] = None
                        elif price_change <= -stop_loss_pct:
                            update_log(f"Stop Loss hit! ({price_change:.2f}%). Selling {selected_symbol}...", level="warning")
                            if execute_trade(selected_symbol, 'sell', 'market', auto_trade_amount_usdt, auto_trade_leverage, None, None, None):
                                update_log(f"SELL order placed for {auto_trade_amount_usdt:.2f} USDT of {selected_symbol} at {current_price:.4f} USDT. Position closed.", level="info")
                                st.session_state['open_position'] = False
                                st.session_state['position_entry_price'] = None
                                st.session_state['position_type'] = None
                            else:
                                update_log(f"Position active. P&L: {price_change:.2f}%. Waiting for TP/SL or next signal.", level="info")
                else:
                    update_log(f"No entry price recorded for open position of {selected_symbol}.", level="warning")
        else:
            position_info_placeholder.info("Auto-trading is currently inactive. No automated trades will be placed.")


        st.sidebar.json({
            "Last Prediction": st.session_state['last_prediction'],
            "Confidence": st.session_state['last_prediction_proba'],
            "Last Trade Type": st.session_state['last_trade_type'],
            "Last Trade Amount": st.session_state['last_trade_amount'],
            "Last Trade Time": st.session_state['last_trade_time'],
            "Open Position": st.session_state['open_position'],
            "Position Entry Price": st.session_state['position_entry_price'],
            "Position Type": st.session_state['position_type'],
            "Current Mode": st.session_state['trade_mode']
        })

        time.sleep(prediction_interval) # Wait for the next prediction cycle
        st.rerun() # Force a rerun to update the UI

    except Exception as e:
        update_log(f"[{selected_symbol}] An unexpected error occurred: {e}. Stopping this symbol's auto-trading.", level="error")
        stop_auto_trade(reason="critical error")
        # No break here, let the loop continue to show error and then sleep/rerun
        # This allows the UI to update with the error message.

# --- Live Trading Log (Moved to bottom) ---
st.subheader("Live Trading Log")
log_container = st.empty() # Define a container for the log messages

# Render the log messages
with log_container.container():
    for msg in st.session_state.trading_messages:
        if "[ERROR]" in msg:
            st.error(msg)
        elif "[WARNING]" in msg:
            st.warning(msg)
        elif "[INFO]" in msg:
            st.info(msg)
        else:
            st.write(msg) # Default for other messages

# Initial log message (outside the loop, so it's always shown when app starts/reloads)
if not st.session_state['auto_trade_started']:
    update_log("No open positions found. Auto-trading is stopped.", level="info")
