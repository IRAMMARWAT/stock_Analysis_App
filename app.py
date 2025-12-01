# stock_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Stock Market Analysis & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stock-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StockAnalysisApp:
    def __init__(self):
        self.stock_data = None
        self.stock_ticker = None

    def fetch_stock_data(self, ticker: str, period: str = "1y") -> bool:
        """Fetch stock data from yfinance and ensure it's usable"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df is None or df.empty:
                return False
            # Ensure columns we expect exist
            expected_cols = {"Open", "High", "Low", "Close", "Volume"}
            if not expected_cols.issubset(set(df.columns)):
                return False
            # Make sure index is DatetimeIndex
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            self.stock_data = df
            self.stock_ticker = ticker.upper()
            return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False

    def display_basic_info(self):
        if self.stock_data is None or self.stock_data.empty:
            return

        st.subheader("📊 Basic Stock Information")
        col1, col2, col3, col4 = st.columns(4)

        try:
            current_price = float(self.stock_data["Close"].iloc[-1])
        except Exception:
            current_price = np.nan

        with col1:
            if not np.isnan(current_price):
                st.metric("Current Price", f"${current_price:.2f}")
            else:
                st.metric("Current Price", "N/A")

        with col2:
            if len(self.stock_data) >= 2:
                prev = float(self.stock_data["Close"].iloc[-2])
                daily_change = current_price - prev
                change_pct = (daily_change / prev) * 100 if prev != 0 else np.nan
                st.metric("Daily Change", f"${daily_change:.2f}", f"{change_pct:.2f}%")
            else:
                st.metric("Daily Change", "N/A")

        with col3:
            try:
                volume = int(self.stock_data["Volume"].iloc[-1])
                st.metric("Volume", f"{volume:,}")
            except Exception:
                st.metric("Volume", "N/A")

        with col4:
            try:
                avg_volume = int(self.stock_data["Volume"].mean())
                st.metric("Avg Volume", f"{avg_volume:,}")
            except Exception:
                st.metric("Avg Volume", "N/A")

    def plot_stock_data(self):
        if self.stock_data is None or self.stock_data.empty:
            return

        st.subheader("📈 Stock Price Chart")

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=self.stock_data.index,
                open=self.stock_data["Open"],
                high=self.stock_data["High"],
                low=self.stock_data["Low"],
                close=self.stock_data["Close"],
                name="Price",
            )
        )

        fig.update_layout(
            title=f"{self.stock_ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_technical_indicators(self):
        if self.stock_data is None or self.stock_data.empty:
            return

        st.subheader("🔧 Technical Indicators")
        df = self.stock_data.copy()

        # Moving averages
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()

        # RSI (14)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        loss = -delta.clip(upper=0).rolling(window=14, min_periods=1).mean()
        # avoid division by zero
        loss = loss.replace(0, 1e-9)
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

        ax1.plot(df.index, df["Close"], label="Close Price")
        ax1.plot(df.index, df["MA20"], label="20-day MA")
        ax1.plot(df.index, df["MA50"], label="50-day MA")
        ax1.set_title("Price and Moving Averages")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(df.index, df["RSI"], label="RSI")
        ax2.axhline(70, linestyle="--", alpha=0.6, label="Overbought")
        ax2.axhline(30, linestyle="--", alpha=0.6, label="Oversold")
        ax2.set_title("Relative Strength Index (RSI)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        st.pyplot(fig)

    def time_series_analysis(self):
        if self.stock_data is None or self.stock_data.empty:
            return

        st.subheader("⏰ Time Series Analysis")

        if len(self.stock_data) > 30:
            try:
                series = self.stock_data["Close"].dropna()
                # choose period as monthly-ish if daily data: 30
                decomposition = seasonal_decompose(series, period=30, model="additive", extrapolate_trend="freq")
                fig, axes = plt.subplots(4, 1, figsize=(12, 12))
                decomposition.observed.plot(ax=axes[0], title="Observed")
                decomposition.trend.plot(ax=axes[1], title="Trend")
                decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
                decomposition.resid.plot(ax=axes[3], title="Residual")
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Seasonal decomposition failed: {e}")
        else:
            st.info("Not enough data for seasonal decomposition (need >30 rows).")

    def random_forest_prediction(self):
        if self.stock_data is None or self.stock_data.empty:
            return

        st.subheader("🌲 Random Forest Prediction")

        df = self.stock_data.copy()

        # Ensure numeric and create features
        df = df[["Close", "Volume"]].copy()
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

        df["Price_Lag1"] = df["Close"].shift(1)
        df["Price_Lag2"] = df["Close"].shift(2)
        df["Price_Lag3"] = df["Close"].shift(3)
        df["Volume_Lag1"] = df["Volume"].shift(1)
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA10"] = df["Close"].rolling(10).mean()

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < 50:
            st.error("Not enough data to train Random Forest. Try increasing the period (e.g., 6mo, 1y, 2y).")
            return

        features = ["Price_Lag1", "Price_Lag2", "Price_Lag3", "Volume_Lag1", "MA5", "MA10"]
        X = df[features]
        y = df["Close"]

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test.index, y_test.values, label="Actual")
        ax.plot(y_test.index, y_pred, label="Predicted")
        ax.set_title(f"Random Forest Prediction (RMSE: {rmse:.2f})")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        feature_importance = pd.DataFrame({
            "feature": features,
            "importance": rf_model.feature_importances_
        }).sort_values("importance", ascending=False)

        fig_imp = px.bar(feature_importance, x="importance", y="feature", orientation="h", title="Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

    def lstm_prediction(self):
        if self.stock_data is None or self.stock_data.empty:
            return

        st.subheader("🧠 LSTM Prediction")

        close_values = self.stock_data["Close"].values.reshape(-1, 1).astype("float32")

        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_values)

        # Train/test split
        training_size = int(len(scaled_data) * 0.8)
        if training_size <= 60 or len(scaled_data) <= 60:
            st.error("Not enough data for LSTM sequences (need > 60 data points). Increase period.")
            return

        train_data = scaled_data[:training_size]
        test_data = scaled_data[training_size - 60 :]  # ensure we have seq_length history for test

        def create_sequences(data, seq_length=60):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i - seq_length : i, 0])
                y.append(data[i, 0])
            X = np.array(X)
            y = np.array(y)
            return X, y

        seq_length = 60
        X_train, y_train = create_sequences(train_data, seq_length)
        X_test, y_test = create_sequences(test_data, seq_length)

        # reshape to [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # build model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mean_squared_error")

        try:
            with st.spinner("Training LSTM model..."):
                history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), verbose=0)
        except Exception as e:
            st.error(f"LSTM training failed: {e}")
            return

        # Predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        ax1.plot(y_train_actual, label="Actual")
        ax1.plot(train_predict, label="Predicted")
        ax1.set_title(f"LSTM - Training Data (RMSE: {train_rmse:.2f})")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(y_test_actual, label="Actual")
        ax2.plot(test_predict, label="Predicted")
        ax2.set_title(f"LSTM - Test Data (RMSE: {test_rmse:.2f})")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        st.pyplot(fig)

        # training history loss
        fig_loss = plt.figure(figsize=(10, 4))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig_loss)

    def prophet_prediction(self):
        if self.stock_data is None or self.stock_data.empty:
            return

        st.subheader("🔮 Prophet Prediction")

        prophet_df = self.stock_data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        # Remove tz info if present
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)
        prophet_df = prophet_df.dropna()

        if len(prophet_df) < 30:
            st.error("Not enough data for Prophet modeling (need at least 30 rows).")
            return

        split_idx = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:split_idx]

        try:
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
            with st.spinner("Training Prophet model..."):
                m.fit(train_df)
        except Exception as e:
            st.error(f"Prophet training failed: {e}")
            return

        future = m.make_future_dataframe(periods=30)  # 30 days into future
        forecast = m.predict(future)

        # Plot forecast (matplotlib fig)
        fig1 = m.plot(forecast)
        plt.title("Prophet Forecast")
        st.pyplot(fig1)

        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)


def main():
    st.markdown('<h1 class="main-header">📈 Stock Market Analysis & Prediction</h1>', unsafe_allow_html=True)

    app = StockAnalysisApp()

    st.sidebar.title("Stock Selection")
    default_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NFLX", "NVDA"]

    stock_symbol = st.sidebar.selectbox("Select Stock Symbol", options=default_stocks, index=0)
    custom_symbol = st.sidebar.text_input("Or enter custom symbol (e.g., BTC-USD):")

    if custom_symbol:
        stock_symbol = custom_symbol.strip().upper()

    period = st.sidebar.selectbox("Select Time Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    st.sidebar.title("Analysis Options")
    show_technical = st.sidebar.checkbox("Technical Indicators", value=True)
    show_time_series = st.sidebar.checkbox("Time Series Analysis", value=True)
    show_rf = st.sidebar.checkbox("Random Forest Prediction", value=True)
    show_lstm = st.sidebar.checkbox("LSTM Prediction", value=True)
    show_prophet = st.sidebar.checkbox("Prophet Prediction", value=True)

    if st.sidebar.button("Fetch Data"):
        with st.spinner(f"Fetching data for {stock_symbol} ..."):
            success = app.fetch_stock_data(stock_symbol, period=period)
            if success:
                st.sidebar.success(f"Data loaded successfully for {stock_symbol}!")
            else:
                st.sidebar.error("Failed to fetch data. Please check the stock symbol or try a different period.")

    if app.stock_data is not None and not app.stock_data.empty:
        app.display_basic_info()
        app.plot_stock_data()

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Technical Analysis", "Time Series", "Random Forest", "LSTM", "Prophet"])

        with tab1:
            if show_technical:
                app.plot_technical_indicators()

        with tab2:
            if show_time_series:
                app.time_series_analysis()

        with tab3:
            if show_rf:
                app.random_forest_prediction()

        with tab4:
            if show_lstm:
                app.lstm_prediction()

        with tab5:
            if show_prophet:
                app.prophet_prediction()

        with st.expander("View Raw Data"):
            st.dataframe(app.stock_data.tail(50))
    else:
        st.info("👈 Select a stock symbol and click 'Fetch Data' to begin analysis.")
        st.subheader("Sample Data Structure")
        sample_data = pd.DataFrame({
            "Date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
            "Open": [150.0, 152.5, 151.8, 153.2, 154.0],
            "High": [152.5, 153.0, 153.5, 154.8, 155.2],
            "Low": [149.5, 151.0, 150.5, 152.0, 153.5],
            "Close": [152.0, 151.8, 153.0, 154.5, 154.8],
            "Volume": [1000000, 1200000, 950000, 1100000, 1050000]
        })
        st.dataframe(sample_data)

if __name__ == "__main__":
    main()
