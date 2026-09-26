import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Stock Trend Prediction",
    page_icon="📈",
    layout="wide"
)


# --------------------------------
# Title
# --------------------------------

st.title("📈 Stock Trend Prediction")

st.write(
    "Machine Learning based prediction of the next-day "
    "stock price direction."
)


# --------------------------------
# Load trained model
# --------------------------------

model = joblib.load("model.pkl")


# --------------------------------
# User Input
# --------------------------------

ticker = st.text_input(
    "Enter Stock Symbol",
    "AAPL"
).upper()


# --------------------------------
# Prediction Button
# --------------------------------

if st.button("Predict Stock Trend"):

    try:

        # Download data
        data = yf.download(
            ticker,
            period="2y",
            auto_adjust=True
        )


        # Check data
        if data.empty:

            st.error(
                "Stock symbol not found."
            )

            st.stop()


        # Fix MultiIndex
        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = (
                data.columns
                .get_level_values(0)
            )


        # --------------------------------
        # Create Features
        # --------------------------------

        data["MA5"] = (
            data["Close"]
            .rolling(5)
            .mean()
        )

        data["MA20"] = (
            data["Close"]
            .rolling(20)
            .mean()
        )

        data["Return"] = (
            data["Close"]
            .pct_change()
        )


        data.dropna(
            inplace=True
        )


        features = [
            "Close",
            "Volume",
            "MA5",
            "MA20",
            "Return"
        ]


        # Latest data
        latest_data = (
            data[features]
            .iloc[-1:]
        )


        # --------------------------------
        # Prediction
        # --------------------------------

        prediction = model.predict(
            latest_data
        )[0]


        # --------------------------------
        # Display Prediction
        # --------------------------------

        st.subheader(
            "Prediction Result"
        )


        if prediction == 1:

            st.success(
                "📈 Predicted Trend: UP"
            )

        else:

            st.error(
                "📉 Predicted Trend: DOWN"
            )


        # --------------------------------
        # Current Price
        # --------------------------------

        current_price = float(
            data["Close"].iloc[-1]
        )


        st.metric(
            "Latest Closing Price",
            f"{current_price:.2f}"
        )


        # --------------------------------
        # Graph
        # --------------------------------

        st.subheader(
            "Stock Price History"
        )


        fig, ax = plt.subplots()


        ax.plot(
            data.index,
            data["Close"],
            label="Close Price"
        )


        ax.plot(
            data.index,
            data["MA5"],
            label="MA5"
        )


        ax.plot(
            data.index,
            data["MA20"],
            label="MA20"
        )


        ax.set_xlabel(
            "Date"
        )

        ax.set_ylabel(
            "Price"
        )

        ax.legend()


        st.pyplot(fig)


        # --------------------------------
        # Recent Data
        # --------------------------------

        st.subheader(
            "Recent Stock Data"
        )


        st.dataframe(
            data.tail(10)
        )


    except Exception as e:

        st.error(
            f"Error: {e}"
        )


# --------------------------------
# Disclaimer
# --------------------------------

st.divider()

st.caption(
    "Educational project only. "
    "This prediction is not financial advice."
)