import streamlit as st
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import openai

# Load your OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Rowley Regis Wealth Management", page_icon="💼")
st.title("💼 Rowley Regis Wealth Management")

st.markdown("""
Welcome to **Rowley Regis Wealth Management** — your AI-powered assistant for intelligent, data-driven investment planning.

📊 Enter your portfolio details, choose your risk level, and let our AI help optimize your investments.
""")

# --- USER INPUT ---
tickers_input = st.text_input("📈 Enter stock tickers (comma-separated):", value="AAPL, MSFT, TSLA")
risk_level = st.selectbox("📊 What's your risk tolerance?", ["Low", "Moderate", "High"])

# --- PROCESS WHEN USER CLICKS ---
if st.button("Run Portfolio Analysis"):

    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

    # --- GET HISTORICAL DATA ---
    st.subheader("🔍 Fetching stock data...")
    try:
        prices = yf.download(tickers, period="1y")['Adj Close']
        if prices.isnull().all().all():
            st.error("❌ Failed to download stock data. Please check your ticker symbols.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        st.stop()

    # --- PORTFOLIO OPTIMIZATION ---
    st.subheader("📈 Optimizing your portfolio...")
    try:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        st.markdown("### 🧮 Recommended Allocation")
        for stock, weight in cleaned_weights.items():
            st.write(f"**{stock}**: {weight:.2%}")
    except Exception as e:
        st.error(f"❌ Error optimizing portfolio: {e}")
        st.stop()

    # --- AI FINANCIAL ADVICE ---
    st.subheader("🤖 Rowley AI Advice")

    prompt = f"""
    A client of Rowley Regis Wealth Management has a {risk_level.lower()} risk tolerance and the following portfolio allocation: {cleaned_weights}.
    Provide a brief, insightful investment recommendation tailored to this profile.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a senior financial advisor at Rowley Regis Wealth Management."},
                {"role": "user", "content": prompt}
            ]
        )
        advice = response.choices[0].message.content
        st.markdown(advice)
    except Exception as e:
        st.error(f"❌ Error getting AI response: {e}")
