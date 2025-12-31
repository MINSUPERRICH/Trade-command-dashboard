import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import random
import google.generativeai as genai
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Options Command Center", layout="wide", page_icon="🚀")

# --- PASSWORD PROTECTION ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    def password_entered():
        if st.session_state["password"] == st.secrets["passwords"]["main_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.text_input("🔑 Enter Password", type="password", on_change=password_entered, key="password")
        return False
    return True

if not check_password():
    st.stop()

# --- CSS STYLING (FIXED COLORS HERE) ---
st.markdown("""
<style>
    /* Added 'color: white' to ensure text is readable on dark backgrounds */
    .metric-card { background-color: #0e1117; border: 1px solid #262730; padding: 20px; border-radius: 10px; color: white; }
    .profit-box { background-color: #1E3D59; padding: 20px; border-radius: 10px; border-left: 5px solid #00FF7F; margin-bottom: 20px; color: white; }
    .theta-box { background-color: #330000; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B; color: white; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def fetch_with_retry(func, *args, retries=3):
    for i in range(retries):
        try:
            return func(*args)
        except Exception as e:
            if "Too Many Requests" in str(e) or "404" in str(e):
                time.sleep((i + 1) * 2)
                continue
            raise e
    return func(*args)

@st.cache_data(ttl=900) 
def get_stock_history_and_info(ticker_symbol):
    def _get():
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period="5d")
        info = stock.info
        return history, info
    return fetch_with_retry(_get)

@st.cache_data(ttl=900)
def get_option_chain_data(ticker_symbol, date):
    def _get():
        stock = yf.Ticker(ticker_symbol)
        opt_chain = stock.option_chain(date)
        calls = opt_chain.calls
        calls['type'] = 'call'
        puts = opt_chain.puts
        puts['type'] = 'put'
        full = pd.concat([calls, puts])
        return full, calls, puts
    return fetch_with_retry(_get)

def get_ticker_object(ticker_symbol):
    return yf.Ticker(ticker_symbol)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    term1 = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    theta_annual = term1 - term2
    theta_daily = theta_annual / 365.0
    return delta, gamma, theta_daily

def plot_greeks_curve(current_price, strike, days_left, iv, risk_free=0.045):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    prices = np.linspace(strike * 0.8, strike * 1.2, 100)
    T = max(days_left / 365.0, 0.001)
    deltas = []
    gammas = []
    for p in prices:
        d, g, t = calculate_greeks(p, strike, T, risk_free, iv)
        deltas.append(d)
        gammas.append(g)
    
    ax1.plot(prices, deltas, color='#4DA6FF', linewidth=3, label='Delta (Speed)')
    ax1.set_xlabel('Stock Price', color='white')
    ax1.set_ylabel('Delta', color='#4DA6FF', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#4DA6FF', colors='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.set_ylim(0, 1)
    
    ax2 = ax1.twinx()
    ax2.plot(prices, gammas, color='#00FF7F', linewidth=2, linestyle='--', label='Gamma (Acceleration)')
    ax2.set_ylabel('Gamma', color='#00FF7F', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#00FF7F', colors='white')
    
    curr_d, curr_g, curr_t = calculate_greeks(current_price, strike, T, risk_free, iv)
    ax1.scatter([current_price], [curr_d], color='white', edgecolor='#4DA6FF', s=100, zorder=10, label='You Are Here')
    
    ax1.set_title(f"Speed (Delta) vs Acceleration (Gamma)", color='white')
    ax1.grid(True, alpha=0.1)
    fig.patch.set_facecolor('#0E1117'); ax1.set_facecolor('#0E1117')
    return fig

def plot_whale_activity(calls_df, current_strike):
    strikes = sorted(calls_df['strike'].unique())
    try:
        idx = strikes.index(current_strike)
        start_idx = max(0, idx - 2)
        end_idx = min(len(strikes), idx + 3)
        relevant_strikes = strikes[start_idx:end_idx]
    except:
        relevant_strikes = strikes[:5]
    subset = calls_df[calls_df['strike'].isin(relevant_strikes)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(subset['strike']))
    width = 0.35
    ax.bar(x - width/2, subset['openInterest'], width, label='Open Interest (Yesterday)', color='#4DA6FF', alpha=0.6)
    ax.bar(x + width/2, subset['volume'], width, label='Volume (Today)', color='#00FF7F')
    ax.set_xticks(x); ax.set_xticklabels(subset['strike'])
    ax.set_title("Whale Detector: Yesterday (OI) vs Today (Vol)", color='white')
    ax.legend(facecolor='#262730', labelcolor='white')
    ax.tick_params(colors='white'); ax.grid(axis='y', alpha=0.1)
    fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
    return fig

def calculate_max_pain(options_chain):
    strikes = options_chain['strike'].unique()
    max_pain_data = []
    for strike in strikes:
        calls_at_strike = options_chain[options_chain['type'] == 'call']
        puts_at_strike = options_chain[options_chain['type'] == 'put']
        call_loss = calls_at_strike.apply(lambda x: max(0, strike - x['strike']) * x['openInterest'], axis=1).sum()
        put_loss = puts_at_strike.apply(lambda x: max(0, x['strike'] - strike) * x['openInterest'], axis=1).sum()
        max_pain_data.append({'strike': strike, 'total_loss': call_loss + put_loss})
    df_pain = pd.DataFrame(max_pain_data)
    if df_pain.empty: return 0
    return df_pain.loc[df_pain['total_loss'].idxmin()]['strike']

# --- MAIN APP ---
st.sidebar.markdown("## ⚙️ Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="PEP").upper()
strike_price = st.sidebar.number_input("Strike Price ($)", value=148.0)

if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear()
    st.rerun()

if ticker:
    try:
        stock_conn = get_ticker_object(ticker)
        with st.spinner('Fetching market data...'):
            history, info = get_stock_history_and_info(ticker)
            current_price = info.get('currentPrice', history['Close'].iloc[-1])
            prev_close = info.get('previousClose', history['Close'].iloc[-2])
            
            expirations = stock_conn.options
            if not expirations:
                st.error("No options data found.")
                st.stop()
            selected_date = st.sidebar.selectbox("Expiration Date", expirations)
            
            full_chain, calls, puts = get_option_chain_data(ticker, selected_date)
            specific_contract = calls.iloc[(calls['strike'] - strike_price).abs().argsort()[:1]]
            contract_iv = specific_contract.iloc[0]['impliedVolatility']
        
        st.title(f"📊 {ticker} Command Center 🔒")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}", f"{current_price - prev_close:.2f}")
        col2.metric("Your Strike", f"${strike_price:.2f}")
        col3.metric("Selected Expiration", selected_date)
        st.markdown("---")

        # --- TABS ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "1. Price", "2. Volume", "3. IV", "4. Rule of 16", 
            "5. Whale Detector", "6. Risk & Profit", "7. Max Pain", "8. News", "9. 🤖 AI Trading Buddy"
        ])

        with tab1:
            st.line_chart(history['Close'])
            gap = current_price - prev_close
            if abs(gap) < 0.50: st.success("✅ Stable Open")
            else: st.warning("⚠️ Volatile Open")

        with tab2: st.metric("Volume", f"{info.get('volume', 0):,}")
        with tab3: st.metric("Implied Volatility", f"{contract_iv * 100:.2f}%")
        with tab4: st.metric("Expected Daily Move", f"${(contract_iv * 100 / 16) / 100 * current_price:.2f}")

        with tab5:
            st.header("Whale Detector")
            fig_whale = plot_whale_activity(calls, strike_price)
            st.pyplot(fig_whale)

        with tab6:
            st.header("Risk & Profit Hub")
            expiry_dt = datetime.strptime(selected_date, "%Y-%m-%d")
            days_left = (expiry_dt - datetime.now()).days
            if days_left < 0: days_left = 0
            
            d, g, t = calculate_greeks(current_price, strike_price, days_left/365, 0.045, contract_iv)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Delta", f"{d:.2f}")
            c2.metric("Gamma", f"{g:.3f}")
            c3.metric("Theta", f"{t:.3f}")
            
            fig_greeks = plot_greeks_curve(current_price, strike_price, days_left, contract_iv)
            st.pyplot(fig_greeks)
            
            st.markdown("---")
            st.subheader("🎯 Profit Target Calculator")
            col_calc1, col_calc2 = st.columns([1, 2])
            with col_calc1:
                desired_profit = st.number_input("Desired Profit ($)", value=50, step=10)
            with col_calc2:
                if d > 0.001:
                    price_change_needed = desired_profit / 100
                    stock_move_needed = price_change_needed / d
                    target_stock_price = current_price + stock_move_needed
                    # Added style directly to div to be safe, plus global style above
                    st.markdown(f"""
                    <div class='profit-box' style='color: white;'>
                        Target Stock Price: <b>${target_stock_price:.2f}</b><br>
                        (Move: +${stock_move_needed:.2f})
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Delta is 0. Cannot calculate target.")
            st.markdown("---")
            st.subheader("🗓️ Holiday Decay Calculator")
            holidays = st.number_input("Days market is closed", value=1, step=1)
            est_loss = abs(t) * holidays * 100
            st.markdown(f"""
            <div class='theta-box' style='color: white;'>
                Estimated Loss: <b>${est_loss:.2f} per contract</b>
            </div>
            """, unsafe_allow_html=True)

        with tab7: st.metric("Max Pain", f"${calculate_max_pain(full_chain):.2f}")
        with tab8:
            try: 
                for item in stock_conn.news[:3]: st.markdown(f"- [{item['title']}]({item['link']})")
            except: st.write("No news found.")

        # --- TAB 9: AI TRADING BUDDY ---
        with tab9:
            st.header("🤖 AI Trading Buddy")
            st.write("Upload your charts. I'll tell you what I really think.")
            
            available_models = [
                "models/gemini-2.0-flash-exp",
                "models/gemini-2.0-flash",
            ]
            selected_model = st.selectbox("🧠 Select Brain:", available_models, index=0)

            uploaded_files = st.file_uploader("Upload Screenshots", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            
            if uploaded_files:
                images = []
                cols = st.columns(len(uploaded_files))
                for i, file in enumerate(uploaded_files):
                    img = Image.open(file)
                    images.append(img)
                    with cols[i]:
                        st.image(img, caption=f"Chart {i+1}", use_container_width=True)
                
                if st.button("Analyze Images"):
                    if "api_keys" in st.secrets and "gemini" in st.secrets["api_keys"]:
                        secure_key = st.secrets["api_keys"]["gemini"]
                        genai.configure(api_key=secure_key)

                        with st.spinner(f"🤖 {selected_model} is thinking..."):
                            try:
                                model = genai.GenerativeModel(selected_model)
                                
                                prompt = """
                                Act as a highly experienced, slightly cynical, but supportive senior options trader. 
                                You are reviewing these charts for your junior partner (the user).
                                
                                Speak directly to the user (use "You" and "I").
                                Do NOT write a formal report. Write a chat message.
                                
                                1. **The Hook:** Start with a reaction. (e.g., "Woah, look at that volume!" or "This looks messy...").
                                2. **The "Real Talk":** Analyze the Whale Detector (Bars) and the Delta Curve. 
                                   - If Green bars > Blue bars: Say "The bulls are awake."
                                   - If Blue bars > Green bars: Say "It's a sleepy day. Walls are holding."
                                   - If the Curve is a cliff: Warn them about "Binary Risk."
                                3. **The Verdict:** specific advice. "If I were you, I would..." or "Be careful of..."
                                
                                Use emojis. Keep it under 200 words. Make it sound like a human text message.
                                """
                                
                                content = [prompt] + images
                                response = model.generate_content(content)
                                st.markdown("### 💬 Chat with Gemini")
                                st.success(response.text)
                                
                            except Exception as e:
                                st.error(f"Error with {selected_model}: {e}")
                    else:
                        st.error("❌ API Key not found! Check Streamlit Secrets.")

    except Exception as e:
        if "Too Many Requests" in str(e):
             st.error("🚦 Traffic Jam. Retrying... wait.")
             time.sleep(2)
             st.rerun()
        else:
            st.error(f"Waiting for inputs... ({e})")
