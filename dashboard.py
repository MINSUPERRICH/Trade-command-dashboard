import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Options Command Center", layout="wide", page_icon="🚀")

# --- PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["passwords"]["main_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("🔑 Enter Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again.
        st.text_input("🔑 Enter Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Stop here if password is wrong

# --- APP BEGINS HERE (Only runs if password is correct) ---

st.markdown("""
<style>
    .metric-card { background-color: #0e1117; border: 1px solid #262730; padding: 20px; border-radius: 10px; color: white; }
    .profit-box { background-color: #1E3D59; padding: 20px; border-radius: 10px; border-left: 5px solid #00FF7F; }
    .theta-box { background-color: #330000; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    history = stock.history(period="5d")
    info = stock.info
    return stock, history, info

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculates Delta, Gamma, AND Theta."""
    if T <= 0 or sigma <= 0: return 0, 0, 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
        
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta (Annual) - Simplified Black-Scholes
    term1 = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    theta_annual = term1 - term2
    theta_daily = theta_annual / 365.0
    
    return delta, gamma, theta_daily

def plot_whale_activity(calls_df, current_strike):
    """Plots Volume vs OI"""
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
ticker = st.sidebar.text_input("Ticker Symbol", value="NKE").upper()
strike_price = st.sidebar.number_input("Strike Price ($)", value=63.0) # Updated default for you

if ticker:
    try:
        stock, history, info = get_stock_data(ticker)
        current_price = info.get('currentPrice', history['Close'].iloc[-1])
        prev_close = info.get('previousClose', history['Close'].iloc[-2])
        
        expirations = stock.options
        selected_date = st.sidebar.selectbox("Expiration Date", expirations)
        
        opt_chain = stock.option_chain(selected_date)
        calls = opt_chain.calls; calls['type'] = 'call'
        puts = opt_chain.puts; puts['type'] = 'put'
        full_chain = pd.concat([calls, puts])
        
        specific_contract = calls.iloc[(calls['strike'] - strike_price).abs().argsort()[:1]]
        contract_iv = specific_contract.iloc[0]['impliedVolatility']
        contract_volume = specific_contract.iloc[0]['volume'] if not np.isnan(specific_contract.iloc[0]['volume']) else 0
        
        # --- HEADER ---
        st.title(f"📊 {ticker} Command Center 🔒")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}", f"{current_price - prev_close:.2f}")
        col2.metric("Your Strike", f"${strike_price:.2f}")
        col3.metric("Selected Expiration", selected_date)
        st.markdown("---")

        # --- TABS ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "1. Price", "2. Volume", "3. IV", "4. Rule of 16", 
            "5. Whale Detector", "6. Greeks & Holiday Calc", "7. Max Pain", "8. News"
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

        # --- UPDATED TAB 6: GREEKS + HOLIDAY CALCULATOR ---
        with tab6:
            st.header("Risk Calculator")
            expiry_dt = datetime.strptime(selected_date, "%Y-%m-%d")
            days_left = (expiry_dt - datetime.now()).days
            if days_left < 0: days_left = 0
            
            # Calculate Greeks
            d, g, t = calculate_greeks(current_price, strike_price, days_left/365, 0.045, contract_iv)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Delta (Speed)", f"{d:.2f}")
            c2.metric("Gamma (Accel)", f"{g:.3f}")
            c3.metric("Theta (Daily Decay)", f"{t:.3f}") # Usually negative
            
            st.markdown("---")
            st.markdown("### 🗓️ Holiday Decay Calculator")
            st.write("Markets are closed on Thursday (Jan 1). Time decay still happens.")
            
            holidays = st.number_input("Days market is closed (incl. weekends)", value=1, step=1)
            est_loss = abs(t) * holidays * 100 # x100 for 1 contract
            
            st.markdown(f"""
            <div class='theta-box'>
                <h4>📉 The "Holiday Tax"</h4>
                <p>If the stock price stays exactly the same at ${current_price:.2f}...</p>
                <p>You will lose approximately <b>${est_loss:.2f} per contract</b> just from holding through the holiday.</p>
            </div>
            """, unsafe_allow_html=True)

        with tab7: st.metric("Max Pain", f"${calculate_max_pain(full_chain):.2f}")
        with tab8:
            try: 
                for item in stock.news[:3]: st.markdown(f"- [{item['title']}]({item['link']})")
            except: st.write("No news found.")

    except Exception as e:
        st.error(f"Waiting for inputs... ({e})")
