import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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

# --- CSS STYLING ---
st.markdown("""
<style>
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

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

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

# --- INTERACTIVE CHARTS (FIXED PAN/ZOOM) ---

def plot_greeks_interactive(current_price, strike, days_left, iv, risk_free=0.045):
    prices = np.linspace(strike * 0.8, strike * 1.2, 100)
    T = max(days_left / 365.0, 0.001)
    deltas = [calculate_greeks(p, strike, T, risk_free, iv)[0] for p in prices]
    gammas = [calculate_greeks(p, strike, T, risk_free, iv)[1] for p in prices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=deltas, mode='lines', name='Delta (Speed)', line=dict(color='#4DA6FF', width=3)))
    fig.add_trace(go.Scatter(x=prices, y=gammas, mode='lines', name='Gamma (Acceleration)', line=dict(color='#00FF7F', width=2, dash='dash'), yaxis="y2"))
    curr_d, curr_g, _ = calculate_greeks(current_price, strike, T, risk_free, iv)
    fig.add_trace(go.Scatter(x=[current_price], y=[curr_d], mode='markers', name='You Are Here', marker=dict(color='white', size=12, line=dict(color='#4DA6FF', width=2))))

    fig.update_layout(
        title="Speed (Delta) vs Acceleration (Gamma)",
        xaxis_title="Stock Price",
        yaxis_title="Delta",
        yaxis2=dict(title="Gamma", overlaying="y", side="right"),
        template="plotly_dark",
        hovermode="x unified",
        dragmode='pan',  # Enables Click-to-Drag
        height=500
    )
    return fig

def plot_simulation_interactive(S, K, days_left, iv, r=0.045, purchase_price=0):
    prices = np.linspace(S * 0.8, S * 1.2, 100)
    T1 = max(days_left / 365.0, 0.0001)
    pnl_today = [black_scholes_price(p, K, T1, r, iv) - purchase_price for p in prices]
    T2 = max((days_left / 2) / 365.0, 0.0001)
    pnl_half = [black_scholes_price(p, K, T2, r, iv) - purchase_price for p in prices]
    T3 = 0.0001
    pnl_exp = [black_scholes_price(p, K, T3, r, iv) - purchase_price for p in prices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=pnl_today, mode='lines', name='Today (T+0)', line=dict(color='#4DA6FF', width=3)))
    fig.add_trace(go.Scatter(x=prices, y=pnl_half, mode='lines', name=f'Halfway (T+{int(days_left/2)})', line=dict(color='#FFD700', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=prices, y=pnl_exp, mode='lines', name='Expiration (Max Risk)', line=dict(color='#FF4B4B', width=2, dash='dot')))
    
    fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.5)
    fig.add_vline(x=S, line_color="gray", line_dash="dash", annotation_text="Current Price")

    fig.update_layout(
        title="🔮 Interactive Future Simulator (Scroll Zoom + Click Drag)",
        xaxis_title="Stock Price ($)",
        yaxis_title="Estimated P&L ($)",
        template="plotly_dark",
        hovermode="x unified",
        dragmode='pan', # Enables Click-to-Drag
        height=600
    )
    return fig

def plot_flow_battle_interactive(calls, puts, current_strike):
    c_vol = calls[['strike', 'volume']].groupby('strike').sum().rename(columns={'volume': 'Call Vol'})
    p_vol = puts[['strike', 'volume']].groupby('strike').sum().rename(columns={'volume': 'Put Vol'})
    df = pd.merge(c_vol, p_vol, on='strike', how='outer').fillna(0)
    strikes = sorted(df.index)
    try: idx = strikes.index(current_strike)
    except: idx = (np.abs(np.array(strikes) - current_strike)).argmin()
    start_idx = max(0, idx - 4); end_idx = min(len(strikes), idx + 5)
    subset = df.iloc[start_idx:end_idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=subset.index, y=subset['Call Vol'], name='Bulls (Calls)', marker_color='#00FF7F'))
    fig.add_trace(go.Bar(x=subset.index, y=subset['Put Vol'], name='Bears (Puts)', marker_color='#FF4B4B'))

    fig.update_layout(
        title="⚔️ Interactive Battle Map",
        xaxis_title="Strike Price",
        yaxis_title="Volume",
        barmode='group',
        template="plotly_dark",
        hovermode="x unified",
        dragmode='pan',
        height=500
    )
    return fig

def plot_whale_activity_interactive(calls_df, current_strike):
    strikes = sorted(calls_df['strike'].unique())
    try: idx = strikes.index(current_strike)
    except: idx = 0
    start_idx = max(0, idx - 3); end_idx = min(len(strikes), idx + 4)
    relevant_strikes = strikes[start_idx:end_idx]
    subset = calls_df[calls_df['strike'].isin(relevant_strikes)].copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=subset['strike'], y=subset['openInterest'], name='Open Interest (Old)', marker_color='#4DA6FF', opacity=0.6))
    fig.add_trace(go.Bar(x=subset['strike'], y=subset['volume'], name='Volume (New)', marker_color='#00FF7F'))

    fig.update_layout(
        title="Whale Detector (Hover to see Counts)",
        xaxis_title="Strike Price",
        yaxis_title="Contracts",
        barmode='group',
        template="plotly_dark",
        hovermode="x unified",
        dragmode='pan',
        height=500
    )
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
            
            days_left = (datetime.strptime(selected_date, "%Y-%m-%d") - datetime.now()).days
            if days_left < 1: days_left = 1
            theo_price = black_scholes_price(current_price, strike_price, days_left/365, 0.045, contract_iv)
        
        st.title(f"📊 {ticker} Command Center 🔒")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}", f"{current_price - prev_close:.2f}")
        col2.metric("Your Strike", f"${strike_price:.2f}")
        col3.metric("Selected Expiration", selected_date)
        st.markdown("---")

        # --- TABS ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
            "1. Price", "2. Volume", "3. IV", "4. Rule of 16", 
            "5. Whale Detector", "6. Risk & Profit", "7. Max Pain", "8. News", 
            "9. 🤖 Chart Analyst", "10. 💬 Strategy Engine", "11. 🔮 Future Simulator", "12. 🌊 Flow Monitor"
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
            fig_whale = plot_whale_activity_interactive(calls, strike_price)
            # ENABLED SCROLL ZOOM AND PAN
            st.plotly_chart(fig_whale, use_container_width=True, config={'scrollZoom': True})

        with tab6:
            st.header("Risk & Profit Hub")
            d, g, t = calculate_greeks(current_price, strike_price, days_left/365, 0.045, contract_iv)
            c1, c2, c3 = st.columns(3)
            c1.metric("Delta", f"{d:.2f}"); c2.metric("Gamma", f"{g:.3f}"); c3.metric("Theta", f"{t:.3f}")
            
            fig_greeks = plot_greeks_interactive(current_price, strike_price, days_left, contract_iv)
            # ENABLED SCROLL ZOOM AND PAN
            st.plotly_chart(fig_greeks, use_container_width=True, config={'scrollZoom': True})
            
            st.markdown("---")
            st.subheader("🎯 Profit Target Calculator")
            col_calc1, col_calc2 = st.columns([1, 2])
            with col_calc1: desired_profit = st.number_input("Desired Profit ($)", value=50, step=10)
            with col_calc2:
                if d > 0.001:
                    price_change_needed = desired_profit / 100
                    stock_move_needed = price_change_needed / d
                    target_stock_price = current_price + stock_move_needed
                    st.markdown(f"<div class='profit-box' style='color: white;'>Target Stock Price: <b>${target_stock_price:.2f}</b><br>(Move: +${stock_move_needed:.2f})</div>", unsafe_allow_html=True)
                else: st.warning("⚠️ Delta is 0. Cannot calculate target.")
            st.markdown("---")
            st.subheader("🗓️ Holiday Decay Calculator")
            holidays = st.number_input("Days market is closed", value=1, step=1)
            est_loss = abs(t) * holidays * 100
            st.markdown(f"<div class='theta-box' style='color: white;'>Estimated Loss: <b>${est_loss:.2f} per contract</b></div>", unsafe_allow_html=True)

        with tab7: st.metric("Max Pain", f"${calculate_max_pain(full_chain):.2f}")
        with tab8:
            try: 
                for item in stock_conn.news[:3]: st.markdown(f"- [{item['title']}]({item['link']})")
            except: st.write("No news found.")

        with tab9:
            st.header("🤖 AI Chart Analyst")
            st.write("Upload screenshots for expert battle analysis.")
            available_models = ["models/gemini-2.0-flash-exp", "models/gemini-2.0-flash"]
            selected_model = st.selectbox("🧠 Select Model:", available_models, index=0)

            uploaded_files = st.file_uploader("Upload Screenshots", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            if uploaded_files and st.button("Analyze Images"):
                if "api_keys" in st.secrets and "gemini" in st.secrets["api_keys"]:
                    secure_key = st.secrets["api_keys"]["gemini"]
                    genai.configure(api_key=secure_key)
                    with st.spinner(f"🤖 Analyzing Battleground..."):
                        try:
                            model = genai.GenerativeModel(selected_model)
                            prompt = "You are a Senior Options Strategist. Analyze these charts for 'Walls', 'Squeezes', and 'Traps'. Be decisive."
                            content = [prompt] + [Image.open(f) for f in uploaded_files]
                            response = model.generate_content(content)
                            st.markdown("### 📝 Strategic Report"); st.write(response.text)
                        except Exception as e: st.error(f"Error: {e}")
                else: st.error("❌ API Key not found!")

        with tab10:
            st.header("💬 AI Strategy Engine")
            col_comp1, col_comp2 = st.columns([1,3])
            with col_comp1: comp_strike = st.number_input("Compare with Strike ($)", value=0.0, step=1.0)
            with col_comp2: user_question = st.text_input("Your Question:", placeholder="e.g. Is this a trap?")

            if user_question:
                if "api_keys" in st.secrets and "gemini" in st.secrets["api_keys"]:
                    secure_key = st.secrets["api_keys"]["gemini"]
                    genai.configure(api_key=secure_key)
                    
                    context_data = f"MAIN: Ticker {ticker} | Price ${current_price:.2f} | Strike ${strike_price:.2f} | IV {contract_iv*100:.2f}% | Delta {d:.3f} | Theta {t:.3f}"
                    if comp_strike > 0:
                        try:
                            c_iv = calls.iloc[(calls['strike'] - comp_strike).abs().argsort()[:1]].iloc[0]['impliedVolatility']
                            c_d, _, c_t = calculate_greeks(current_price, comp_strike, days_left/365, 0.045, c_iv)
                            context_data += f"\nCOMPARE: Strike ${comp_strike} | Delta {c_d:.3f} | Theta {c_t:.3f}"
                        except: pass

                    with st.spinner("🤖 Consulting Senior Trader..."):
                        try:
                            model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
                            prompt = f"You are a cynical, expert Options Trader. Analyze: {context_data}\nUser: {user_question}\nVerdict:"
                            st.write(model.generate_content(prompt).text)
                        except Exception as e: st.error(f"Error: {e}")
                else: st.error("❌ API Key not found!")

        with tab11:
            st.header("🔮 Future P&L Simulator")
            st.write("Scroll UP on the chart to zoom in. Check for gaps between lines!")
            fig_sim = plot_simulation_interactive(current_price, strike_price, days_left, contract_iv, purchase_price=theo_price)
            # ENABLED SCROLL ZOOM AND PAN
            st.plotly_chart(fig_sim, use_container_width=True, config={'scrollZoom': True})
            st.info("💡 **Hover** over any line to see the exact value. **Scroll** to zoom in. **Click & Drag** to move.")

        with tab12:
            st.header("🌊 Market Flow: Bulls vs Bears")
            total_call_vol = calls['volume'].sum(); total_put_vol = puts['volume'].sum()
            pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Call Vol", f"{int(total_call_vol):,}"); c2.metric("Put Vol", f"{int(total_put_vol):,}")
            c3.metric("PCR", f"{pcr:.2f}", "🐻 Bearish" if pcr > 1 else "🐂 Bullish")
            st.markdown("---")
            st.subheader("⚔️ Interactive Battle Map")
            fig_flow = plot_flow_battle_interactive(calls, puts, strike_price)
            # ENABLED SCROLL ZOOM AND PAN
            st.plotly_chart(fig_flow, use_container_width=True, config={'scrollZoom': True})

    except Exception as e:
        if "Too Many Requests" in str(e):
             st.error("🚦 Traffic Jam. Retrying... wait.")
             time.sleep(2)
             st.rerun()
        else:
            st.error(f"Waiting for inputs... ({e})")
