import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Options Command Center", layout="wide", page_icon="🚀")

# --- CUSTOM CSS FOR "DASHBOARD" FEEL ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .pass { color: #00FF7F; font-weight: bold; }
    .fail { color: #FF4B4B; font-weight: bold; }
    .warning { color: #FFA500; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_stock_data(ticker_symbol):
    """Fetches real-time stock info and history."""
    stock = yf.Ticker(ticker_symbol)
    history = stock.history(period="5d")  # Get recent days for gap check
    info = stock.info
    return stock, history, info

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    """
    Calculates Option Delta using Black-Scholes formula.
    S: Stock Price, K: Strike, T: Time to Expiration (years),
    r: Risk-free rate, sigma: Implied Volatility
    """
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def calculate_max_pain(options_chain):
    """Calculates the strike price where option writers lose the least money."""
    # This is a simplified estimation summing cash value of all expiring options
    strikes = options_chain['strike'].unique()
    max_pain_data = []

    for strike in strikes:
        # Assume stock expires at this strike
        # Value of Calls = max(0, Strike - K) * OI
        calls_at_strike = options_chain[options_chain['type'] == 'call']
        puts_at_strike = options_chain[options_chain['type'] == 'put']
        
        call_loss = calls_at_strike.apply(lambda x: max(0, strike - x['strike']) * x['openInterest'], axis=1).sum()
        put_loss = puts_at_strike.apply(lambda x: max(0, x['strike'] - strike) * x['openInterest'], axis=1).sum()
        
        max_pain_data.append({'strike': strike, 'total_loss': call_loss + put_loss})
    
    df_pain = pd.DataFrame(max_pain_data)
    if df_pain.empty: return 0
    return df_pain.loc[df_pain['total_loss'].idxmin()]['strike']

# --- SIDEBAR INPUTS ---
st.sidebar.header("⚙️ Trade Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="NKE").upper()

strike_price = st.sidebar.number_input("Strike Price ($)", value=61.0)
target_profit_move = st.sidebar.number_input("Target Stock Move ($)", value=2.0, help="How many dollars do you need the stock to move today?")

# --- MAIN APP LOGIC ---

if ticker:
    try:
        # 1. FETCH DATA
        with st.spinner(f'Fetching data for {ticker}...'):
            stock, history, info = get_stock_data(ticker)
            current_price = info.get('currentPrice', history['Close'].iloc[-1])
            prev_close = info.get('previousClose', history['Close'].iloc[-2])
            
            # Get Options Dates
            expirations = stock.options
            if not expirations:
                st.error("No options data found.")
                st.stop()
                
            selected_date = st.sidebar.selectbox("Expiration Date", expirations)
            
            # Get Option Chain for selected date
            opt_chain = stock.option_chain(selected_date)
            calls = opt_chain.calls
            calls['type'] = 'call'
            puts = opt_chain.puts
            puts['type'] = 'put'
            full_chain = pd.concat([calls, puts])
            
            # Find specific contract user is interested in
            # Look for the call closest to the strike input
            specific_contract = calls.iloc[(calls['strike'] - strike_price).abs().argsort()[:1]]
            if specific_contract.empty:
                st.warning("Strike price not found in chain.")
                st.stop()
            
            contract_data = specific_contract.iloc[0]
            contract_iv = contract_data['impliedVolatility']
            contract_volume = contract_data['volume'] if not np.isnan(contract_data['volume']) else 0
            contract_oi = contract_data['openInterest'] if not np.isnan(contract_data['openInterest']) else 0
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    # --- DASHBOARD HEADER ---
    st.title(f"📊 {ticker} Trade Command Center")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:.2f}", f"{current_price - prev_close:.2f}")
    col2.metric("Target Strike", f"${strike_price:.2f}")
    col3.metric("Selected Expiration", selected_date)

    st.markdown("---")

    # --- THE 6 TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1. Price (Gap)", 
        "2. Volume", 
        "3. IV (Fear)", 
        "4. Rule of 16", 
        "5. Vol vs OI", 
        "6. Delta (Odds)",
        "💀 Max Pain"
    ])

    # ---------------- TAB 1: PRICE GAP ----------------
    with tab1:
        st.header("Price Gap Check")
        gap = current_price - prev_close
        gap_percent = (gap / prev_close) * 100
        
        st.metric("Overnight Gap", f"${gap:.2f}", f"{gap_percent:.2f}%")
        
        if abs(gap) < 0.50:
            st.success("✅ PASSED: Gap is small (< $0.50). 'Rule of 16' is valid.")
        elif gap < -1.00:
            st.error("⚠️ WARNING: Big Gap Down. Watch your Stop Loss ($0.41) immediately!")
        else:
            st.info("ℹ️ NOTE: Significant Gap. Volatility is high.")

    # ---------------- TAB 2: VOLUME ----------------
    with tab2:
        st.header("Stock Volume Check")
        avg_vol = info.get('averageVolume', 0)
        curr_vol = info.get('volume', 0) # Note: 'volume' in info is often delayed
        
        st.write(f"**Average Daily Volume:** {avg_vol:,}")
        if curr_vol > 0:
            st.write(f"**Current Volume:** {curr_vol:,}")
        else:
            st.write("*(Market might be closed or pre-market volume not available via free API)*")
            
        st.info("💡 Look for 'Heavy Volume' at 9:35 AM to confirm the move is real.")

    # ---------------- TAB 3: IV (FEAR) ----------------
    with tab3:
        st.header("Implied Volatility (IV) Check")
        iv_percent = contract_iv * 100
        st.metric("IV for your Strike", f"{iv_percent:.2f}%")
        
        if iv_percent < 20:
            st.write("🧊 **Low IV:** Options are cheap, but stock might be boring.")
        elif iv_percent > 50:
            st.warning("🔥 **High IV:** Options are expensive. Be careful of 'IV Crush'.")
        else:
            st.success("✅ **Normal IV:** Good balance of risk/reward.")

    # ---------------- TAB 4: RULE OF 16 ----------------
    with tab4:
        st.header("The Reality Check (Rule of 16)")
        
        daily_move_pct = iv_percent / 16
        daily_move_dollar = current_price * (daily_move_pct / 100)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Expected Daily Move %", f"{daily_move_pct:.2f}%")
            st.metric("Expected Daily Move $", f"${daily_move_dollar:.2f}")
        
        with col_b:
            st.write(f"**Your Target Move:** ${target_profit_move:.2f}")
            if target_profit_move > daily_move_dollar:
                st.error(f"❌ **FAIL:** You need ${target_profit_move}, but market only expects ${daily_move_dollar:.2f}. This is statistically unlikely today.")
            else:
                st.success(f"✅ **PASS:** Your target (${target_profit_move}) is within the expected range (${daily_move_dollar:.2f}).")

    # ---------------- TAB 5: VOL vs OI ----------------
    with tab5:
        st.header("Trend Check (New Money)")
        col_x, col_y = st.columns(2)
        col_x.metric("Today's Volume", f"{contract_volume:.0f}")
        col_y.metric("Open Interest (Yesterday)", f"{contract_oi:.0f}")
        
        if contract_oi == 0:
            ratio = 0
        else:
            ratio = contract_volume / contract_oi
            
        st.write(f"**Vol / OI Ratio:** {ratio:.2f}")
        
        if contract_volume > contract_oi:
            st.success("✅ **PASS (Breakout Signal):** Volume > Open Interest. New money is flooding in aggressively!")
        elif ratio > 0.5:
             st.warning("⚠️ **WATCH:** Moderate activity.")
        else:
            st.error("❌ **FAIL:** Low Volume. Mostly old traders passing contracts around.")

    # ---------------- TAB 6: DELTA (ODDS) ----------------
    with tab6:
        st.header("Odds Check (Delta)")
        
        # Calculate Time to Expiry in Years
        expiry_dt = datetime.strptime(selected_date, "%Y-%m-%d")
        days_to_expiry = (expiry_dt - datetime.now()).days
        if days_to_expiry < 0: days_to_expiry = 0
        T = days_to_expiry / 365.0
        
        # Risk free rate approx 4.5%
        delta = calculate_delta(current_price, strike_price, T, 0.045, contract_iv, 'call')
        
        st.metric("Delta (Win Probability)", f"{delta:.2f}")
        
        if delta < 0.30:
            st.error(f"❌ **High Risk:** Only {delta*100:.0f}% probability of expiring ITM.")
        elif delta > 0.70:
            st.success(f"✅ **Safe (Deep ITM):** {delta*100:.0f}% probability, but expensive.")
        else:
            st.success(f"✅ **Good Balance:** {delta*100:.0f}% probability. Standard swing trade.")

    # ---------------- TAB 7: MAX PAIN ----------------
    with tab7:
        st.header("💀 Max Pain (The Magnet)")
        st.write("Calculating Max Pain for this expiration... (this assumes simple cash value)")
        
        pain_price = calculate_max_pain(full_chain)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Current Stock Price", f"${current_price:.2f}")
        col_m2.metric("Max Pain Price", f"${pain_price:.2f}")
        
        diff = current_price - pain_price
        if abs(diff) < 1.00:
            st.warning("🧲 **MAGNET EFFECT:** Stock is pinned near Max Pain.")
        elif current_price > pain_price:
            st.info(f"📉 **Drag Risk:** Stock is ${diff:.2f} ABOVE Max Pain. Market Makers might want it lower.")
        else:
            st.info(f"📈 **Lift Potential:** Stock is ${abs(diff):.2f} BELOW Max Pain. Market Makers might want it higher.")