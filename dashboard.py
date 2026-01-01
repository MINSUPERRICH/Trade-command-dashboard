import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go  # For Screen (Interactive)
import matplotlib.pyplot as plt    # For Word Report (Safe)
from datetime import datetime, timedelta
import time
import random
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import xlsxwriter
from docx import Document
from docx.shared import Inches

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
            if "too many requests" in str(e).lower():
                time.sleep(random.uniform(1, 3)) # Anti-block wait
                continue
            raise e
    return func(*args)

@st.cache_data(ttl=900) 
def get_stock_history_and_info(ticker_symbol):
    def _get():
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period="1mo")
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

# --- 1. INTERACTIVE PLOTS (PLOTLY - FOR SCREEN) ---
def plot_greeks_interactive(current_price, strike, days_left, iv, risk_free=0.045):
    prices = np.linspace(strike * 0.8, strike * 1.2, 100)
    T = max(days_left / 365.0, 0.001)
    deltas = [calculate_greeks(p, strike, T, risk_free, iv)[0] for p in prices]
    gammas = [calculate_greeks(p, strike, T, risk_free, iv)[1] for p in prices]
    curr_d, _, _ = calculate_greeks(current_price, strike, T, risk_free, iv)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=deltas, mode='lines', name='Delta (Speed)', line=dict(color='#4DA6FF', width=3)))
    fig.add_trace(go.Scatter(x=prices, y=gammas, mode='lines', name='Gamma (Accel)', line=dict(color='#00FF7F', width=2, dash='dash'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=[current_price], y=[curr_d], mode='markers', name='You', marker=dict(color='white', size=10)))
    fig.update_layout(title="Greeks", xaxis_title="Price", yaxis2=dict(overlaying="y", side="right"), template="plotly_dark", height=450, dragmode='pan')
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
    fig.add_trace(go.Scatter(x=prices, y=pnl_today, mode='lines', name='Today', line=dict(color='#4DA6FF', width=3)))
    fig.add_trace(go.Scatter(x=prices, y=pnl_half, mode='lines', name='Halfway', line=dict(color='#FFD700', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=prices, y=pnl_exp, mode='lines', name='Expiration', line=dict(color='#FF4B4B', width=2, dash='dot')))
    fig.add_hline(y=0, line_color="white", opacity=0.5)
    fig.update_layout(title="Future Simulator", xaxis_title="Price", yaxis_title="P&L", template="plotly_dark", height=500, dragmode='pan')
    return fig

def plot_whale_activity_interactive(calls_df, current_strike):
    strikes = sorted(calls_df['strike'].unique())
    try: idx = strikes.index(current_strike)
    except: idx = 0
    start_idx = max(0, idx - 4); end_idx = min(len(strikes), idx + 5)
    subset = calls_df[calls_df['strike'].isin(strikes[start_idx:end_idx])].copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=subset['strike'], y=subset['openInterest'], name='OI', marker_color='#4DA6FF'))
    fig.add_trace(go.Bar(x=subset['strike'], y=subset['volume'], name='Vol', marker_color='#00FF7F'))
    fig.update_layout(title="Whale Detector", template="plotly_dark", height=450, dragmode='pan')
    return fig

def plot_flow_battle_interactive(calls, puts, current_strike):
    c_vol = calls[['strike', 'volume']].groupby('strike').sum().rename(columns={'volume': 'Call Vol'})
    p_vol = puts[['strike', 'volume']].groupby('strike').sum().rename(columns={'volume': 'Put Vol'})
    df = pd.merge(c_vol, p_vol, on='strike', how='outer').fillna(0)
    strikes = sorted(df.index)
    try: idx = strikes.index(current_strike)
    except: idx = (np.abs(np.array(strikes) - current_strike)).argmin()
    start_idx = max(0, idx - 5); end_idx = min(len(strikes), idx + 6)
    subset = df.iloc[start_idx:end_idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=subset.index, y=subset['Call Vol'], name='Bulls', marker_color='#00FF7F'))
    fig.add_trace(go.Bar(x=subset.index, y=subset['Put Vol'], name='Bears', marker_color='#FF4B4B'))
    fig.update_layout(title="Battle Map", template="plotly_dark", height=450, dragmode='pan')
    return fig

# --- 2. STATIC PLOTS (MATPLOTLIB - FOR WORD REPORT) ---
def create_static_plots_for_report(ticker, S, K, days, iv, calls):
    plots = {}
    
    # Static Whale
    strikes = sorted(calls['strike'].unique())
    try: idx = strikes.index(K)
    except: idx = 0
    sub = calls[calls['strike'].isin(strikes[max(0, idx-3):min(len(strikes), idx+4)])]
    
    fig1, ax1 = plt.subplots(figsize=(6,3))
    x = np.arange(len(sub))
    ax1.bar(x-0.2, sub['openInterest'], 0.4, label='OI', color='#4DA6FF')
    ax1.bar(x+0.2, sub['volume'], 0.4, label='Vol', color='#00FF7F')
    ax1.set_xticks(x); ax1.set_xticklabels(sub['strike'].astype(int))
    ax1.set_title(f"Whale Activity: {ticker}"); ax1.legend()
    b1 = BytesIO(); fig1.savefig(b1, format='png'); b1.seek(0); plots['whale'] = b1
    plt.close(fig1)

    # Static Sim
    prices = np.linspace(K*0.8, K*1.2, 50)
    T = max(days/365, 0.001)
    base = black_scholes_price(S, K, T, 0.045, iv)
    pnl = [black_scholes_price(p, K, T, 0.045, iv) - base for p in prices]
    
    fig2, ax2 = plt.subplots(figsize=(6,3))
    ax2.plot(prices, pnl, color='#4DA6FF')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.axvline(S, color='red', linestyle=':')
    ax2.set_title("P&L Simulator"); ax2.grid(True, alpha=0.3)
    b2 = BytesIO(); fig2.savefig(b2, format='png'); b2.seek(0); plots['sim'] = b2
    plt.close(fig2)
    
    return plots

# --- REPORT GENERATOR ---
def generate_report(ticker, S, K, date, d, g, t, ai_txt, plots, scan_res):
    doc = Document()
    doc.add_heading(f"TRADING REPORT: {ticker}", 0)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Summary
    doc.add_heading("1. Executive Summary", 1)
    table = doc.add_table(rows=2, cols=4)
    table.style = 'Table Grid'
    row = table.rows[1].cells
    row[0].text = ticker; row[1].text = f"${S:.2f}"; row[2].text = f"${K}"; row[3].text = str(date)
    
    # Greeks
    doc.add_heading("2. Greeks", 1)
    doc.add_paragraph(f"Delta: {d:.2f} | Theta: {t:.3f} | Gamma: {g:.3f}")
    
    # AI
    doc.add_heading("3. AI Analysis", 1)
    doc.add_paragraph(ai_txt if ai_txt else "No AI analysis run.")
    
    # Charts
    doc.add_heading("4. Charts", 1)
    doc.add_picture(plots['whale'], width=Inches(5.5))
    doc.add_picture(plots['sim'], width=Inches(5.5))
    
    # Scan Results
    if scan_res is not None and not scan_res.empty:
        doc.add_heading("5. ATM Scan Results", 1)
        st_table = doc.add_table(rows=1, cols=4)
        st_table.style = 'Table Grid'
        h = st_table.rows[0].cells
        h[0].text = 'Ticker'; h[1].text = 'Strike'; h[2].text = 'Price'; h[3].text = 'Vol'
        for _, r in scan_res.head(15).iterrows():
            row = st_table.add_row().cells
            row[0].text = str(r['Ticker'])
            row[1].text = str(r['ATM Strike'])
            row[2].text = str(r['Option Price'])
            row[3].text = str(r['Volume'])

    b = BytesIO(); doc.save(b); return b

# --- SCANNER ---
def run_scan(tickers):
    res = []
    bar = st.progress(0); txt = st.empty()
    for i, t in enumerate(tickers):
        time.sleep(random.uniform(0.5, 1.5)) # Anti-block
        try:
            txt.text(f"Scanning {t}...")
            stk = yf.Ticker(t)
            curr = stk.history(period='1d')['Close'].iloc[-1]
            dates = stk.options
            if not dates: continue
            calls = stk.option_chain(dates[0]).calls
            
            # ATM Logic: Smallest difference between Strike and Current Price
            calls['diff'] = abs(calls['strike'] - curr)
            atm = calls.loc[calls['diff'].idxmin()]
            
            res.append({
                'Ticker': t, 'ATM Strike': atm['strike'], 'Exp Date': dates[0],
                'Option Price': atm['lastPrice'], 'Volume': atm['volume'], 'Open Int': atm['openInterest']
            })
        except: pass
        bar.progress((i+1)/len(tickers))
    txt.empty(); bar.empty()
    return pd.DataFrame(res)

# --- MAIN APP ---
st.sidebar.markdown("## ⚙️ Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="PEP").upper()
strike_price = st.sidebar.number_input("Strike Price ($)", value=148.0)

if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear(); st.rerun()

if ticker:
    try:
        stock_conn = get_ticker_object(ticker)
        with st.spinner('Fetching market data...'):
            history, info = get_stock_history_and_info(ticker)
            current_price = info.get('currentPrice', history['Close'].iloc[-1])
            prev_close = info.get('previousClose', history['Close'].iloc[-2])
            
            expirations = stock_conn.options
            if not expirations: st.error("No options data found."); st.stop()
            selected_date = st.sidebar.selectbox("Expiration Date", expirations)
            
            full_chain, calls, puts = get_option_chain_data(ticker, selected_date)
            specific_contract = calls.iloc[(calls['strike'] - strike_price).abs().argsort()[:1]]
            contract_iv = specific_contract.iloc[0]['impliedVolatility']
            
            days_left = (datetime.strptime(selected_date, "%Y-%m-%d") - datetime.now()).days
            if days_left < 1: days_left = 1
            theo_price = black_scholes_price(current_price, strike_price, days_left/365, 0.045, contract_iv)
            d, g, t = calculate_greeks(current_price, strike_price, days_left/365, 0.045, contract_iv)

        st.title(f"📊 {ticker} Command Center 🔒")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}", f"{current_price - prev_close:.2f}")
        col2.metric("Your Strike", f"${strike_price:.2f}")
        col3.metric("Selected Expiration", selected_date)
        st.markdown("---")

        # --- 13 TABS RESTORED ---
        tabs = st.tabs([
            "1. Price", "2. Volume", "3. IV", "4. Rule of 16", 
            "5. Whale Detector", "6. Risk & Profit", "7. Max Pain", "8. News", 
            "9. 🤖 Chart Analyst", "10. 💬 Strategy Engine", "11. 🔮 Future Simulator", "12. 🌊 Flow Monitor",
            "13. 🔍 ATM Scanner"
        ])

        with tabs[0]: st.line_chart(history['Close'])
        with tabs[1]: st.metric("Volume", f"{info.get('volume', 0):,}")
        with tabs[2]: st.metric("Implied Volatility", f"{contract_iv * 100:.2f}%")
        with tabs[3]: st.metric("Expected Daily Move", f"${(contract_iv * 100 / 16) / 100 * current_price:.2f}")

        with tabs[4]: # Whale
            st.header("Whale Detector")
            st.plotly_chart(plot_whale_activity_interactive(calls, strike_price), use_container_width=True, config={'scrollZoom': True})

        with tabs[5]: # Greeks
            st.header("Risk & Profit Hub")
            c1, c2, c3 = st.columns(3)
            c1.metric("Delta", f"{d:.2f}"); c2.metric("Gamma", f"{g:.3f}"); c3.metric("Theta", f"{t:.3f}")
            st.plotly_chart(plot_greeks_interactive(current_price, strike_price, days_left, contract_iv), use_container_width=True, config={'scrollZoom': True})

        with tabs[6]: st.metric("Max Pain", f"${calculate_max_pain(full_chain):.2f}")
        with tabs[7]:
            try: 
                for item in stock_conn.news[:3]: st.markdown(f"- [{item['title']}]({item['link']})")
            except: st.write("No news found.")

        with tabs[8]: # AI
            st.header("🤖 AI Chart Analyst")
            if "ai_result" not in st.session_state: st.session_state["ai_result"] = ""
            up_files = st.file_uploader("Upload Charts", type=["jpg", "png"], accept_multiple_files=True)
            if up_files and st.button("Analyze Images"):
                if "api_keys" in st.secrets:
                    genai.configure(api_key=st.secrets["api_keys"]["gemini"])
                    try:
                        model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
                        prompt = "Analyze these charts for walls and traps."
                        content = [prompt] + [Image.open(f) for f in up_files]
                        with st.spinner("Analyzing..."):
                            st.session_state["ai_result"] = model.generate_content(content).text
                            st.rerun()
                    except Exception as e: st.error(str(e))
            if st.session_state["ai_result"]: st.write(st.session_state["ai_result"])

        with tabs[9]: # Strategy
            st.header("💬 Strategy Engine")
            q = st.text_input("Ask a strategy question:")
            if q and st.button("Ask AI"):
                 if "api_keys" in st.secrets:
                    genai.configure(api_key=st.secrets["api_keys"]["gemini"])
                    model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
                    st.write(model.generate_content(f"Context: {ticker} ${strike_price}. Question: {q}").text)

        with tabs[10]: # Sim
            st.header("🔮 Future Simulator")
            st.plotly_chart(plot_simulation_interactive(current_price, strike_price, days_left, contract_iv), use_container_width=True, config={'scrollZoom': True})

        with tabs[11]: # Flow
            st.header("🌊 Market Flow")
            st.plotly_chart(plot_flow_battle_interactive(calls, puts, strike_price), use_container_width=True, config={'scrollZoom': True})

        with tabs[12]: # Scanner
            st.header("🔍 ATM Options Scanner")
            up_xl = st.file_uploader("Upload Excel List", type=['xlsx'])
            if up_xl and st.button("🚀 Start ATM Scan"):
                df_input = pd.read_excel(up_xl)
                tickers = df_input.iloc[:, 0].dropna().astype(str).tolist()
                st.session_state["scan_results"] = run_scan(tickers)
            
            if "scan_results" in st.session_state:
                st.dataframe(st.session_state["scan_results"], use_container_width=True)
                # Excel Download
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    st.session_state["scan_results"].to_excel(writer, index=False)
                st.download_button("📥 Download Excel", buffer.getvalue(), "ATM_Scan.xlsx", "application/vnd.ms-excel")

        # --- REPORT EXPORT (SIDEBAR) ---
        st.sidebar.markdown("---")
        safe_plots = create_static_plots_for_report(ticker, current_price, strike_price, days_left, contract_iv, calls)
        scan_data = st.session_state.get("scan_results", None)
        
        report_file = generate_report(
            ticker, current_price, strike_price, selected_date, d, g, t, 
            st.session_state.get("ai_result", ""), 
            safe_plots, 
            scan_data
        )
        st.sidebar.download_button("📄 Download Full Dossier", report_file.getvalue(), f"{ticker}_Dossier.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    except Exception as e:
        st.error(f"Waiting for data... ({e})")
