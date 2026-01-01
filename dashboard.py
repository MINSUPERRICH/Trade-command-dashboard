import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import matplotlib.pyplot as plt # CRITICAL FOR REPORT
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

# --- HELPER FUNCTIONS (Anti-Block) ---
def fetch_with_retry(func, *args, retries=5):
    for i in range(retries):
        try:
            return func(*args)
        except Exception as e:
            error_msg = str(e).lower()
            if "too many requests" in error_msg or "429" in error_msg:
                wait_time = (2 ** i) + random.uniform(0, 1) 
                time.sleep(wait_time)
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

# --- REPORT GENERATOR (THE BULLETPROOF VERSION) ---
def generate_safe_report_plots(ticker, price, strike, days_left, iv, calls, puts):
    # This function creates Matplotlib charts (Server Safe) instead of Plotly
    plots = {}
    
    # 1. Whale Chart
    strikes = sorted(calls['strike'].unique())
    try: idx = strikes.index(strike)
    except: idx = 0
    start = max(0, idx - 3); end = min(len(strikes), idx + 4)
    subset = calls[calls['strike'].isin(strikes[start:end])]
    
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    x = np.arange(len(subset))
    width = 0.35
    ax1.bar(x - width/2, subset['openInterest'], width, label='Open Int', color='#4DA6FF')
    ax1.bar(x + width/2, subset['volume'], width, label='Volume', color='#00FF7F')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subset['strike'].astype(int))
    ax1.set_title(f"Whale Activity: {ticker}")
    ax1.legend()
    buf1 = BytesIO(); fig1.savefig(buf1, format="png"); buf1.seek(0)
    plots['whale'] = buf1
    plt.close(fig1)

    # 2. Simulator Chart
    prices = np.linspace(strike * 0.8, strike * 1.2, 50)
    T1 = max(days_left / 365.0, 0.001)
    theo = black_scholes_price(price, strike, T1, 0.045, iv)
    pnl = [black_scholes_price(p, strike, T1, 0.045, iv) - theo for p in prices]
    
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.plot(prices, pnl, color='#4DA6FF', linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.axvline(price, color='red', linestyle=':', label='Current Price')
    ax2.set_title("P&L Simulator (Today)")
    ax2.grid(True, alpha=0.3)
    buf2 = BytesIO(); fig2.savefig(buf2, format="png"); buf2.seek(0)
    plots['sim'] = buf2
    plt.close(fig2)

    return plots

def generate_full_report(ticker, price, strike, exp, d, g, t, ai_text, plots, scan_df=None):
    doc = Document()
    doc.add_heading(f'CONFIDENTIAL TRADING DOSSIER: {ticker}', 0)
    doc.add_paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 1. Executive Summary
    doc.add_heading('1. Executive Summary', level=1)
    table = doc.add_table(rows=2, cols=4)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells; row = table.rows[1].cells
    hdr[0].text = 'Asset'; hdr[1].text = 'Price'; hdr[2].text = 'Target Strike'; hdr[3].text = 'Expiration'
    row[0].text = ticker; row[1].text = f"${price:.2f}"; row[2].text = f"${strike}"; row[3].text = str(exp)
    
    # 2. Risk Profile
    doc.add_heading('2. Risk Profile (The Greeks)', level=1)
    doc.add_paragraph(f"Delta ({d:.2f}): Directional exposure.")
    doc.add_paragraph(f"Theta ({t:.3f}): Time decay per day.")
    doc.add_paragraph(f"Gamma ({g:.3f}): Acceleration factor.")
    
    # 3. AI Analysis
    doc.add_heading('3. AI Strategic Analysis', level=1)
    doc.add_paragraph(ai_text if ai_text else "No AI analysis was run for this session.")
    
    # 4. Intelligence Visuals (Safe Images)
    doc.add_heading('4. Intelligence Visuals', level=1)
    doc.add_paragraph("Whale Activity Detector:")
    doc.add_picture(plots['whale'], width=Inches(5.5))
    doc.add_paragraph("P&L Simulation Curve:")
    doc.add_picture(plots['sim'], width=Inches(5.5))

    # 5. ATM Scanner Results (NEW!)
    if scan_df is not None and not scan_df.empty:
        doc.add_heading('5. ATM Scanner Results', level=1)
        doc.add_paragraph("Top ATM Candidates found in your Scan:")
        
        # Create Table with Header
        t = doc.add_table(rows=1, cols=5)
        t.style = 'Table Grid'
        hdr_cells = t.rows[0].cells
        headers = ['Ticker', 'Price', 'Strike', 'Last', 'Vol']
        for i, h in enumerate(headers): hdr_cells[i].text = h
        
        # Fill Rows (Limit to top 20 to save space)
        for index, row in scan_df.head(20).iterrows():
            row_cells = t.add_row().cells
            row_cells[0].text = str(row['Ticker'])
            row_cells[1].text = str(row['Price']) if 'Price' in row else str(row['Stock Price'])
            row_cells[2].text = str(row['Strike']) if 'Strike' in row else str(row['ATM Strike'])
            row_cells[3].text = str(row['Last']) if 'Last' in row else str(row['Option Price'])
            row_cells[4].text = str(row['Vol']) if 'Vol' in row else str(row['Volume'])

    bio = BytesIO()
    doc.save(bio)
    return bio

# --- SCANNER ---
def scan_atm_options(tickers):
    results = []
    progress = st.progress(0)
    status = st.empty()
    for i, t in enumerate(tickers):
        time.sleep(random.uniform(1.0, 2.0)) # Anti-Block Pause
        try:
            status.text(f"Scanning {t}...")
            stock = yf.Ticker(t)
            curr = stock.history(period="1d")['Close'].iloc[-1]
            dates = stock.options
            if not dates: continue
            chain = stock.option_chain(dates[0]).calls
            chain['diff'] = abs(chain['strike'] - curr)
            atm = chain.loc[chain['diff'].idxmin()]
            results.append({
                "Ticker": t, "Price": round(curr, 2), "Strike": atm['strike'], "Exp": dates[0],
                "Last": atm['lastPrice'], "Vol": atm['volume'], "OI": atm['openInterest']
            })
        except: pass
        progress.progress((i+1)/len(tickers))
    status.empty(); progress.empty()
    return pd.DataFrame(results)

# --- PLOTLY HELPERS (Interactive for UI) ---
def create_plotly_figs(calls, puts, strike):
    # Whale
    strikes = sorted(calls['strike'].unique())
    try: idx = strikes.index(strike)
    except: idx = 0
    start = max(0, idx - 3); end = min(len(strikes), idx + 4)
    sub = calls[calls['strike'].isin(strikes[start:end])]
    fig_whale = go.Figure()
    fig_whale.add_trace(go.Bar(x=sub['strike'], y=sub['openInterest'], name='OI', marker_color='#4DA6FF'))
    fig_whale.add_trace(go.Bar(x=sub['strike'], y=sub['volume'], name='Vol', marker_color='#00FF7F'))
    fig_whale.update_layout(title="Whale Detector", template="plotly_dark", height=400)
    
    # Battle
    c_vol = calls[['strike', 'volume']].groupby('strike').sum()
    p_vol = puts[['strike', 'volume']].groupby('strike').sum()
    df = pd.merge(c_vol, p_vol, on='strike', how='outer').fillna(0)
    sub_b = df.iloc[max(0, len(df)//2 - 4): min(len(df), len(df)//2 + 5)]
    fig_battle = go.Figure()
    fig_battle.add_trace(go.Bar(x=sub_b.index, y=sub_b['volume_x'], name='Bulls', marker_color='#00FF7F'))
    fig_battle.add_trace(go.Bar(x=sub_b.index, y=sub_b['volume_y'], name='Bears', marker_color='#FF4B4B'))
    fig_battle.update_layout(title="Battle Map", template="plotly_dark", height=400)
    
    return fig_whale, fig_battle

def create_sim_plot(S, K, days, iv, theo):
    prices = np.linspace(S*0.8, S*1.2, 100)
    T = max(days/365, 0.001)
    pnl = [black_scholes_price(p, K, T, 0.045, iv) - theo for p in prices]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=pnl, mode='lines', name='Today', line=dict(color='#4DA6FF')))
    fig.add_hline(y=0, line_color="white", opacity=0.5)
    fig.update_layout(title="Simulator", template="plotly_dark", height=400)
    return fig

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
            selected_date = st.sidebar.selectbox("Expiration Date", expirations)
            
            full_chain, calls, puts = get_option_chain_data(ticker, selected_date)
            specific_contract = calls.iloc[(calls['strike'] - strike_price).abs().argsort()[:1]]
            contract_iv = specific_contract.iloc[0]['impliedVolatility']
            days_left = (datetime.strptime(selected_date, "%Y-%m-%d") - datetime.now()).days
            if days_left < 1: days_left = 1
            theo_price = black_scholes_price(current_price, strike_price, days_left/365, 0.045, contract_iv)
            d, g, t = calculate_greeks(current_price, strike_price, days_left/365, 0.045, contract_iv)

        st.title(f"📊 {ticker} Command Center 🔒")

        # --- TABS ---
        tabs = st.tabs([
            "1. Price", "2. Volume", "3. IV", "4. Rule of 16", 
            "5. Whale Detector", "6. Risk & Profit", "7. Max Pain", "8. News", 
            "9. 🤖 Chart Analyst", "10. 💬 Strategy Engine", "11. 🔮 Future Simulator", "12. 🌊 Flow Monitor",
            "13. 🔍 ATM Scanner"
        ])

        fig_whale, fig_battle = create_plotly_figs(calls, puts, strike_price)
        fig_sim = create_sim_plot(current_price, strike_price, days_left, contract_iv, theo_price)

        with tabs[0]: st.line_chart(history['Close'])
        with tabs[1]: st.metric("Volume", f"{info.get('volume', 0):,}")
        with tabs[2]: st.metric("IV", f"{contract_iv * 100:.2f}%")
        with tabs[3]: st.metric("Exp Move", f"${(contract_iv * 100 / 16) / 100 * current_price:.2f}")
        with tabs[4]: st.plotly_chart(fig_whale, use_container_width=True)
        with tabs[5]: 
            c1, c2, c3 = st.columns(3)
            c1.metric("Delta", f"{d:.2f}"); c2.metric("Gamma", f"{g:.3f}"); c3.metric("Theta", f"{t:.3f}")
        with tabs[6]: st.metric("Max Pain", f"${calculate_max_pain(full_chain):.2f}")
        
        with tabs[8]:
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

        with tabs[10]: st.plotly_chart(fig_sim, use_container_width=True)
        with tabs[11]: st.plotly_chart(fig_battle, use_container_width=True)
        
        # --- TAB 13: SCANNER ---
        with tabs[12]:
            st.header("🔍 ATM Scanner")
            up_list = st.file_uploader("Upload Excel", type=['xlsx'])
            if up_list and st.button("🚀 Scan"):
                df = pd.read_excel(up_list)
                st.session_state["scan_res"] = scan_atm_options(df.iloc[:,0].astype(str).tolist())
            
            if "scan_res" in st.session_state:
                st.dataframe(st.session_state["scan_res"])

        # --- REPORT EXPORT ---
        st.sidebar.markdown("---")
        # Generate SAFE plots for Word
        safe_plots = generate_safe_report_plots(ticker, current_price, strike_price, days_left, contract_iv, calls, puts)
        # Pass Scan Results if they exist
        scan_results = st.session_state.get("scan_res", None)
        
        report_file = generate_full_report(
            ticker, current_price, strike_price, selected_date, d, g, t, 
            st.session_state.get("ai_result", ""), 
            safe_plots, scan_results
        )
        
        st.sidebar.download_button("📄 Download Full Dossier", report_file.getvalue(), f"{ticker}_Dossier.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    except Exception as e:
        st.error(f"Waiting for data... ({e})")
