import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go  # For Interactive UI
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

# --- HELPER FUNCTIONS ---
def fetch_with_retry(func, *args, retries=5):
    for i in range(retries):
        try:
            return func(*args)
        except Exception as e:
            if "too many requests" in str(e).lower():
                time.sleep(random.uniform(1, 3))
                continue
            raise e
    return func(*args)

@st.cache_data(ttl=900) 
def get_stock_data(ticker):
    def _get():
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        return hist, stock.info
    return fetch_with_retry(_get)

@st.cache_data(ttl=900)
def get_chain(ticker, date):
    def _get():
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(date)
        return pd.concat([chain.calls, chain.puts]), chain.calls, chain.puts
    return fetch_with_retry(_get)

def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def get_greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = (-(S*sigma*norm.pdf(d1))/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d1 - sigma*np.sqrt(T)))/365
    return delta, gamma, theta

# --- 1. INTERACTIVE PLOTS (FOR UI - PLOTLY) ---
def plot_ui_whale(calls, strike):
    strikes = sorted(calls['strike'].unique())
    try: idx = strikes.index(strike)
    except: idx = (np.abs(np.array(strikes) - strike)).argmin()
    sub = calls[calls['strike'].isin(strikes[max(0, idx-4): min(len(strikes), idx+5)])]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sub['strike'], y=sub['openInterest'], name='OI', marker_color='#4DA6FF'))
    fig.add_trace(go.Bar(x=sub['strike'], y=sub['volume'], name='Vol', marker_color='#00FF7F'))
    fig.update_layout(title="Whale Detector (Interactive)", template="plotly_dark", height=400, dragmode='pan')
    return fig

def plot_ui_sim(S, K, days, iv, theo):
    prices = np.linspace(S*0.8, S*1.2, 100)
    T = max(days/365, 0.001)
    pnl = [black_scholes(p, K, T, 0.045, iv) - theo for p in prices]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=pnl, mode='lines', name='Today', line=dict(color='#4DA6FF', width=3)))
    fig.add_hline(y=0, line_color="white", opacity=0.5)
    fig.update_layout(title="Future Simulator (Interactive)", template="plotly_dark", height=500, dragmode='pan', hovermode="x unified")
    return fig

def plot_ui_battle(calls, puts, strike):
    c = calls.groupby('strike')['volume'].sum(); p = puts.groupby('strike')['volume'].sum()
    df = pd.merge(c, p, on='strike', how='outer').fillna(0).sort_index()
    try: idx = df.index.get_loc(strike)
    except: idx = len(df)//2
    sub = df.iloc[max(0, idx-5):min(len(df), idx+6)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sub.index, y=sub['volume_x'], name='Bulls', marker_color='#00FF7F'))
    fig.add_trace(go.Bar(x=sub.index, y=sub['volume_y'], name='Bears', marker_color='#FF4B4B'))
    fig.update_layout(title="Battle Map (Interactive)", template="plotly_dark", height=400, dragmode='pan')
    return fig

def plot_ui_greeks(S, K, days, iv):
    prices = np.linspace(K*0.8, K*1.2, 100)
    T = max(days/365, 0.001)
    deltas = [get_greeks(p, K, T, 0.045, iv)[0] for p in prices]
    gammas = [get_greeks(p, K, T, 0.045, iv)[1] for p in prices]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=deltas, name='Delta', line=dict(color='#4DA6FF')))
    fig.add_trace(go.Scatter(x=prices, y=gammas, name='Gamma', line=dict(color='#00FF7F', dash='dash'), yaxis='y2'))
    fig.update_layout(
        title="Greeks Profile", template="plotly_dark", height=400, dragmode='pan',
        yaxis2=dict(overlaying='y', side='right')
    )
    return fig

# --- 2. STATIC PLOTS (FOR WORD REPORT - MATPLOTLIB) ---
def create_static_plots(ticker, S, K, days, iv, calls):
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
    base = black_scholes(S, K, T, 0.045, iv)
    pnl = [black_scholes(p, K, T, 0.045, iv) - base for p in prices]
    
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
            row[1].text = str(r['Strike'])
            row[2].text = str(r['Last'])
            row[3].text = str(r['Vol'])

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
                'Ticker': t, 'Strike': atm['strike'], 'Exp': dates[0],
                'Last': atm['lastPrice'], 'Vol': atm['volume'], 'OI': atm['openInterest']
            })
        except: pass
        bar.progress((i+1)/len(tickers))
    txt.empty(); bar.empty()
    return pd.DataFrame(res)

# --- MAIN APP ---
st.sidebar.header("⚙️ Settings")
sym = st.sidebar.text_input("Ticker", "PEP").upper()
strike = st.sidebar.number_input("Strike", 148.0)

if st.sidebar.button("Refresh"): st.cache_data.clear(); st.rerun()

if sym:
    try:
        stock = yf.Ticker(sym)
        with st.spinner("Loading..."):
            hist, info = get_stock_data(sym)
            curr = info.get('currentPrice', hist['Close'].iloc[-1])
            
            dates = stock.options
            exp = st.sidebar.selectbox("Expiration", dates)
            full, calls, puts = get_chain(sym, exp)
            
            days = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
            if days < 1: days = 1
            
            # Find IV for Strike
            try:
                row = calls.iloc[(calls['strike'] - strike).abs().argsort()[:1]]
                iv = row.iloc[0]['impliedVolatility']
            except: iv = 0.3 # Fallback
            
            theo = black_scholes(curr, strike, days/365, 0.045, iv)
            d, g, t = get_greeks(curr, strike, days/365, 0.045, iv)

        st.title(f"🚀 {sym} Command Center")
        
        # UI TABS
        tabs = st.tabs(["Charts", "Whale", "Greeks", "Sim", "Flow", "AI Analyst", "Scanner"])
        
        with tabs[0]: st.line_chart(hist['Close'])
        
        with tabs[1]: # Whale
            st.plotly_chart(plot_ui_whale(calls, strike), use_container_width=True, config={'scrollZoom': True})
            
        with tabs[2]: # Greeks
            c1, c2, c3 = st.columns(3)
            c1.metric("Delta", f"{d:.2f}"); c2.metric("Gamma", f"{g:.3f}"); c3.metric("Theta", f"{t:.3f}")
            st.plotly_chart(plot_ui_greeks(curr, strike, days, iv), use_container_width=True, config={'scrollZoom': True})
            
        with tabs[3]: # Sim
            st.plotly_chart(plot_ui_sim(curr, strike, days, iv, theo), use_container_width=True, config={'scrollZoom': True})
            
        with tabs[4]: # Flow
            st.plotly_chart(plot_ui_battle(calls, puts, strike), use_container_width=True, config={'scrollZoom': True})
            
        with tabs[5]: # AI
            if "ai_res" not in st.session_state: st.session_state["ai_res"] = ""
            up = st.file_uploader("Upload Charts", type=['png','jpg'], accept_multiple_files=True)
            if up and st.button("Analyze"):
                if "api_keys" in st.secrets:
                    genai.configure(api_key=st.secrets["api_keys"]["gemini"])
                    model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
                    st.session_state["ai_res"] = model.generate_content(["Analyze for traps:", *[Image.open(f) for f in up]]).text
            if st.session_state["ai_res"]: st.write(st.session_state["ai_res"])
            
        with tabs[6]: # Scanner
            up_xl = st.file_uploader("Upload Excel List", type=['xlsx'])
            if up_xl and st.button("Scan"):
                df = pd.read_excel(up_xl)
                st.session_state["scan"] = run_scan(df.iloc[:,0].astype(str).tolist())
            if "scan" in st.session_state:
                st.dataframe(st.session_state["scan"])

        # REPORT EXPORT
        st.sidebar.divider()
        safe_plots = create_static_plots(sym, curr, strike, days, iv, calls)
        rpt = generate_report(
            sym, curr, strike, exp, d, g, t, 
            st.session_state.get("ai_res", ""), 
            safe_plots, 
            st.session_state.get("scan", None)
        )
        st.sidebar.download_button("📥 Download Dossier", rpt.getvalue(), f"{sym}_Report.docx")

    except Exception as e:
        st.error(f"Waiting for data... ({str(e)})")
