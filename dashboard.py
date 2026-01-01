import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
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
            if "Too Many Requests" in str(e) or "404" in str(e):
                time.sleep((i + 1) * 2)
                continue
            raise e
    return func(*args)

@st.cache_data(ttl=900) 
def get_stock_history_and_info(ticker_symbol):
    def _get():
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period="1mo") # Grab 1 month for charts
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

# --- FIGURE GENERATORS (Separated for Reporting) ---
def create_greeks_fig(current_price, strike, days_left, iv, risk_free=0.045):
    prices = np.linspace(strike * 0.8, strike * 1.2, 100)
    T = max(days_left / 365.0, 0.001)
    deltas = [calculate_greeks(p, strike, T, risk_free, iv)[0] for p in prices]
    gammas = [calculate_greeks(p, strike, T, risk_free, iv)[1] for p in prices]
    curr_d, _, _ = calculate_greeks(current_price, strike, T, risk_free, iv)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=deltas, mode='lines', name='Delta', line=dict(color='#4DA6FF', width=3)))
    fig.add_trace(go.Scatter(x=prices, y=gammas, mode='lines', name='Gamma', line=dict(color='#00FF7F', width=2, dash='dash'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=[current_price], y=[curr_d], mode='markers', name='You', marker=dict(color='white', size=10)))
    fig.update_layout(title="Greeks Curve", template="plotly_dark", height=400, yaxis2=dict(overlaying="y", side="right"))
    return fig

def create_sim_fig(S, K, days_left, iv, r=0.045, purchase_price=0):
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
    fig.update_layout(title="Future P&L Simulator", template="plotly_dark", height=400)
    return fig

def create_whale_fig(calls_df, current_strike):
    strikes = sorted(calls_df['strike'].unique())
    try: idx = strikes.index(current_strike)
    except: idx = 0
    start_idx = max(0, idx - 3); end_idx = min(len(strikes), idx + 4)
    subset = calls_df[calls_df['strike'].isin(strikes[start_idx:end_idx])].copy()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=subset['strike'], y=subset['openInterest'], name='OI (Old)', marker_color='#4DA6FF'))
    fig.add_trace(go.Bar(x=subset['strike'], y=subset['volume'], name='Vol (New)', marker_color='#00FF7F'))
    fig.update_layout(title="Whale Detector", template="plotly_dark", height=400, barmode='group')
    return fig

def create_battle_fig(calls, puts, current_strike):
    c_vol = calls[['strike', 'volume']].groupby('strike').sum()
    p_vol = puts[['strike', 'volume']].groupby('strike').sum()
    df = pd.merge(c_vol, p_vol, on='strike', how='outer').fillna(0)
    strikes = sorted(df.index)
    try: idx = strikes.index(current_strike)
    except: idx = (np.abs(np.array(strikes) - current_strike)).argmin()
    start_idx = max(0, idx - 4); end_idx = min(len(strikes), idx + 5)
    subset = df.iloc[start_idx:end_idx]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=subset.index, y=subset['volume_x'], name='Bulls', marker_color='#00FF7F'))
    fig.add_trace(go.Bar(x=subset.index, y=subset['volume_y'], name='Bears', marker_color='#FF4B4B'))
    fig.update_layout(title="Battle Map", template="plotly_dark", height=400, barmode='group')
    return fig

# --- WORD REPORT GENERATOR (FULL DOSSIER) ---
def generate_full_report(ticker, price, strike, exp, d, g, t, ai_text, fig_whale, fig_sim, fig_battle):
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
    doc.add_paragraph(f"Delta ({d:.2f}): Directional exposure per share.")
    doc.add_paragraph(f"Theta ({t:.3f}): Time decay per day (approx ${abs(t)*100:.2f}).")
    doc.add_paragraph(f"Gamma ({g:.3f}): Acceleration factor.")
    
    # 3. AI Analysis
    doc.add_heading('3. AI Strategic Analysis', level=1)
    doc.add_paragraph(ai_text if ai_text else "No AI analysis was run for this session.")
    
    # 4. Charts (Images)
    doc.add_heading('4. Intelligence Visuals', level=1)
    
    try:
        # Whale Detector
        doc.add_heading('Whale Activity (Volume vs OI)', level=2)
        img_bytes = fig_whale.to_image(format="png", width=600, height=300)
        doc.add_picture(BytesIO(img_bytes), width=Inches(6))
        
        # Simulator
        doc.add_heading('Future P&L Simulator', level=2)
        img_bytes_sim = fig_sim.to_image(format="png", width=600, height=300)
        doc.add_picture(BytesIO(img_bytes_sim), width=Inches(6))
        
        # Battle Map
        doc.add_heading('Bull vs Bear Battle Map', level=2)
        img_bytes_bat = fig_battle.to_image(format="png", width=600, height=300)
        doc.add_picture(BytesIO(img_bytes_bat), width=Inches(6))
        
    except Exception as e:
        doc.add_paragraph(f"[Error generating chart images: {e}. Please ensure 'kaleido' is installed in requirements.txt]")

    # Save
    bio = BytesIO()
    doc.save(bio)
    return bio

def scan_atm_options(tickers):
    results = []
    progress = st.progress(0)
    status = st.empty()
    for i, t in enumerate(tickers):
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
                "Ticker": t, "Price": curr, "Strike": atm['strike'], "Exp": dates[0],
                "Last": atm['lastPrice'], "Vol": atm['volume'], "OI": atm['openInterest']
            })
        except: pass
        progress.progress((i+1)/len(tickers))
    status.empty(); progress.empty()
    return pd.DataFrame(results)

# --- MAIN APP ---
st.sidebar.markdown("## ⚙️ Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="PEP").upper()
strike_price = st.sidebar.number_input("Strike Price ($)", value=148.0)

if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear()
    st.session_state["ai_result"] = "" 
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

        # --- PRE-GENERATE FIGURES FOR REPORT ---
        fig_whale = create_whale_fig(calls, strike_price)
        fig_greeks = create_greeks_fig(current_price, strike_price, days_left, contract_iv)
        fig_sim = create_sim_fig(current_price, strike_price, days_left, contract_iv, 0.045, theo_price)
        fig_flow = create_battle_fig(calls, puts, strike_price)

        st.title(f"📊 {ticker} Command Center 🔒")

        # --- TABS ---
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
        with tabs[4]: st.header("Whale Detector"); st.plotly_chart(fig_whale, use_container_width=True)
        with tabs[5]: 
            st.header("Risk & Profit Hub")
            c1, c2, c3 = st.columns(3)
            c1.metric("Delta", f"{d:.2f}"); c2.metric("Gamma", f"{g:.3f}"); c3.metric("Theta", f"{t:.3f}")
            st.plotly_chart(fig_greeks, use_container_width=True)
        with tabs[6]: st.metric("Max Pain", f"${calculate_max_pain(full_chain):.2f}")
        with tabs[7]:
            try: 
                for item in stock_conn.news[:3]: st.markdown(f"- [{item['title']}]({item['link']})")
            except: st.write("No news found.")

        # --- TAB 9: AI (PERSISTENT) ---
        with tabs[8]:
            st.header("🤖 AI Chart Analyst")
            if "ai_result" not in st.session_state: st.session_state["ai_result"] = ""
            
            uploaded_files = st.file_uploader("Upload Charts", type=["jpg", "png"], accept_multiple_files=True)
            if uploaded_files and st.button("Analyze Images"):
                if "api_keys" in st.secrets:
                    genai.configure(api_key=st.secrets["api_keys"]["gemini"])
                    try:
                        model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
                        prompt = "You are a Senior Trader. Analyze these charts for 'Walls', 'Squeezes', and 'Traps'."
                        content = [prompt] + [Image.open(f) for f in uploaded_files]
                        with st.spinner("Analyzing..."):
                            resp = model.generate_content(content)
                            st.session_state["ai_result"] = resp.text
                            st.rerun()
                    except Exception as e: st.error(str(e))
            
            if st.session_state["ai_result"]:
                st.markdown("### 📝 Analysis Report"); st.write(st.session_state["ai_result"])

        with tabs[9]: st.header("💬 Strategy Engine"); st.info("Chat functionality ready.")
        with tabs[10]: st.header("🔮 Simulator"); st.plotly_chart(fig_sim, use_container_width=True)
        with tabs[11]: st.header("🌊 Market Flow"); st.plotly_chart(fig_flow, use_container_width=True)
        
        # --- TAB 13: EXCEL SCANNER ---
        with tabs[12]:
            st.header("🔍 ATM Options Scanner")
            up_file = st.file_uploader("Upload Excel List", type=['xlsx'])
            if up_file and st.button("🚀 Scan List"):
                df = pd.read_excel(up_file)
                tickers = df.iloc[:,0].astype(str).tolist()
                res = scan_atm_options(tickers)
                st.session_state["scan_res"] = res
            
            if "scan_res" in st.session_state:
                st.dataframe(st.session_state["scan_res"], use_container_width=True)
                # EXCEL DOWNLOAD
                buff = BytesIO()
                with pd.ExcelWriter(buff, engine='xlsxwriter') as writer:
                    st.session_state["scan_res"].to_excel(writer, index=False)
                st.download_button("📥 Download Excel", buff.getvalue(), "ATM_Scan.xlsx", "application/vnd.ms-excel")

        # --- MAIN REPORT DOWNLOAD BUTTON (SIDEBAR) ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📄 Export")
        
        # Generate the report binary
        report_file = generate_full_report(
            ticker, current_price, strike_price, selected_date, d, g, t, 
            st.session_state.get("ai_result", ""), 
            fig_whale, fig_sim, fig_flow
        )
        
        st.sidebar.download_button(
            label="Download Full Dossier (Word)",
            data=report_file.getvalue(),
            file_name=f"{ticker}_Full_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    except Exception as e:
        st.error(f"Waiting for data... ({e})")
