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
def get_stock_data(ticker):
    def _get():
        stock = yf.Ticker(ticker)
        return stock.history(period="5d"), stock.info
    return fetch_with_retry(_get)

@st.cache_data(ttl=900)
def get_option_chain(ticker, date):
    def _get():
        stock = yf.Ticker(ticker)
        opt = stock.option_chain(date)
        calls = opt.calls; calls['type'] = 'call'
        puts = opt.puts; puts['type'] = 'put'
        return pd.concat([calls, puts]), calls, puts
    return fetch_with_retry(_get)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call': delta = norm.cdf(d1)
    else: delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    term1 = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    theta = (term1 - term2) / 365.0
    return delta, gamma, theta

def plot_greeks(S, K, T, iv):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    prices = np.linspace(K * 0.8, K * 1.2, 100)
    deltas = []; gammas = []
    for p in prices:
        d, g, t = calculate_greeks(p, K, max(T, 0.001), 0.045, iv)
        deltas.append(d); gammas.append(g)
    ax1.plot(prices, deltas, color='#4DA6FF', linewidth=3, label='Delta')
    ax2 = ax1.twinx()
    ax2.plot(prices, gammas, color='#00FF7F', linewidth=2, linestyle='--', label='Gamma')
    curr_d, _, _ = calculate_greeks(S, K, max(T, 0.001), 0.045, iv)
    ax1.scatter([S], [curr_d], color='white', s=100, zorder=10)
    ax1.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117')
    ax1.tick_params(colors='white'); ax2.tick_params(colors='white')
    return fig

def plot_whale(calls, strike):
    try:
        strikes = sorted(calls['strike'].unique())
        idx = strikes.index(strike)
        subset = calls[calls['strike'].isin(strikes[max(0, idx-2):min(len(strikes), idx+3)])]
    except: subset = calls.head(5)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(subset)); width = 0.35
    ax.bar(x - width/2, subset['openInterest'], width, label='Open Interest', color='#4DA6FF', alpha=0.6)
    ax.bar(x + width/2, subset['volume'], width, label='Volume', color='#00FF7F')
    ax.set_xticks(x); ax.set_xticklabels(subset['strike'], color='white')
    ax.legend(facecolor='#262730', labelcolor='white')
    ax.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    return fig

# --- MAIN APP ---
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Ticker", value="PEP").upper()
strike = st.sidebar.number_input("Strike ($)", value=148.0)

if ticker:
    try:
        stock = yf.Ticker(ticker)
        history, info = get_stock_data(ticker)
        curr_price = info.get('currentPrice', history['Close'].iloc[-1])
        
        exps = stock.options
        if not exps: st.stop()
        sel_date = st.sidebar.selectbox("Expiration", exps)
        
        full, calls, puts = get_option_chain(ticker, sel_date)
        contract = calls.iloc[(calls['strike'] - strike).abs().argsort()[:1]]
        iv = contract.iloc[0]['impliedVolatility']
        
        st.title(f"📊 {ticker} Command Center")
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"${curr_price:.2f}")
        c2.metric("Strike", f"${strike:.2f}")
        c3.metric("Date", sel_date)
        
        tabs = st.tabs(["1. Whale Detector", "2. Risk/Profit", "3. 🤖 AI Analyst"])
        
        with tabs[0]:
            st.pyplot(plot_whale(calls, strike))
            
        with tabs[1]:
            days = (datetime.strptime(sel_date, "%Y-%m-%d") - datetime.now()).days
            d, g, t = calculate_greeks(curr_price, strike, days/365, 0.045, iv)
            k1, k2, k3 = st.columns(3)
            k1.metric("Delta", f"{d:.2f}"); k2.metric("Gamma", f"{g:.3f}"); k3.metric("Theta", f"{t:.3f}")
            st.pyplot(plot_greeks(curr_price, strike, days/365, iv))
            
            st.write("---")
            target = st.number_input("Desired Profit ($)", 50)
            if d > 0.01:
                move = (target/100)/d
                st.info(f"To make **${target}**, {ticker} needs to hit **${curr_price + move:.2f}**")

        with tabs[2]:
            st.header("🤖 AI Chart Analyst")
            files = st.file_uploader("Upload Charts", accept_multiple_files=True)
            
            if st.button("Analyze"):
                if "api_keys" in st.secrets:
                    key = st.secrets["api_keys"]["gemini"]
                    genai.configure(api_key=key)
                    
                    # 1. TRY TO FIND THE RIGHT MODEL
                    try:
                        # Priority 1: Flash 1.5
                        model = genai.GenerativeModel('gemini-1.5-flash')
                    except:
                        try:
                            # Priority 2: Pro Vision (Old reliable)
                            model = genai.GenerativeModel('gemini-pro-vision')
                        except:
                            st.error("❌ Could not find a working model. Please update requirements.txt!")
                            st.stop()
                            
                    # 2. ANALYZE
                    if files:
                        imgs = [Image.open(f) for f in files]
                        st.image(imgs, width=200)
                        with st.spinner(f"Analyzing with {model.model_name}..."):
                            try:
                                resp = model.generate_content(["Analyze these trading charts. Verdict: Bullish or Bearish?", *imgs])
                                st.write(resp.text)
                            except Exception as e:
                                st.error(f"AI Error: {e}")
                else:
                    st.error("Missing API Key in Secrets!")
            
            # DEBUG TOOL
            with st.expander("🛠️ Debug: Check My Models"):
                if st.button("List Available Models"):
                    if "api_keys" in st.secrets:
                        genai.configure(api_key=st.secrets["api_keys"]["gemini"])
                        try:
                            for m in genai.list_models():
                                if 'generateContent' in m.supported_generation_methods:
                                    st.write(f"- {m.name}")
                        except Exception as e:
                            st.error(f"Error listing models: {e}")

    except Exception as e:
        st.error(f"Loading... ({e})")
