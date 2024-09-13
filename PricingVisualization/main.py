import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # For Black-Scholes

# Binomial Option Pricing Functions
def fast_american_tree(S0, K, T, r, N, u, d, option_type):
    '''
    Write about the the formula and its components on the page
    '''
    dt = T / N
    q = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    S = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(N + 1))

    if option_type == 'C':
        C = np.maximum(S - K, 0)
    else:
        C = np.maximum(K - S, 0)

    for i in np.arange(N - 1, -1, -1):
        S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(i + 1))
        C[:i + 1] = discount * (q * C[1:i + 2] + (1 - q) * C[0:i + 1])
        C = C[:-1]
        if option_type == 'C':
            C = np.maximum(C, S - K)
        else:
            C = np.maximum(C, K - S)

    return C[0]

def binomial_option_pricing(S0, K, T, r, N, u, d, option_type):
    dt = T / N
    q = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    C = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(N + 1))

    if option_type == 'C':
        C = np.maximum(C - K, 0)
    else:
        C = np.maximum(K - C, 0)

    for i in np.arange(N, 0, -1):
        C = discount * (q * C[1:i + 1] + (1 - q) * C[0:i])

    return C[0]

# Black-Scholes Option Pricing Function
def black_scholes(r, S, K, T, sigma, option_type):
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'C':
        price = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
    elif option_type == 'P':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Streamlit App
st.title("Option Pricing Models")

# Select between Binomial or Black-Scholes
page = st.sidebar.selectbox("Choose a model", ["Binomial Option Pricing", "Black-Scholes Option Pricing"])

# Common parameters
S0 = st.sidebar.number_input("Initial Stock Price (S0)", min_value=50.0, max_value=150.0, value=100.0, step=1.0, format="%.2f")
K = st.sidebar.number_input("Strike Price (K)", min_value=50.0, max_value=150.0, value=100.0, step=1.0, format="%.2f")
T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.2f")
r = st.sidebar.number_input("Risk-Free Rate (r, in %)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, format="%.2f") / 100
option_type = st.sidebar.selectbox("Option Type", ["Call (C)", "Put (P)"])
option_type = 'C' if option_type == "Call (C)" else 'P'

if page == "Binomial Option Pricing":
    st.header("Binomial Option Pricing Model")
    
    # Binomial-specific parameters
    N = st.sidebar.number_input("Number of Time Steps (N)", min_value=1, max_value=100, value=3, step=1)
    u = st.sidebar.number_input("Up Factor (u)", min_value=1.01, max_value=2.0, value=1.1, step=0.01, format="%.2f")
    d = 1.0 / u  # Automatically calculate d from u
    option_style = st.sidebar.selectbox("Option Style", ["European", "American"])

    # Calculate option price based on selected style
    if option_style == "European":
        option_price = binomial_option_pricing(S0, K, T, r, N, u, d, option_type)
    else:
        option_price = fast_american_tree(S0, K, T, r, N, u, d, option_type)

    st.write(f"## Calculated Option Price: ${option_price:.2f}")

    # Plot the payoff diagram at maturity
    st.write("## Option Payoff Diagram")
    stock_prices = np.linspace(50, 150, 100)
    if option_type == 'C':  # Call option payoff
        payoffs = np.maximum(stock_prices - K, 0)
    else:  # Put option payoff
        payoffs = np.maximum(K - stock_prices, 0)

    # Plot the payoff diagram
    fig, ax = plt.subplots()
    ax.plot(payoffs, stock_prices, label=f'{option_type} Option Payoff')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.axhline(S0, color='red', linestyle='--', label='Initial Stock Price')
    ax.set_xlabel("Option Payoff")
    ax.set_ylabel("Stock Price at Maturity")
    ax.legend()
    ax.set_title(f"{option_type} Option Payoff at Maturity")
    st.pyplot(fig)

elif page == "Black-Scholes Option Pricing":
    st.header("Black-Scholes Option Pricing Model")
    
    # Additional parameter for Black-Scholes: Volatility
    sigma = st.sidebar.number_input("Volatility (sigma, in %)", min_value=1.0, max_value=100.0, value=20.0, step=0.5, format="%.2f") / 100
    
    # Calculate Black-Scholes option price
    option_price = black_scholes(r, S0, K, T, sigma, option_type)
    
    st.write(f"## Calculated Black-Scholes Option Price: ${option_price:.2f}")
    
    # Plot the payoff diagram at maturity
    st.write("## Option Payoff Diagram")
    stock_prices = np.linspace(50, 150, 100)
    if option_type == 'C':  # Call option payoff
        payoffs = np.maximum(stock_prices - K, 0)
    else:  # Put option payoff
        payoffs = np.maximum(K - stock_prices, 0)

    # Plot the payoff diagram
    fig, ax = plt.subplots()
    ax.plot(payoffs, stock_prices, label=f'{option_type} Option Payoff')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.axhline(S0, color='red', linestyle='--', label='Initial Stock Price')
    ax.set_xlabel("Option Payoff")
    ax.set_ylabel("Stock Price at Maturity")
    ax.legend()
    ax.set_title(f"{option_type} Option Payoff at Maturity")
    st.pyplot(fig)
