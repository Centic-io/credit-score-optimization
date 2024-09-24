import streamlit as st
import json
import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(__file__)
other_dir = os.path.join(current_dir, '..', 'new-model')
other_dir = os.path.abspath(other_dir)

parameter_path = os.path.join(other_dir, 'results', 'GA.json')
data_path = os.path.join(other_dir, 'data', 'Lending-Data-Ethereum_Normalized.csv')


with open(parameter_path, "r") as f:
    parameter = np.array(json.load(f))

df = pd.read_csv(data_path)

st.title("Wallet Credit Score")
chain = st.selectbox("Chain", ["Ethereum", "Binance Smart Chain"])
address = st.text_input("Address", "")

# Button to get wallets with the same owners
if st.button("Get Credit Score"):
    if address:
        # st.write(f"Caculating credit score for {address} on {chain}...")
        # Add your logic to fetch wallets here
        result = df[df["address"] == address]
        if len(result) > 0:
            values = np.array(result.drop(columns=['address']).values.flatten())
            score = int(np.dot(parameter, values))
            st.write(f"Credit score for **{address}** is **{score}**")
        else:
            st.write(f"No information about this wallet.")
    else:
        st.write("Please enter a valid address.")

# Footer
st.markdown("""
    <footer style='text-align: center;'>
        <hr>
        <p>Copyright © My Website 2024.</p>
    </footer>
""", unsafe_allow_html=True)
