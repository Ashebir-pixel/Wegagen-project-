import streamlit as st
import pandas as pd
import os

st.title("🇪🇹 Wegagen Investment & Agriculture Tracker")

# 1. Setup the Storage File
DATA_FILE = "my_investment_data.csv"

# Create the file if it doesn't exist
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=["Date", "Asset", "Price_ETB", "My_Balance"])
    df.to_csv(DATA_FILE, index=False)

# 2. Sidebar for Manual Entry
st.sidebar.header("Update Data")
asset_name = st.sidebar.selectbox("Select Asset", ["Wegagen Stock", "Meki Teff", "Hotel Service Fee"])
current_price = st.sidebar.number_input("Enter Current Price (ETB)", min_value=0.0)
current_balance = st.sidebar.number_input("Your Current Account Balance", value=3400.0)

if st.sidebar.button("Save Entry"):
    new_data = pd.DataFrame([[pd.Timestamp.now(), asset_name, current_price, current_balance]], 
                            columns=["Date", "Asset", "Price_ETB", "My_Balance"])
    new_data.to_csv(DATA_FILE, mode='a', header=False, index=False)
    st.sidebar.success("Data saved locally!")

# 3. Main Dashboard Display
st.subheader("Your Investment History")
history_df = pd.read_csv(DATA_FILE)

if not history_df.empty:
    st.dataframe(history_df) # Shows your data like an Excel sheet
    
    # Filter for Chart
    asset_filter = st.selectbox("View Trend For:", history_df["Asset"].unique())
    filtered_df = history_df[history_df["Asset"] == asset_filter]
    st.line_chart(filtered_df.set_index("Date")["Price_ETB"])
else:
    st.info("No data saved yet. Enter a price in the sidebar to begin.")