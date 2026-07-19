# pyright: reportMissingImports=false
try:
    import streamlit as st  # type: ignore[import]
except Exception:
    import sys
    print("Error: streamlit is not installed or could not be imported. Install it with: pip install streamlit")
    sys.exit(1)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
try:
    import matplotlib.pyplot as plt
except Exception:
    import sys
    print("Error: matplotlib is not installed or could not be imported. Install it with: pip install matplotlib")
    sys.exit(1)

st.set_page_config(page_title="Sales Predictor", layout="wide")
st.title("📈 Predictive Sales Analytics Dashboard")

# 1. Generate Dummy Data (Simulating a retail dataset)
@st.cache_data
def load_data():
    np.random.seed(42)
    months = pd.date_range(start='2022-01-01', end='2025-12-01', freq='MS')
    sales = 10000 + (np.arange(len(months)) * 150) + np.random.randint(-500, 500, size=len(months))
    marketing_spend = sales * 0.15 + np.random.randint(-200, 200, size=len(months))
    df = pd.DataFrame({'Month': months, 'Sales': sales, 'Marketing_Spend': marketing_spend})
    return df

df = load_data()

# Sidebar for filters
st.sidebar.header("Filter Options")
year_filter = st.sidebar.multiselect("Select Year:", df['Month'].dt.year.unique(), default=df['Month'].dt.year.unique())
filtered_df = df[df['Month'].dt.year.isin(year_filter)]

# 2. Display Data
col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw Sales Data")
    st.dataframe(filtered_df.head(10))

with col2:
    st.subheader("Sales Trend")
    fig, ax = plt.subplots()
    ax.plot(filtered_df['Month'], filtered_df['Sales'], marker='o', linestyle='-', color='b')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 3. Machine Learning Prediction
st.header("🔮 Future Sales Prediction")
months_to_predict = st.slider("How many months to predict?", 1, 12, 3)

if st.button("Run Prediction Model"):
    # Prepare data for ML
    X = np.arange(len(df), dtype=float).reshape(-1, 1)
    y = df['Sales'].to_numpy(dtype=float)
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future
    future_X = np.arange(len(df), len(df) + months_to_predict).reshape(-1, 1)
    future_sales = model.predict(future_X)
    
    # Plot predictions
    fig2, ax2 = plt.subplots()
    ax2.plot(df['Month'], y, label='Historical Sales', color='blue')
    
    future_dates = pd.date_range(start=df['Month'].iloc[-1] + pd.DateOffset(months=1), periods=months_to_predict, freq='MS')
    ax2.plot(future_dates, future_sales, label='Predicted Sales', color='red', marker='x')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    
    st.success(f"Prediction complete! Expected sales for next month: ${future_sales[0]:,.2f}")