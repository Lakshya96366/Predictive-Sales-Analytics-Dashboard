
---

### 📄 Predictive Sales Analytics Dashboard

```markdown
# 📈 Predictive Sales Analytics Dashboard

An interactive, data-driven web application that forecasts future revenue trends and visualizes historical retail data. Built with **Streamlit** and **Scikit-learn**, this dashboard transforms raw time-series data into actionable business insights through machine learning.

## 🚀 Features
- **Interactive UI**: Clean, responsive dashboard with dynamic sidebar filtering (e.g., filter by year).
- **Machine Learning Forecasting**: Utilizes **Scikit-learn’s Linear Regression** to analyze historical time-series data and predict future sales trajectories based on user-defined timeframes.
- **Performance Optimization**: Implements Streamlit’s `@st.cache_data` decorator to ensure efficient, instantaneous data loading and prevent redundant computations.
- **Rich Visualizations**: Contrasts historical vs. projected sales using **Matplotlib**, making complex data easily digestible for stakeholders.

## 🛠️ Tech Stack
- **Language**: Python
- **Frontend/Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Linear Regression)
- **Visualization**: Matplotlib

## 🧠 How the ML Pipeline Works
1. **Data Generation/Loading**: Simulates a realistic retail dataset with monthly sales and marketing spend.
2. **Preprocessing**: Converts dates to numerical indices (`X`) and extracts sales figures (`y`) for model training.
3. **Training**: Fits a `LinearRegression` model to identify the underlying trend in the historical data.
4. **Prediction**: Extrapolates the learned trend into the future based on the user's selected "months to predict" slider.
5. **Visualization**: Plots historical data (blue) and predicted data (red) on a unified timeline for clear comparison.

## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Lakshya96366/Predictive-Sales-Analytics-Dashboard.git
   cd Predictive-Sales-Analytics-Dashboard
