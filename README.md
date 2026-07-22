---

### 📄 Predictive Sales Analytics Dashboard

```markdown
# 📈 Predictive Sales Analytics Dashboard

An interactive, data-driven web application that forecasts future revenue trends and visualizes historical retail data. Built with **Streamlit** and **Scikit-learn**, this dashboard transforms raw time-series data into actionable business insights through machine learning.

## 🚀 Features
- **Document Upload & Validation**: Upload sales documents (.txt, .csv, .md) with automatic AI-powered validation to ensure relevance.
- **Smart Document Classification**: Uses **Naive Bayes ML classifier** combined with keyword analysis to detect if uploaded documents are sales-related.
- **Automatic Data Extraction**: Parses sales data from uploaded documents using pattern recognition for dates and revenue figures.
- **Interactive UI**: Clean, responsive dashboard with dynamic sidebar filtering (e.g., filter by year).
- **Machine Learning Forecasting**: Utilizes **Scikit-learn's Linear Regression** to analyze historical time-series data and predict future sales trajectories based on user-defined timeframes.
- **Performance Optimization**: Implements Streamlit's `@st.cache_data` decorator to ensure efficient, instantaneous data loading and prevent redundant computations.
- **Rich Visualizations**: Contrasts historical vs. projected sales using **Matplotlib**, making complex data easily digestible for stakeholders.

## 🛠️ Tech Stack
- **Language**: Python
- **Frontend/Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Linear Regression, Naive Bayes, TF-IDF Vectorization)
- **Visualization**: Matplotlib

## 🤖 AI Document Classification System
The app includes an intelligent document validation system that:
1. **Trains a Naive Bayes classifier** on sales-related vs non-sales documents
2. **Extracts text** from uploaded files (.txt, .csv, .md)
3. **Analyzes content** using both ML prediction and keyword matching
4. **Provides confidence scores** and detailed feedback on document relevance
5. **Warns users** if uploaded documents are unrelated to sales analytics

### Supported Document Formats
- **Sales reports** with monthly/quarterly revenue data
- **Financial forecasts** with projected earnings
- **Retail performance metrics** with transaction data
- **CSV files** with date and sales columns

## 🧠 How the ML Pipeline Works
1. **Document Upload**: User uploads a sales document through the sidebar.
2. **Content Extraction**: Text is extracted from the file based on its format.
3. **Relevance Detection**: 
   - TF-IDF vectorization converts text to numerical features
   - Naive Bayes classifier predicts sales relevance
   - Keyword matching provides additional scoring
   - Combined score determines if document is sales-related
4. **Data Parsing**: If sales-related, regex patterns extract date-value pairs.
5. **Model Training**: Fits a `LinearRegression` model to the extracted or sample data.
6. **Prediction**: Extrapolates the learned trend into the future.
7. **Visualization**: Plots historical data (blue) and predicted data (red) on a unified timeline.

## 📋 Usage Instructions

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Uploading Sales Documents
1. Click the **"Upload Sales Document"** button in the sidebar
2. Select a `.txt`, `.csv`, or `.md` file containing sales data
3. The system will automatically:
   - Validate if the document is sales-related
   - Display a confidence score
   - Extract sales data if possible
   - Show warnings for non-sales documents

### Example Sales Document Format
```
Monthly Sales Report - Q1 2024

January 2024: $15,000
February 2024: $18,500
March 2024: $22,000

Revenue Growth: 15%
Customer Orders: 450
```

## 🔒 Error Handling
- **Non-sales documents**: Displays warning with confidence score
- **Unreadable files**: Shows error message and uses sample data
- **Missing data**: Falls back to generated dummy data
- **Parsing failures**: Continues with sample data while notifying user
```
