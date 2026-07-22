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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import io

st.set_page_config(page_title="Sales Predictor", layout="wide")
st.title("📈 Predictive Sales Analytics Dashboard")

# Sales-related keywords for document classification
SALES_KEYWORDS = [
    'sales', 'revenue', 'income', 'profit', 'earnings', 'quarterly', 'annual',
    'forecast', 'projection', 'trend', 'growth', 'retail', 'transaction',
    'customer', 'order', 'purchase', 'sold', 'units', 'margin', 'expense',
    'financial', 'budget', 'target', 'goal', 'performance', 'metric', 'kpi',
    'month', 'year', 'q1', 'q2', 'q3', 'q4', 'january', 'february', 'march',
    'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'
]

# Train a simple classifier for sales document detection
@st.cache_resource
def train_sales_classifier():
    """Train a simple Naive Bayes classifier to detect sales-related documents."""
    # Training data - positive examples (sales-related)
    sales_docs = [
        "Monthly sales report showing revenue growth and profit margins",
        "Quarterly earnings forecast with customer acquisition metrics",
        "Annual retail performance analysis with transaction data",
        "Sales trend analysis showing units sold and income projections",
        "Revenue budget report with financial targets and goals",
        "Customer purchase history and order transaction summary",
        "Profit margin analysis with expense tracking and KPI metrics",
        "Q1 Q2 Q3 Q4 sales performance with growth projections"
    ]
    
    # Training data - negative examples (non-sales)
    non_sales_docs = [
        "Employee handbook with HR policies and procedures",
        "Marketing campaign creative brief for brand awareness",
        "Software development sprint planning and code review",
        "Legal contract terms and conditions agreement",
        "Recipe instructions for cooking and food preparation",
        "Travel itinerary with flight bookings and hotel reservations",
        "Medical diagnosis report with treatment recommendations",
        "Academic research paper on biology and chemistry experiments"
    ]
    
    # Combine training data
    all_docs = sales_docs + non_sales_docs
    labels = [1] * len(sales_docs) + [0] * len(non_sales_docs)  # 1 = sales, 0 = non-sales
    
    # Train TF-IDF + Naive Bayes classifier
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(all_docs)
    
    classifier = MultinomialNB()
    classifier.fit(X, labels)
    
    return vectorizer, classifier

def extract_text_from_file(uploaded_file):
    """Extract text content from uploaded file (supports .txt, .csv, .md)."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'txt':
            return uploaded_file.read().decode('utf-8')
        elif file_extension == 'csv':
            # Read CSV and convert to text representation
            df_temp = pd.read_csv(uploaded_file)
            return df_temp.to_string()
        elif file_extension in ['md', 'markdown']:
            return uploaded_file.read().decode('utf-8')
        else:
            # Try to read as text anyway
            return uploaded_file.read().decode('utf-8')
    except Exception as e:
        return None

def is_sales_document(text):
    """Check if the document is sales-related using ML classifier and keyword matching."""
    if not text or len(text.strip()) < 10:
        return False, 0.0, "Document is too short or empty"
    
    vectorizer, classifier = train_sales_classifier()
    
    # Get ML prediction
    X_test = vectorizer.transform([text])
    ml_prediction = classifier.predict(X_test)[0]
    ml_probability = classifier.predict_proba(X_test)[0][1]  # Probability of being sales-related
    
    # Keyword-based scoring
    text_lower = text.lower()
    keyword_count = sum(1 for keyword in SALES_KEYWORDS if keyword in text_lower)
    keyword_score = min(keyword_count / 10, 1.0)  # Normalize to 0-1
    
    # Combined score (weighted average)
    combined_score = 0.6 * ml_probability + 0.4 * keyword_score
    
    # Determine if sales-related (threshold: 0.5)
    is_sales = combined_score >= 0.5
    
    if not is_sales:
        reason = f"Low relevance score ({combined_score:.2f}). Found {keyword_count} sales-related keywords."
    else:
        reason = f"High relevance score ({combined_score:.2f}). Found {keyword_count} sales-related keywords."
    
    return is_sales, combined_score, reason

def parse_sales_data(text):
    """Parse sales data from text content. Returns DataFrame with Month and Sales columns."""
    # Try to find patterns like: date/month, sales/revenue numbers
    patterns = [
        # Pattern: Month Year: $Number or Month Year, Number
        r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})[\s:,$]+([\d,]+(?:\.\d+)?)',
        # Pattern: YYYY-MM, Sales Number
        r'(\d{4}-\d{2})[\s:,$]+([\d,]+(?:\.\d+)?)',
        # Pattern: MM/DD/YYYY, Sales Number
        r'(\d{1,2}/\d{1,2}/\d{4})[\s:,$]+([\d,]+(?:\.\d+)?)',
    ]
    
    data = []
    for pattern in patterns:
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    try:
                        date_str, sales_str = match
                        sales_value = float(sales_str.replace(',', ''))
                        
                        # Try to parse date
                        for date_format in ['%B %Y', '%b %Y', '%Y-%m', '%m/%d/%Y']:
                            try:
                                parsed_date = pd.to_datetime(date_str, format=date_format)
                                data.append({'Month': parsed_date, 'Sales': sales_value})
                                break
                            except:
                                continue
                    except:
                        continue
                if data:
                    break
        except:
            continue
    
    if data:
        df = pd.DataFrame(data)
        df = df.sort_values('Month').drop_duplicates(subset='Month', keep='last')
        return df
    
    return None

# 1. Generate Dummy Data (Simulating a retail dataset) - fallback option
@st.cache_data
def load_dummy_data():
    np.random.seed(42)
    months = pd.date_range(start='2022-01-01', end='2025-12-01', freq='MS')
    sales = 10000 + (np.arange(len(months)) * 150) + np.random.randint(-500, 500, size=len(months))
    marketing_spend = sales * 0.15 + np.random.randint(-200, 200, size=len(months))
    df = pd.DataFrame({'Month': months, 'Sales': sales, 'Marketing_Spend': marketing_spend})
    return df

# File upload section
st.sidebar.header("📁 Upload Sales Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload a sales report (.txt, .csv, .md)",
    type=['txt', 'csv', 'md'],
    help="Upload a document containing sales data. The system will automatically detect if it's sales-related."
)

df = None
data_source = "dummy"

if uploaded_file is not None:
    # Extract text from file
    text_content = extract_text_from_file(uploaded_file)
    
    if text_content:
        # Check if document is sales-related
        is_sales, confidence, reason = is_sales_document(text_content)
        
        if is_sales:
            st.sidebar.success(f"✅ Document validated as sales-related")
            st.sidebar.info(f"Confidence: {confidence:.2%}\n{reason}")
            
            # Try to parse sales data from document
            parsed_df = parse_sales_data(text_content)
            
            if parsed_df is not None and len(parsed_df) > 0:
                df = parsed_df
                data_source = "uploaded"
                st.sidebar.success(f"Successfully extracted {len(df)} data points from document")
            else:
                st.sidebar.warning("Could not extract structured sales data from document. Using sample data.")
                df = load_dummy_data()
        else:
            st.sidebar.error(f"⚠️ Warning: Document appears unrelated to sales")
            st.sidebar.error(f"Confidence: {confidence:.2%}\n{reason}")
            st.error("""
                ### 🚫 Invalid Document Type
                
                The uploaded document does not appear to be related to sales analytics.
                
                **Expected content:** Sales reports, revenue data, financial forecasts, 
                retail performance metrics, quarterly earnings, etc.
                
                Please upload a relevant sales document or use the default sample data.
                """)
            # Still load dummy data so the app remains functional
            df = load_dummy_data()
    else:
        st.sidebar.error("Could not read the uploaded file. Please ensure it's a valid text file.")
        df = load_dummy_data()
else:
    # No file uploaded, use dummy data
    df = load_dummy_data()

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