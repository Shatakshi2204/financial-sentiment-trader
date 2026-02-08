import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# For Streamlit Cloud deployment - override with secrets if available
try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    os.environ["NEWS_API_KEY"] = NEWS_API_KEY
except:
    pass  # Use .env file instead (local development)

# Page config
st.set_page_config(
    page_title="AI Trading Signal System",
    page_icon="📈",
    layout="wide"
)

# Load FinBERT model (cached)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# Your functions from Jupyter
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_score = predictions.detach().numpy()[0]
    sentiment_labels = ['positive', 'negative', 'neutral']
    sentiment = sentiment_labels[sentiment_score.argmax()]
    confidence = sentiment_score.max()
    return sentiment, confidence

def fetch_financial_news(company_name, days_back=7):
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': company_name,
        'from': from_date,
        'to': to_date,
        'language': 'en',
        'sortBy': 'publishedAt',
        'apiKey': NEWS_API_KEY
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        articles = response.json()['articles']
        news_data = []
        for article in articles:
            news_data.append({
                'title': article['title'],
                'description': article['description'],
                'published_at': article['publishedAt'],
                'source': article['source']['name'],
                'url': article['url']
            })
        return pd.DataFrame(news_data)
    return None

def get_news_with_sentiment(company_name, days_back=7):
    news_df = fetch_financial_news(company_name, days_back)
    
    if news_df is None or len(news_df) == 0:
        return None
    
    sentiments = []
    confidences = []
    
    for idx, row in news_df.iterrows():
        text = f"{row['title']}. {row['description']}"
        try:
            sentiment, confidence = analyze_sentiment(text)
            sentiments.append(sentiment)
            confidences.append(confidence)
        except:
            sentiments.append('neutral')
            confidences.append(0.0)
    
    news_df['sentiment'] = sentiments
    news_df['confidence'] = confidences
    
    return news_df

def get_stock_data(ticker, days=30):
    stock = yf.Ticker(ticker)
    df = stock.history(period=f"{days}d")
    
    if df.empty:
        return None
    
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    return df

def generate_trading_signal(ticker, company_name, news_days=3):
    news_df = get_news_with_sentiment(company_name, days_back=news_days)
    
    if news_df is None:
        return None
    
    sentiment_score = (
        news_df['sentiment'].value_counts().get('positive', 0) - 
        news_df['sentiment'].value_counts().get('negative', 0)
    ) / len(news_df)
    
    stock_df = get_stock_data(ticker, days=30)
    
    if stock_df is None:
        return None
    
    current_price = stock_df['Close'][-1]
    sma_5 = stock_df['SMA_5'][-1]
    sma_20 = stock_df['SMA_20'][-1]
    
    technical_signal = "BULLISH" if sma_5 > sma_20 else "BEARISH"
    
    if sentiment_score > 0.1 and technical_signal == "BULLISH":
        signal = "🟢 STRONG BUY"
        reason = "Positive sentiment + bullish technicals"
    elif sentiment_score > 0.1:
        signal = "🟡 BUY"
        reason = "Positive sentiment, but mixed technicals"
    elif sentiment_score < -0.1 and technical_signal == "BEARISH":
        signal = "🔴 STRONG SELL"
        reason = "Negative sentiment + bearish technicals"
    elif sentiment_score < -0.1:
        signal = "🟠 SELL"
        reason = "Negative sentiment, but mixed technicals"
    else:
        signal = "⚪ HOLD"
        reason = "Neutral sentiment and technicals"
    
    return {
        'ticker': ticker,
        'signal': signal,
        'sentiment_score': sentiment_score,
        'technical_signal': technical_signal,
        'current_price': current_price,
        'sma_5': sma_5,
        'sma_20': sma_20,
        'news_df': news_df,
        'stock_df': stock_df,
        'reason': reason
    }

# Streamlit UI
st.title("📈 AI-Powered Financial Trading Signal System")
st.markdown("### Real-time News Sentiment + Technical Analysis")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
company_name = st.sidebar.text_input("Company Name", "Apple")
news_days = st.sidebar.slider("News lookback (days)", 1, 14, 3)

if st.sidebar.button("Generate Signal", type="primary"):
    with st.spinner("Analyzing news and stock data..."):
        result = generate_trading_signal(ticker, company_name, news_days)
        
        if result:
            # Display signal
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Trading Signal", result['signal'])
                st.caption(result['reason'])
            
            with col2:
                st.metric("Current Price", f"${result['current_price']:.2f}")
                st.caption(f"Technical: {result['technical_signal']}")
            
            with col3:
                st.metric("Sentiment Score", f"{result['sentiment_score']:.3f}")
                sentiment_counts = result['news_df']['sentiment'].value_counts()
                st.caption(f"✅ {sentiment_counts.get('positive', 0)} | ❌ {sentiment_counts.get('negative', 0)} | ⚪ {sentiment_counts.get('neutral', 0)}")
            
            # Price chart
            st.subheader("📊 Price Chart with Moving Averages")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=result['stock_df'].index,
                y=result['stock_df']['Close'],
                name='Price',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=result['stock_df'].index,
                y=result['stock_df']['SMA_5'],
                name='5-day MA',
                line=dict(color='orange', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=result['stock_df'].index,
                y=result['stock_df']['SMA_20'],
                name='20-day MA',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # News sentiment table
            st.subheader("📰 Recent News with Sentiment")
            display_df = result['news_df'][['title', 'sentiment', 'confidence', 'published_at']].head(10)
            st.dataframe(display_df, use_container_width=True)
        
        else:
            st.error("Could not fetch data. Please check ticker symbol.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with:**")
st.sidebar.markdown("🤖 FinBERT AI Model")
st.sidebar.markdown("📰 NewsAPI")
st.sidebar.markdown("📊 Yahoo Finance")