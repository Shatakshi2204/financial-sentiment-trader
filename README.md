# 📈 AI-Powered Financial Trading Signal System

> Real-time sentiment analysis of financial news combined with technical indicators to generate actionable trading signals

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]https://financial-sentiment-trader-ritgenosattaupa4nwecc4.streamlit.app/

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This system combines **Natural Language Processing (NLP)** with **technical stock analysis** to generate data-driven trading signals. It scrapes real-time financial news, analyzes sentiment using FinBERT (a BERT model fine-tuned for financial text), and combines this with moving average crossover strategies to produce buy/sell/hold recommendations.

### **Key Features:**
- 📰 Real-time news scraping from multiple financial sources
- 🤖 AI-powered sentiment analysis using FinBERT
- 📊 Technical analysis with moving averages (SMA 5 & 20)
- 🎨 Interactive web dashboard built with Streamlit
- 📈 Historical backtesting capabilities
- ⚡ Live trading signals updated daily

## 🚀 Live Demo

**Try it here:** https://financial-sentiment-trader-ritgenosattaupa4nwecc4.streamlit.app/

<img width="1568" height="538" alt="image" src="https://github.com/user-attachments/assets/a9f4b11e-a26d-4cf9-a1b4-b22c1780cdea" />
<img width="1568" height="575" alt="image" src="https://github.com/user-attachments/assets/aed85507-819f-4625-a17d-994b2c0720e8" />
<img width="1568" height="569" alt="image" src="https://github.com/user-attachments/assets/b4b37572-6a11-45c9-b0b6-ce5b73ec20f0" />



## 🛠️ Tech Stack

- **Machine Learning:** FinBERT (Transformers), PyTorch
- **Data Sources:** NewsAPI, Yahoo Finance API
- **Backend:** Python, Pandas, NumPy
- **Frontend:** Streamlit, Plotly
- **Deployment:** Streamlit Cloud

## 📊 How It Works

1. **News Aggregation:** Fetches recent financial news articles using NewsAPI
2. **Sentiment Analysis:** FinBERT analyzes each article's sentiment (positive/negative/neutral)
3. **Technical Analysis:** Calculates moving averages and identifies trends
4. **Signal Generation:** Combines sentiment score + technical indicators
   - 🟢 **STRONG BUY:** Positive sentiment + bullish technicals
   - 🟡 **BUY:** Positive sentiment, mixed technicals
   - ⚪ **HOLD:** Neutral sentiment/technicals
   - 🟠 **SELL:** Negative sentiment, mixed technicals
   - 🔴 **STRONG SELL:** Negative sentiment + bearish technicals

## 🏃 Quick Start

### Prerequisites
- Python 3.8+
- API Keys (free):
  - [NewsAPI](https://newsapi.org/) - for financial news
  - [Alpha Vantage](https://www.alphavantage.co/) - for stock data (backup)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/financial-sentiment-trader.git
cd financial-sentiment-trader
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file with your API keys:
```
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

5. Run the app:
```bash
streamlit run app.py
```

## 📈 Performance

Backtesting results (Jan 2025 - Feb 2026):
- **Strategy Return:** Varies by stock and time period
- **Benchmark (S&P 500):** +X.XX%
- **Win Rate:** XX%

*Note: Past performance does not guarantee future results. This is an educational project.*

## 🎓 What I Learned

- Implementing transformer models (FinBERT) for domain-specific NLP
- Integrating multiple data sources (news APIs, stock APIs)
- Building production-ready ML pipelines
- Creating interactive data visualizations
- Deploying ML applications to the cloud

## 📝 Future Enhancements

- [ ] Add more technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Implement deep learning price prediction
- [ ] Add portfolio optimization features
- [ ] Include macroeconomic indicators (FRED data)
- [ ] Real-time websocket updates
- [ ] Email/SMS alert system

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not financial advice. Always do your own research and consult with a financial advisor before making investment decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author
Shatakshi Guha


## 🙏 Acknowledgments

- [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) for the pre-trained sentiment model
- [NewsAPI](https://newsapi.org/) for news data
- [Yahoo Finance](https://finance.yahoo.com/) for stock data

---

⭐ Star this repo if you found it helpful!
