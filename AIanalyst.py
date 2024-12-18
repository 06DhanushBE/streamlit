import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import datetime

openai.api_key = 'sk-None-sIut5dnWgCjrTdQsEQltT3BlbkFJXSXYIaFfSgrGYrJ0eTga'

st.set_page_config(layout="wide")
st.title("AI-Powered Technical and Fundamental Stock Analysis Dashboard")
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., INFY.NS):", "INFY.NS")
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365) 

st.sidebar.date_input("Start Date", value=start_date.date(), max_value=end_date.date())
st.sidebar.date_input("End Date", value=end_date.date(), max_value=end_date.date())

if st.sidebar.button("Fetch Data"):
    st.session_state["stock_data"] = yf.download(ticker, start=start_date.date(), end=end_date.date())
    st.success("Stock data loaded successfully!")

if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]

    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"
        )
    ])

    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
        default=["20-Day SMA"]
    )

    def add_indicator(indicator):
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
        elif indicator == "VWAP":
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

    for indicator in indicators:
        add_indicator(indicator)

    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    st.subheader("Fundamental Analysis")
    stock = yf.Ticker(ticker)
    info = stock.info
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    quarterly_financials = stock.quarterly_financials

    fundamental_data = {
        "Market Cap": info.get('marketCap', 'N/A'),
        "Current Price": info.get('currentPrice', 'N/A'),
        "High / Low": f"{info.get('dayHigh', 'N/A')} / {info.get('dayLow', 'N/A')}",
        "P/E Ratio": info.get('trailingPE', 'N/A'),
        "Book Value": info.get('bookValue', 'N/A'),
        "Dividend Yield": f"{info.get('dividendYield', 0) * 100:.2f} %",
        "Shares Outstanding": info.get('sharesOutstanding', 'N/A')
    }

    st.write("### Financials")
    st.dataframe(financials)
    st.write("### Balance Sheet")
    st.dataframe(balance_sheet)
    st.write("### Quarterly Financials")
    st.dataframe(quarterly_financials)

    for key, value in fundamental_data.items():
        st.metric(label=key, value=value)

    st.subheader("AI-Powered Analysis")
    st.write("AI will analyze both **fundamental** and **technical** data to provide insights.")

    if st.button("Make Analysis"):
        context = f"""
        Stock Data for {ticker} from {start_date.date()} to {end_date.date()}:
        {data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).to_string()}

        Technical Indicators:
        {', '.join(indicators)}

        Fundamental Metrics:
        {', '.join([f"{key}: {value}" for key, value in fundamental_data.items()])}
        """

        template = """
        You are a helpful assistant. Answer the following question based on the context provided.
        Context: {context}
        Question: Provide insights based on the stock data, technical indicators, and fundamental data. in answer mention technical strength fundamental strength listout importent data 
        """

        prompt = PromptTemplate(input_variables=["context"], template=template)

        chat = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai.api_key)

        llm_chain = LLMChain(prompt=prompt, llm=chat)

        result = llm_chain.run(context=context)

        st.write("**AI Analysis Results:**")
        st.write(result)
