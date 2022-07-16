import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array
from sklearn.preprocessing import StandardScaler
import pandas_datareader as data 



import streamlit as st

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

#import Prophet as ph
#from fbprophet.plot import plot_plotly
from datetime import date 

from plotly import graph_objs as go


START = '2016-01-01'

TODAY= date.today().strftime("%Y-%m-%d")


st.title('Stock Prediction App')

stocks = st.text_input('Enter Stock Ticker', 'AAPL')

#stocks = ('AAPL','GOOG','MSFT')
selected_stocks = stocks 

n_years = st.number_input('Enter Days of Predictions',30)

#n_years = st.slider("Years of prediction:",1,4)
period = 1 * n_years

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load data...")

data = load_data(selected_stocks)
data_load_state.text("Data Loading is Done...Thanks of your patience!")

##DEscribing data 
st.subheader('Data from 2016 to Today')

st.write(data.describe())


# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

##model train
m = Prophet()
m.fit(df_train)

##futute
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)


# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.describe())
    
st.write(f'Forecast plot for {n_years} Days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


##performance matrix 

from prophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.head()