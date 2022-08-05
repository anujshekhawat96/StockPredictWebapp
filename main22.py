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

TODAY= '2018-12-30'


st.title('Company Stock Price Prediction & Financials Info App')

stocks = st.text_input('Enter Stock Ticker', 'AAPL')

#stocks = ('AAPL','GOOG','MSFT')
selected_stocks = stocks 

n_days = st.number_input('Enter Days of Predictions',30)

#n_years = st.slider("Years of prediction:",1,4)
period = 1 * n_days

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load data...")

data = load_data(selected_stocks)
data_load_state.text("Data Loading is Done...Thanks for your patience!")

##DEscribing data 
st.subheader('Data from 2016 to 2018')

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
    
st.write(f'Forecast plot for {n_days} Days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)







# ##performance matrix 

# from prophet.diagnostics import performance_metrics
# metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
# metric_df.tail()
# metric_df.dropna(inplace=True)
# r2 = r2_score(metric_df.y, metric_df.yhat)
# mse = mean_squared_error(metric_df.y, metric_df.yhat)
# mae =mean_absolute_error(metric_df.y, metric_df.yhat)
# df_p = performance_metrics(df_cv)
# df_p.head()


tail_data =forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

st.write(tail_data)

# st.write(r2)
# st.write(mse)



###MERGING OF FINANCE APP TOO 

##TICKER VALUE
ticker_pass = yf.Ticker(selected_stocks)





##SHOW MAJOR HOLDERS

major_hold = ticker_pass.major_holders
st.subheader('Stakeholders Distribution')
st.write(major_hold)

csv = major_hold.to_csv(index=False)

st.download_button('Download Stakeholders Data', csv, file_name='StakeholderDistributionData.csv')


#major_hold.to_excel('major_holdings.xlsx', sheet_name='majorholdings', index=False)

# show institutional holders
inst_hold = ticker_pass.institutional_holders
st.subheader('Institutional Stakeholders')
st.write(inst_hold)

csv1 = inst_hold.to_csv(index=False)

st.download_button('Download Institutional Stakeholders Data', csv1, file_name='InstitutionalStakeholderData.csv')

##SHow financials 
quaterly_fin = ticker_pass.quarterly_financials
st.subheader('Quaterly Financials')
st.write(quaterly_fin)

csv2 = quaterly_fin.to_csv(index=False)

st.download_button('Download Quaterly Financials Data', csv2, file_name='QuaterlyFinancialData.csv')



# show balance sheet

st.subheader('Quaterly Balance Sheet')
balancesheet_quater = ticker_pass.quarterly_balance_sheet

st.write(balancesheet_quater)

csv3 = balancesheet_quater.to_csv(index=False)

st.download_button('Download Quaterly Balance Sheet Data', csv3, file_name='QuaterlyBalanceSheetData.csv')

# show cashflow

st.subheader('Quaterly CashFlow')
quatercash = ticker_pass.quarterly_cashflow
st.write(quatercash)

csv4 = quatercash.to_csv(index=False)

st.download_button('Download Quaterly Cashflow', csv4, file_name='QuaterlyCashFlow.csv')



# show earnings
st.subheader('Quaterly Earnings')
earn = ticker_pass.quarterly_earnings
st.write(earn)

csv5 = earn.to_csv(index=False)

st.download_button('Download Quaterly Earning Data', csv5, file_name='QuaterlyEarning.csv')


# show analysts recommendations
st.subheader('Analyst Recommendations')
recom = ticker_pass.recommendations
st.write(recom)

csv6 = recom.to_csv(index=False)

st.download_button('Download Analyst Recommendation Data', csv6, file_name='AnalystRecommendation.csv')


# show dividend nd stock split
st.subheader('Dividends and Stock Splits')
divi = ticker_pass.actions
st.write(divi)

csv7 = divi.to_csv(index=False)

st.download_button('Download Dividend and Stock Splits Data', csv7, file_name='StockSplit.csv')


##Non Fin Info

emp = ticker_pass.info['fullTimeEmployees']
st.subheader('Company Non Financial Info')


sector = ticker_pass.info['sector']
addrs = ticker_pass.info['address1']
hqcity = ticker_pass.info['city']
hqcountry = ticker_pass.info['country']


web = ticker_pass.info['website']



st.write('Full Time Employees: ', emp)
st.write('Sector: ', sector)
st.write('Address: ', addrs)
st.write('City: ', hqcity)
st.write('Country: ', hqcountry)

st.write('Website: ', web)


