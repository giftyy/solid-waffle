import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import matplotlib.dates as mdates
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time 
import math
import seaborn as sns

option = st.sidebar.selectbox('Select one option', ( 'Select','Analyze', 'Predict'))

uploaded_file = st.file_uploader('Choose a file',type=["csv","xls"])
df1=tempfile.NamedTemporaryFile(delete=False)
if uploaded_file is not None:
	df1=pd.read_csv(uploaded_file)

if option =='Analyze':
	def head():
		h = st.write(df1.head())
		t = st.write(df1.tail())
		return h,t
	def plots():
		df1["Date"]=pd.to_datetime(df1.Date,format="%Y-%m-%d")
		df1.index=df1['Date']
		fig=plt.figure(figsize=(16,8))
		plt.title('Close Price History')
		plt.xlabel('Date', fontsize=18)
		plt.ylabel('Close Price USD ($)', fontsize=18)
		plt.plot(df1["Close"],label='Close Price history')
		st.pyplot(fig)
		#st.line_chart(df1,columns=['Date','Adj Close'])#,columns=['adj close','Date'])
	#def histogram():
	#	st.subheader('adj Clcose')
	#	st.write(df1)
	def sub():
		fig, ax=plt.subplots(figsize=(5,5),sharex=True)
		fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
		fig.gca().xaxis.set_major_locator(mdates.YearLocator())
		ax.plot(df1.index, df1['Adj Close'],color='r')
		ax.grid(True)
		ax.tick_params(labelrotation=90)
		ax.set_title('IBM subplot');
		st.pyplot(fig)
	def vol():
		df1["Date"]=pd.to_datetime(df1.Date,format="%Y-%m-%d")
		df1.index=df1["Date"]
		fig=plt.figure(figsize=(16,8))
		plt.title("volume of trades")
		plt.xlabel("Date",fontsize=18)
		plt.ylabel("volume",fontsize=18)
		plt.plot(df1["Volume"],label='volume')
		st.pyplot(fig)
#Bar Chart
	#	st.bar_chart(df1['Adj Close'])

	def bins():
		#Daily Percentage
		daily_close= df1[['Adj Close']]
		# Daily returns
		daily_pct_change= daily_close.pct_change()
		# Replace NA values with 0
		daily_pct_change.fillna(0, inplace=True)
		st.write(daily_pct_change.head())
		daily_pct_change.hist(bins=50)
		# Show the plot
		plt.show()
		st.pyplot()
	#def volatility():
		#daily_pct_change=bins()
		#Volatility
		st.write("Volatility")
		min_periods = 75 
		# Calculate the volatility
		vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods) 
		vol.fillna(0,inplace=True)
		st.write(vol.tail())
		# Plot volatility
		vol.plot(figsize=(10, 8))
		plt.show()
		st.pyplot()

	def trends():
		#Rolling Means (Trends and Seasonality)
		adj_close_px = df1['Adj Close']
		# Short moving window rolling mean
		df1['42'] = adj_close_px.rolling(window=40).mean()

		# Long moving window rolling mean
		df1['252'] = adj_close_px.rolling(window=252).mean()

		# Plot the adjusted closing price, the short and long windows of rolling means
		df1[['Adj Close', '42', '252']].plot(title="IBM")

		# Showing plot
		plt.show()
		st.pyplot()



	def main():
		head()
		plots()
		sub()
		vol()
		bins()
		#volatility()
		trends()

		st.write("End of analysis")
	main()
if option =='Predict':
	def new():
		data=df1.sort_index(ascending=True,axis=0)
		new_dataset=pd.DataFrame(index=range(0,len(df1)),columns=['Date','Close'])

		for i in range(0,len(data)):
			new_dataset["Date"][i]=data['Date'][i]
			new_dataset["Close"][i]=data["Close"][i]

	#def scale():
	#	new_dataset = new()
		scaler=MinMaxScaler(feature_range=(0,1))
		new_dataset.index=new_dataset.Date
		new_dataset.drop("Date",axis=1,inplace=True)
		final_dataset=new_dataset.values

		train_data=final_dataset[0:987,:]
		valid_data=final_dataset[987:,:]

		
		scaler=MinMaxScaler(feature_range=(0,1))
		scaled_data=scaler.fit_transform(final_dataset)
		x_train_data,y_train_data=[],[]
		for i in range(60,len(train_data)):
			x_train_data.append(scaled_data[i-60:i,0])
			y_train_data.append(scaled_data[i,0])    
		x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
		x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

	#def mode():
	#	new_dataset=new()
		lstm_model=Sequential()
		lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
		lstm_model.add(LSTM(units=50))
		lstm_model.add(Dense(1))

		inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
		inputs_data=inputs_data.reshape(-1,1)
		inputs_data=scaler.transform(inputs_data)

		lstm_model.compile(loss='mean_squared_error',optimizer='adam')
		lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

	#def samp():
		X_test=[]
		for i in range(60,inputs_data.shape[0]):
			X_test.append(inputs_data[i-60:i,0])
		X_test=np.array(X_test)
		X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
		predicted_closing_price=lstm_model.predict(X_test)
		predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

	#def visual():
		train_data=new_dataset[:987]
		valid_data=new_dataset[987:]
		valid_data['Predictions']=predicted_closing_price
		plt.title("actual vs predicted")
		plt.xlabel("time")
		plt.ylabel("stock price")
		plt.plot(train_data["Close"])
		plt.plot(valid_data[['Close',"Predictions"]])
		plt.legend()
		st.pyplot()
		st.write(valid_data.head(),label="Predictions")


	def main():
		new()
		#scale()
		#mode()
		#samp()
		#visual()
	main()
