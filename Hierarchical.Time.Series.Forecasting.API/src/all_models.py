# import hts
# import optuna
import random
import warnings
import collections
import numpy as np
import pandas as pd
# import lightgbm as lgb
import statsmodels.api as sm
# from fbprophet import Prophet
from tqdm.notebook import tqdm as tqdm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

"""
This script contains functions for various timeseries forecasting techniques/methods. 
The list of supported models include:
	1. Naive
	2. Moving Average
	3. Holt Linear
	4. Exponential Smoothing
	5. Arima
	6. Prophet
	7. LightGBM 

"""

def get_forecast(model_name, data):
	day_cols = [col for col in data.columns if 'd_' in col]
	training_days = 730 # 2 yrs of data for training
	testing_days = 30 # 1 month of data for testing
	index = training_days + testing_days

	training_dataset = data[day_cols[-index:-testing_days]]
	validation_dataset = data[day_cols[-testing_days:]]

	available_models = {"Naive": naive, "Moving Average": moving_average, \
		"Holt Linear": holt_linear, "Exponential Smoothing": exponential_smoothing, \
		"Arima": arima, "Prophet": prophet}

	model = available_models[model_name]
	error, predictions = model(training_dataset, validation_dataset)

	return (error, predictions)

def naive(train_set, test_set):
	"""
	Method to forecast the next day's sales as the current day's sales
	"""
	predictions = []
	for i in range(len(test_set.columns)):
		if i == 0:
			predictions.append(train_set[train_set.columns[-1]].values)
		else:
			predictions.append(test_set[test_set.columns[i-1]].values)

	predictions = np.transpose(np.array([row.tolist() for row in predictions]))
	error = np.linalg.norm(predictions[:3] - test_set.values[:3])/len(predictions[0])

	return (error, predictions)

def moving_average(train_set, test_set):
	"""
	Method to forecast the next day's sales as the mean of last n days
	"""	
	n_days = 30
	predictions = []
	for i in range(len(test_set.columns)):
		if i == 0:
			predictions.append(np.mean(train_set[train_set.columns[-n_days:]].values, axis=1))
		if i < n_days+1 and i > 0:
			predictions.append(0.5 * (np.mean(train_set[train_set.columns[-n_days+i:]].values, axis=1) + \
									  np.mean(predictions[:i], axis=0)))
		if i > n_days+1:
			predictions.append(np.mean([predictions[:i]], axis=1))
		
	predictions = np.transpose(np.array([row.tolist() for row in predictions]))
	error = np.linalg.norm(predictions[:3] - test_set.values[:3])/len(predictions[0])

	return (error, predictions)

def holt_linear(train_set, test_set):
	"""
	Method to capture the high-level trends in time series data using a linear function
	"""
	n_days = 30
	predictions = []
	for row in tqdm(train_set[train_set.columns[-n_days:]].values[:3]):
		fit = Holt(row).fit(smoothing_level = 0.3, smoothing_trend = 0.01)
		predictions.append(fit.forecast(n_days))
	predictions = np.array(predictions).reshape((-1, n_days))
	error = np.linalg.norm(predictions - test_set.values[:len(predictions)])/len(predictions[0])

	return (error, predictions)

def exponential_smoothing(train_set, test_set):
	"""
	Method to give different weightage to different time steps, instead of giving the same weightage 
	to all time steps (like the moving average method). This ensures that recent sales data is given 
	more importance than old sales data while making the forecast
	"""	
	n_days = 30
	predictions = []
	for row in tqdm(train_set[train_set.columns[-n_days:]].values[:3]):
		fit = ExponentialSmoothing(row, seasonal_periods=3).fit()
		predictions.append(fit.forecast(n_days))
	predictions = np.array(predictions).reshape((-1, n_days))
	error = np.linalg.norm(predictions[:3] - test_set.values[:3])/len(predictions[0])

	return(error, predictions)

def arima(train_set, test_set):
	"""
	Method for Auto Regressive Integrated Moving Average, while exponential smoothing models were based on 
	a description of trend and seasonality in data, ARIMA models aim to describe the correlations 
	in the time series
	"""
	n_days = 30
	predictions = []
	for row in tqdm(train_set[train_set.columns[-n_days:]].values[:3]):
		fit = sm.tsa.statespace.SARIMAX(row, seasonal_order=(0, 1, 1, 7)).fit()
		predictions.append(fit.forecast(n_days))
	predictions = np.array(predictions).reshape((-1, n_days))
	error = np.linalg.norm(predictions[:3] - test_set.values[:3])/len(predictions[0])

	return (error, predictions)

def prophet(train_set, test_set):
	"""
	Method for an additive model where non-linear trends are fit with yearly, 
	weekly, and daily seasonality, including holiday effects. It works best with time series 
	that have strong seasonal effects and several seasons of historical data. It is also 
	supposed to be more robust to missing data and shifts in trend compared to other models
	"""	
	n_days = 30
	dates = ["2016-12-" + str(i) for i in range(1, n_days+1)]
	predictions = []
	for row in tqdm(train_set[train_set.columns[-n_days:]].values[:3]):
		df = pd.DataFrame(np.transpose([dates, row]))
		df.columns = ["ds", "y"]
		model = Prophet(daily_seasonality=True)
		model.fit(df)
		future = model.make_future_dataframe(periods=n_days)
		forecast = model.predict(future)["yhat"].loc[n_days:].values
		predictions.append(forecast)
	predictions = np.array(predictions).reshape((-1, n_days))
	error = np.linalg.norm(predictions[:3] - test_set.values[:3])/len(predictions[0])	

	return (error, predictions)

def lightgbm(train_set, test_set):
	"""
	Method to implement Light GBM which is a fast, distributed, high-performance gradient 
	boosting framework based on decision tree algorithm, used for ranking, classification 
	and many other machine learning tasks
	"""
	n_days = 30
	model = lgb.LGBMRegressor(**param)
	model.fit(X_train, y_train)

	# model.fit(X_train, y_train, categorical_feature=[encoding_col]) TODO: Add support for categorical features 
		
	predictions = model.predict(test_set)
	predictions = np.array(predictions).reshape((-1, n_days))

	error = np.linalg.norm(predictions[:3] - test_set.values[:3])/len(predictions[0])	

	return (error, predictions)
