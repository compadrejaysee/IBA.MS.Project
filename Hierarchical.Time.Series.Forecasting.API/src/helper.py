import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_features(data):
	"""
	Given dataframe of time series, create features for modeling
	"""
	data["is_weekend"] = data['date'].dt.dayofweek > 4
	data['day'] = data['date'].dt.day
	data['week'] = data['date'].dt.isocalendar().week.astype(int)
	data['month'] = data['date'].dt.month
	data['year'] = data['date'].dt.year	

	return data

def apply_encoding(data, encoding_col, label_encoder=None):
	"""
	Encode categorical column as numerical labels for better performance
	"""	
	if label_encoder is None:
		label_encoder = LabelEncoder()
		data[encoding_col] = label_encoder.fit_transform(data[encoding_col])
	else:
		data[encoding_col] = label_encoder.fit_transform(data[encoding_col])

	return data.set_index('date', inplace=True)      

def get_rolling_mean(data, encoding_col, col):
	"""
	Get rolling mean over selected windows
	"""
	for i in [7,14,30]:
		data['rolling_mean_'+str(i)] = data.groupby([encoding_col])[col].transform(lambda x: x.shift(lags).rolling(i).mean())
		data['rolling_std_'+str(i)]  = data.groupby([encoding_col])[col].transform(lambda x: x.shift(lags).rolling(i).std())

def get_lags(data, lags=7, encoding_col, col):
	"""
	Get previous values over selected shifts
	"""		
	for lag in range(lags):
		data[f'sales_lag{lag+1}'] = data[col].shift(lag+1)
		else:
			data[f'sales_lag{lag+1}'] = data.groupby([encoding_col])[col].shift(lag+1)
		lag_columns.append(f'sales_lag{lag+1}')
	data.dropna(subset=lag_columns, inplace=True)	
