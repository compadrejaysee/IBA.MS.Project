import uvicorn
import pandas as pd
from io import StringIO
from src.all_models import *
from fastapi import FastAPI, Query, FastAPI, File, UploadFile, Form

def get_model_results(model_name, data_file):
	df = pd.read_csv(StringIO(str(data_file.file.read(), 'utf-8')), encoding='utf-8')
	error, predictions = get_forecast(model_name, df)
	return (error, predictions)

app = FastAPI()

@app.post('/preprocess/{model_name}')
async def preprocess(model_name: str = Query(None, min_length=2, max_length=25), data_file: UploadFile = File(...)):
	"""
	build / call module that pre-processes data
	"""
	return {"model_name": model_name}

@app.post('/train/{model_name}')
async def train(model_name: str = Query(None, min_length=2, max_length=25), data_file: UploadFile = File(...)):
	"""
	build / call module that trains model on provided data
	"""
	return {"model_name": model_name}

@app.post('/validate/{model_name}')
async def validate(model_name: str = Query(None, min_length=2, max_length=25), data_file: UploadFile = File(...)):
	"""
	build / call module that validates model on provided data
	"""
	return {"model_name": model_name}	

@app.post('/forecast/{model_name}')
async def forecast(model_name: str = Query(None, min_length=2, max_length=25), data_file: UploadFile = File(...)):
	
	error, predictions = get_model_results(model_name, data_file)
	return {"model_name": model_name, "rmse": error}	

if __name__ == '__main__':
	localhost = "127.0.0.1"
	port = 8000
	uvicorn.run(app, host=localhost, port=port)
