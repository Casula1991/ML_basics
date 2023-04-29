import pandas as pd 
import numpy as np 
import load_data as ld
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

try:
	df = pd.read_csv(os.getcwd() + "\\training_dataframe.csv")
	print('Dataframe de treino carregado;')
except:
	print('Dataframe de treino sendo criado..')
	X, y = ld.get_data(size = 100)
	df = ld.to_dataframe(X,y)
	ld.save_df(df, df_name = 'training_dataframe')

print(df)

X, y = df.X.values.reshape(-1,1), df.target

def linear_fit(X, y):
	lm = LinearRegression().fit(X, y)
	return lm

def poly_fit(X, y, deg):
	pol = PolynomialFeatures(degree = deg)
	Xpol = pol.fit_transform(X)
	lm = LinearRegression().fit(Xpol, y)
	return lm

def save_model(model, model_name):
	print()
	print('Saving model ' + model_name + ' at: ', os.getcwd())
	print()
	print('Model path:', os.getcwd() + "\\" + model_name + ".sav")
	print()
	print()
	pickle.dump(model, open(os.getcwd() + "\\" + model_name + ".sav", 'wb'))


lin_reg = linear_fit(X, y)
poly2fit = poly_fit(X, y, deg = 2)
poly3fit = poly_fit(X, y, deg = 3)

save_model(lin_reg, model_name = 'lin_reg')
save_model(poly2fit, model_name = 'poly2fit')
save_model(poly3fit, model_name = 'poly3fit')

