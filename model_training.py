import pandas as pd 
import numpy as np 
import load_data as ld
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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



