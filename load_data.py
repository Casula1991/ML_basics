import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os

PATH = os.getcwd()

def get_data(size):
	a1, a2, a3, a4, a5 = 1, -0.75, -0.5, 5, 2
	X = np.random.normal(size = size)
	error = np.random.normal(size = size, scale = 0.5)
	y = X* a1 + X * a2 + X * a3 + X * X * a4 + X * X * X * a5 + error
	return X, y

def to_dataframe(X,y):
	df = pd.DataFrame(np.c_[X,y], columns = ['X','y'])
	return df

def save_df(df, df_name):
	path_df = PATH + "\\" + df_name + ".csv"
	print('Saving df at: ', path_df)
	df.to_csv(path_df, index = False)

	




