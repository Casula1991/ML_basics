import pandas as pd 
import numpy as np 
import load_data as ld
import os

try:
	df = pd.read_csv(os.getcwd() + "\\training_dataframe.csv")
	print('Dataframe de treino carregado;')
except:
	print('Dataframe de treino sendo criado..')
	X, y = ld.get_data(size = 100)
	df = ld.to_dataframe(X,y)
	ld.save_df(df, df_name = 'training_dataframe')

print(df)



