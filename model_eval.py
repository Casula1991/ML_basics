import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import load_data as ld
from sklearn.preprocessing import PolynomialFeatures
import pickle

B = int(input('quantos datasets de testes para avaliar? '))
S = int(input('qual o tamanho de cada dataset? '))
save_it = input('vc quer salvar os datasets? (sim/nao) ')

path__ = 'C:\\Users\\Fabio Casula\\OneDrive\\Área de Trabalho\\MyProjects_v2\\ml_basics\\'

lm = pickle.load(open(path__ + "lin_reg.sav",'rb'))
p2 = pickle.load(open(path__ + "poly2fit.sav",'rb'))
p3 = pickle.load(open(path__ + "poly3fit.sav",'rb'))

pol2 = PolynomialFeatures(degree = 2)
pol3 = PolynomialFeatures(degree = 3)

#print(lm, p2, p3)

def rmse(ytrue, ypred):
	return np.sqrt(((ytrue-ypred)**2).mean())

for b in range(B):
	X, y = ld.get_data(size = S)
	X = X.reshape(-1,1)
	lmpred = lm.predict(X)
	p2pred = p2.predict(pol2.fit_transform(X))
	p3pred = p3.predict(pol3.fit_transform(X))

	print('lin:', rmse(ytrue = y, ypred = lmpred))
	print('pol2:', rmse(ytrue = y, ypred = p2pred))
	print('pol3:', rmse(ytrue = y, ypred = p3pred))
	print()


