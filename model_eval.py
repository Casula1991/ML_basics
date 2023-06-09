import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import load_data as ld
from sklearn.preprocessing import PolynomialFeatures
import pickle

B = int(input('quantos datasets de testes para avaliar? '))
S = int(input('qual o tamanho de cada dataset? '))
save_it = input('vc quer salvar os resultados? (sim/nao) ')

final_plot = input('vc quer o grafico no fim? (sim/nao)')

path__ = 'C:\\Users\\Fabio Casula\\OneDrive\\Área de Trabalho\\MyProjects_v2\\ml_basics\\'

lm = pickle.load(open(path__ + "lin_reg.sav",'rb'))
p2 = pickle.load(open(path__ + "poly2fit.sav",'rb'))
p3 = pickle.load(open(path__ + "poly3fit.sav",'rb'))

pol2 = PolynomialFeatures(degree = 2)
pol3 = PolynomialFeatures(degree = 3)

def rmse(ytrue, ypred):
	return np.sqrt(((ytrue-ypred)**2).mean())

L1, L2, L3 = [], [], []
print()
print('Results:')
print()
for b in range(B):
	X, y = ld.get_data(size = S)
	X = X.reshape(-1,1)
	lmpred = lm.predict(X)
	p2pred = p2.predict(pol2.fit_transform(X))
	p3pred = p3.predict(pol3.fit_transform(X))
	r1 = rmse(ytrue = y, ypred = lmpred)
	r2 = rmse(ytrue = y, ypred = p2pred)
	r3 = rmse(ytrue = y, ypred = p3pred)
	if save_it == 'sim':
		L1.append(r1); L2.append(r2); L3.append(r3)
	print('lin:', r1)
	print('pol2:', r2)
	print('pol3:', r3)
	print()

dfres = pd.DataFrame(np.c_[L1,L2,L3], columns = ['lin_reg','poly2fit','poly3fit'])
print('df_results head:')
print(dfres.head())
print()
print('Avg error:')
print(dfres.mean(axis = 0))
if final_plot == 'sim':
	plt.figure(figsize = [10,6])
	plt.hist(L1, bins = 50, label = 'lin_reg')
	plt.hist(L2, bins = 50, label = 'poly2')
	plt.hist(L3, bins = 50, label = 'poly3')
	plt.legend(fontsize = 14)
	plt.grid()
plt.show()



