#Importar bibliotecas

import numpy as np

import pandas as p

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.datasets import load_wine #importa bibliotecas separadas

from sklearn.model_selection import (train_test_split)

from sklearn.preprocessing import StandardScaler 

from sklearn.pipeline import Pipeline 

from sklearn.esemble import RandomForestClassifier 

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

print("As bibliotecas podem ser usadas para carregar o data set")

#Carregar data set

dataset= load_wine()
print("o dataset foi carregado")
df= dataset

#Separar X e y 

X= df.data.data 

y= df['target']

#Treinar o modelo #X e y são as variáveis que irão ajudar a treinar o data set.

data=("X,y")
X= data.data
y= data.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 42)
print("As variáveis testadas foram : {X,y}")


#PCA 

pca= ("componentes")
plt.scatter = (pca)
colorbar= (pca)



#Gráfico 

plt.figure(figsize=(6,8))
plt.title("As variáveis testadas foram")
plt.xlabel("Variável 1")
plt.ylabel("Variável 2")
plt.legend()
plt.show


#Testar o Random Forest 

modelo = RandomForestClassifier(random_state = 42)

modelo.fit=(X_test,y_test)


#Previsão do modelo 
previsao_do_modelo = ['X_train,X_test,y_train,y_test = train_test_split']
y_predict=modelo.predict(X_train,y_test)


#Acurácia
acurácia = accuracy_score = (y_test,y_predict)
print("A acuracia do modelo é: {acuracia}")


#Matriz de Confusão 

confusion_matrix= (y_test,y_predict)

print(confusion_matrix)



#F1 Score

f1_score= (y_train,y_predict)

print(f1_score)



