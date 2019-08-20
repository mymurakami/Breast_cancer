import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

previsores = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')


classificador= Sequential()

classificador.add(Dense(units=8,activation='relu',
                        kernel_initializer='normal',
                        input_dim=30))

#Adicionando dropout para evitar overfitting
classificador.add(Dropout(0.2)) #20%

#Adicionando mais 1 camada oculta
classificador.add(Dense(units=8,activation='relu',
                        kernel_initializer='normal'))
#Adicionando dropout para evitar overfitting
classificador.add(Dropout(0.2)) #20%

#Saida = sigmoid (entre 1 e 0)
classificador.add(Dense(units=1, activation='sigmoid'))


classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

classificador.fit(previsores,classes,batch_size=30,epochs=150)

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

previsao = classificador.predict(novo)
previsao = (previsao>0.5)