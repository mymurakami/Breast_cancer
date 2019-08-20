import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')


classificador= Sequential()

classificador.add(Dense(units=10,activation='relu',
                        kernel_initializer='normal',
                        input_dim=30))

#Adicionando dropout para evitar overfitting
classificador.add(Dropout(0.2)) #20%

#Adicionando mais 1 camada oculta
classificador.add(Dense(units=10,activation='relu',
                        kernel_initializer='normal'))
#Adicionando dropout para evitar overfitting
classificador.add(Dropout(0.4)) #20%

#Saida = sigmoid (entre 1 e 0)
classificador.add(Dense(units=1, activation='sigmoid'))


classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

classificador.fit(previsores,classes,batch_size=10,epochs=100)

#Salvar os parametros em disco do classificador

classificador_json = classificador.to_json()

with open('classificador_breast.json','w') as json_file:
    json_file.write(classificador_json)
    
#Salvar pesos
classificador.save_weights('classificador_breast.h5')

resulto = classificador.evaluate(previsores,classes) 

