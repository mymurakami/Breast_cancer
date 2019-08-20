import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
#import para validação cruzada
from keras.wrappers.scikit_learn import  KerasClassifier
from sklearn.model_selection import cross_val_score




def criarRede ():
    
    classificador= Sequential()

    classificador.add(Dense(units=16,activation='relu',
                            kernel_initializer='random_uniform',
                            input_dim=30))
    
    #Adicionando dropout para evitar overfitting
    classificador.add(Dropout(0.2)) #20%
    
    #Adicionando mais 1 camada oculta
    classificador.add(Dense(units=16,activation='relu',
                            kernel_initializer='random_uniform'))
    #Adicionando dropout para evitar overfitting
    classificador.add(Dropout(0.2)) #20%
    
    #Saida = sigmoid (entre 1 e 0)
    classificador.add(Dense(units=1, activation='sigmoid'))
    
    optimizador = keras.optimizers.Adam(lr=0.001,decay=0.0001, clipvalue =0.8)
    
    classificador.compile(optimizer=optimizador, loss='binary_crossentropy',
                          metrics=['binary_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn=criarRede,
                                epochs=100,
                                batch_size=10)

#cv = qtd de vezes do teste (K), 10 rodadas
resultados=cross_val_score(estimator=classificador,
                           X = previsores,
                           y = classes,
                           cv = 10,scoring='accuracy')



media = resultados.mean()
desvio = resultados.std() 
#desvio padrão, quanto maior o valor, mais a tendencia 
#de overfitting, se adapta demais aos dados e com dados novo não vai se adpatar
#Exemplo: apenas elefantes grande e se tiver dados novos com elefantes pequenos
#pode se atrapalhar para classificação