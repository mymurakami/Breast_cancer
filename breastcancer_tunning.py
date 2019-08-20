import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
#import para validação cruzada
from keras.wrappers.scikit_learn import  KerasClassifier
from sklearn.model_selection import GridSearchCV #Search dos melhores parametros


previsores = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')

def criarRede (optimizer, loos, kernel_initializer, activation, neurons):
    
    classificador= Sequential()

    classificador.add(Dense(units=neurons,activation=activation,
                            kernel_initializer=kernel_initializer,
                            input_dim=30))
    
    #Adicionando dropout para evitar overfitting
    classificador.add(Dropout(0.2)) #20%
    
    #Adicionando mais 1 camada oculta
    classificador.add(Dense(units=neurons,activation=activation,
                            kernel_initializer=kernel_initializer))
    #Adicionando dropout para evitar overfitting
    classificador.add(Dropout(0.2)) #20%
    
    #Saida = sigmoid (entre 1 e 0)
    classificador.add(Dense(units=1, activation='sigmoid'))
    
    
    classificador.compile(optimizer=optimizer, loss=loos,
                          metrics=['binary_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size':[10, 30,],
              'epochs':[50, 150],
              'optimizer':['adam', 'sgd'],
              'loos':['binary_crossentropy','hinge'],
              'kernel_initializer':['random_uniform','normal'],
              'activation':['relu','tanh'],
              'neurons':[16,8]
              }

grid_search = GridSearchCV(estimator = classificador, 
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv=5)
#cv=qtd de testes, combinação de todos com todos

grid_search = grid_search.fit(previsores, classes)
melhores_parametros = grid_search.best_params_
melhor_previsao = grid_search.best_score_