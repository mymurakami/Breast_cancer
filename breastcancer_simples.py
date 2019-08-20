import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')

#Separação dos dados testes e de treinamento através do sklearn
#com a porcentagem de 25%
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste,  classe_treinamento, classe_teste = train_test_split(previsores,classes,test_size=0.25)

import keras
#Sequencial rede neural
from keras.models import Sequential
#Camdas densas na rede neural - Cada um dos neuronios é ligados 
#aos neuronios da camada subsequente
from keras.layers import Dense 

classificador= Sequential()
#Documentacao Keras
#Camada Oculta
#units=quantidade de neuronios da camada oculta
#calculo de neuronios = 30 das entradas + qtd de saidas (1) = 31/2 = 15.5 = 16
#kernel_initializer=inicializador dos pesos
#input_dim=qtd de elementos de entrada
#activation = funcao de ativacao
classificador.add(Dense(units=16,activation='relu',
                        kernel_initializer='random_uniform',
                        input_dim=30))

#Adicionando mais 1 camada oculta
classificador.add(Dense(units=16,activation='relu',
                        kernel_initializer='random_uniform'))

#Saida = sigmoid (entre 1 e 0)
classificador.add(Dense(units=1, activation='sigmoid'))

#Configuracao do optimizador
#lr = learning rate
#decay = reducao do learning rate
#clipvalue = ˜prender˜ o valor max value and min value
optimizador = keras.optimizers.Adam(lr=0.001,decay=0.0001, clipvalue =0.8)
#funcao para fazer os ajustes dos pesos
#descida do gradiente nesse caso stocastico
#loss = calculo do erro
#metrics=porcentagem binaria
#classificador.compile(optimizer='adam', loss='binary_crossentropy',
#                      metrics=['binary_accuracy'])

classificador.compile(optimizer=optimizador, loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

#Iniciar o treinamento com os parametros de treinamento
#batch_size = N, então a cada N registro calcula-se o erro e atualiza os pesos.
#epochs = quantidade de epocas, ou seja, quantidade de vezes que vai se 
#realizar esse treinamento
classificador.fit(previsores_treinamento,classe_treinamento,batch_size=10,
                  epochs=100)

#2, 1 pesos entrada -> camanda oculta, 2 pesos para o Bias para a camada oculta
pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))

#da primeira camada oculta para a segunda camada oculta (contem 2 partes por causa do BIAS)
pesos1 = classificador.layers[1].get_weights()

#da ultima camada oculta para a a camada de saida (contem 2 partes por causa do BIAS)
pesos2 = classificador.layers[2].get_weights()


#Realizar teste com os valores provisores_Teste
previsoes = classificador.predict(previsores_teste)

#Transforma em 0 e 1 a saida
previsoes = (previsoes >0.5)

#Calculo da precisao de acordo com a base teste
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)

#Comparações de quantidade de dados de saida do teste com o valores calculados

matriz = confusion_matrix(classe_teste,previsoes)


#via keras
#onde 0 é o valor da funcao erro
# e 1 é o valor de precisao
resultado = classificador.evaluate(previsores_teste,classe_teste)










