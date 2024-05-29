import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []


"""
CONTEXTO:
    TESTE FARMACÊUTIC, 5% DAS PESSOAS ENTRE 13 A 64 ANOS TIVERAM EFEITOS COLATERAIS
    E 95% DAS PESSOAS ENTRA 65 A 100 ANOS TIVERAM EFEITOS COLATERAIS
"""
for i in range(50):
    #5% EFEITOS COLATERAIS
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    #5% SEM EFEITOS COLATERAIS
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)
    
for i in range(1000):
    #95% SEM EFEITOS COLATERAIS
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    #95% EFEITOS COLATERAIS
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)
    
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

"""
ESTE CÓDIGO SÓ FUNCIONA PARA GPU's
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPU DISPONÍVEL: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""

#CRIANDO AS HIDDEN LAYERS DA REDE NEURAL, COM OS NÓS E A FUNÇÃO DE ATIVAÇÃO DE CADA NÓ
model = Sequential([
        Dense(units=16, input_shape=(1,), activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=2, activation='softmax')
    ])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.3, batch_size=10, epochs=30, shuffle=True, verbose=0)

#EXECUTANDO TESTES

test_labels = []
test_samples = []


"""
CONTEXTO:
    TESTE FARMACÊUTIC, 5% DAS PESSOAS ENTRE 13 A 64 ANOS TIVERAM EFEITOS COLATERAIS
    E 95% DAS PESSOAS ENTRA 65 A 100 ANOS TIVERAM EFEITOS COLATERAIS
"""
for i in range(10):
    #5% EFEITOS COLATERAIS
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    
    #5% SEM EFEITOS COLATERAIS
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)
    
for i in range(200):
    #95% SEM EFEITOS COLATERAIS
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    
    #95% EFEITOS COLATERAIS
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)
    
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

for i in predictions:
    print(i)
    
rounded_predictions = np.argmax(predictions, axis=-1)

for i in rounded_predictions:
    print(i)