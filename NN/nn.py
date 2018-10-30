import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam


with open('X_train.pkl_1.pkl', 'rb') as fx:
    X = pickle.load(fx)
    input_length = X[0].shape[0]

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=input_length))
model.add(Dropout(0.2))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = Adam()
model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])

file_number = 1
total = 0
    
with open('Y_train.pkl', 'rb') as fy:
    Y_all = pickle.load(fy)

while True:
    file_path = 'X_train.pkl_' + str(file_number) + '.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fx:
            X = pickle.load(fx)
        examples_length = X.shape[0]
        Y = Y_all[total:total+examples_length]
        model.fit(X, Y, batch_size=32, epochs=5, verbose=2)
        total += examples_length
    else:
        break
    file_number += 1
	
with open('nn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
