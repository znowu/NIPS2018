import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(7)
model = Sequential()
model.add( Dense(2, input_dim=2, activation = 'relu'))
model.add( Dense(1, input_dim=2, activation = 'sigmoid'))
model.compile( loss='binary crossentropy', optimize = 'adam', metrics =['accuracy'])

model.fit(X, Y, epochs = 20, batch_size = 2)
scores = model.evaluate(X, Y)