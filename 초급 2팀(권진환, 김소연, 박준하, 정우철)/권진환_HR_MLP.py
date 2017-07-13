import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def df_to_np(df):
    return np.array(df)

data = pd.read_csv('prep_data.csv', index_col=0)
X_train, X_test, Y_train, Y_test = train_test_split(data.ix[:,:'support'],data.ix[:,'left'],test_size=0.3)


X_train = df_to_np(X_train)
X_test = df_to_np(X_test)
Y_train = df_to_np(Y_train)
Y_test = df_to_np(Y_test)

model = Sequential()
model.add(Dense(64, input_dim = 18,activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid' ))
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=10,batch_size=64)

score = model.evaluate(X_test, Y_test, batch_size=128)

print('loss: ', score[0])
print('accuracy: ',score[1])


