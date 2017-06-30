# -*- coding: utf-8 -*-
# part 1 - Data processing 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

# [ data preprocessing ]
# importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# 1. categorical data (0,1, numeric/ otherwiese label encoder required)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1-1. numerical transformation 
labelencoder_x_1 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1]) 

labelencoder_x_2 = LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])

# 1-2. dummy variabel for country 
onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()

# 1-3. avoid dummy variable trap  
x = x[:, 1:]

# 2. data split 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

# 3. Feature scaling = to avoid one independent variablbe domincating others 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# [ ANN Construction ]  
import keras 

# 1. sequential and dense modules 
from keras.models import Sequential
from keras.layers import Dense

# 2. initializing ANN / Definie as sequence of layers 
classifier = Sequential ()
## classifier = object of an sequential class   

# 3. Adding layers (input, hidden)
# 11 independent variable = 11 input nodes 
# dense function will help put random weights (not zeros)
# output = 1 (binary)
# hidden = avg of input & output 
# uniform funtion to initialise wiegh uniformly 
# ReLu for input and sigmoid for output 

# initialise input layer &  first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer/ keep number of avg of output & input 
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer 
# to get the probability, activation function was changed to sigmoid 
# if dependent variable has 3 categories, output dim should be 3 
# here output is ont-hot encoded = output dim is 1 
# if dependent varialbe has more than 2 categories, softmax works better 
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling ANN 
# loss function theoretical search required 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
 # Fitting the ANN to the Training set 
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test and results 
y_pred = classifier.predict(x_test)
# establish threshold and change the result to binary 
y_pred = (y_pred > 0.5)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)