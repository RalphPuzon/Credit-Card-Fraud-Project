#MEGA CASE STUDY:
""" in the CC app case, identify the frauds using an SOM, and predict the
probability of being a true positive using an ANN"""

#code from SOM section:
#Self Organizing Maps:
import os
os.chdir('C:\\Users\\Ralph\\Desktop\\Courses\\DeepLearning\\Deep_Learning_A_Z\\Deep_Learning_A_Z\\Volume 2 - Unsupervised Deep Learning\\Part 4 - Self Organizing Maps (SOM)')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

#Feature Scaling:
#we are scaling to y (class). no need to scale this as well, just X.
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#Train the SOM:
#import minisom
from minisom import MiniSom
som = MiniSom(x = 10,
              y = 10,
              input_len = 15,
              sigma = 1.0,
              learning_rate = 0.5,
              random_seed= 1)

#initialize weights:
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#VIZ:
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's'] #circle, square
colors = ['r', 'g'] # red, green

#loop on each row of the dataset:
for i, j in enumerate(X):
    w = som.winner(j)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
#Obtain explicit list of 'cheaters':

mappings = som.win_map(X)
frauds = np.concatenate((mappings[7,1], mappings[8,4]),axis = 0)
frauds = sc.inverse_transform(frauds)

#PART 2:
#Supervised deep learning
#Creating the new MOF:

customers = dataset.iloc[:, 1:].values

#creating the target variable:
#initialize an array of zeros:
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

#Scale the data:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


#Make the ANN:

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform',
                     activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform',
                     activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 4)

# Predicting the probabilities of fraud:
y_pred = classifier.predict(customers)

#Making the data presentable:
#let's return the y_pred probs with their corresponding cust. IDs:
y_pred = np.concatenate((dataset.iloc[:,0:1],y_pred), axis = 1)
#TRICK!: .iloc array trick [:,0:1]

#TRICK!: sorting with proper association:
y_pred = y_pred[y_pred[:,1].argsort()]


