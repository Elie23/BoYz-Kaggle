# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:54:57 2018

@author: Louis
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Importing the Keras libraries and packages
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import ZeroPadding2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

# fix random seed for reproducibility and load data
seed = 7
np.random.seed(seed)
x_test = np.loadtxt("test_x.csv", delimiter=",")
x = np.loadtxt("train_x.csv", delimiter=",")# load from text 
y = np.loadtxt("train_y.csv", delimiter=",")
x[x<255]=0

# reshape to be [samples][width][height][pixels]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20)
x_train = x_train.reshape(x_train.shape[0], 64, 64, 1).astype('float32')
x_val = x_val.reshape(x_val.shape[0], 64, 64, 1).astype('float32')
# normalize inputs from 0-255 to -1-1
x_train = x_train / 127.5 - 1 
x_val = x_val / 127.5 - 1
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
num_classes = y_val.shape[1]



model = Sequential()    
model.add(Conv2D(30, (5, 5), input_shape=(64, 64, 1), activation='relu'))
model.add(Conv2D(30, (5, 5), activation='relu'))
model.add(Conv2D(30, (5, 5), activation='relu'))
model.add(Conv2D(30, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(30, (5, 5), activation='relu'))
model.add(Conv2D(30, (5, 5), activation='relu'))
model.add(Conv2D(30, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))   
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.50))
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))
    # Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=50)
# Final evaluation of the model
scores = model.evaluate(x_val, y_val, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))



#plot learning curves
print(history.history.keys())

learning_curve = (history.history.keys())
np.savetxt('learning.csv',learning_curve,delimiter=',')
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import csv
with open('learning2.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.writer(f)
    for key,values in history.history.items():
        for value in values:
            w.writerow([key,value])


# Compute performance of model on the test set
import h5py
model.save('model2.h5')#0.95% on valid, 100 epochs avec seuil x>255=0
modeltest = load_model('model2.h5')
#preprocessing x_test
x_test[x_test<255]=0
x_test = x_test.reshape(x_test.shape[0], 64, 64, 1).astype('float32')
x_test = x_test / 127.5 - 1 

y_test = modeltest.predict_classes(x_test)
y_test_id = np.arange(y_test.shape[0]).reshape(y_test.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

pred = np.concatenate((y_test_id,y_test),axis=1).astype(int)
np.savetxt('pred2.csv',pred,delimiter=',',comments='',header='Id,label',fmt='%i')