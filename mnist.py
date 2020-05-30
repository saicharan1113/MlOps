#!/usr/bin/env python
# coding: utf-8

# In[24]:


# if you keras is not using tensorflow as backend set "KERAS_BACKEND=tensorflow" use this command

from keras.utils import np_utils 
from keras.datasets import mnist 
import seaborn as sns
from keras.initializers import RandomNormal
from keras.models import Sequential 
from keras.layers import Dense, Activation 


# In[25]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import numpy as np
import time


# In[26]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[27]:


#X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[28]:



X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) 
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]) 


# In[29]:



X_train = X_train/255
X_test = X_test/255


# In[30]:


Y_train = np_utils.to_categorical(y_train, 10) 
Y_test = np_utils.to_categorical(y_test, 10)


# In[31]:


# some model parameters

output_dim = 10
input_dim = X_train.shape[1]

batch_size = 128 
nb_epoch = 20


# In[32]:


# Multilayer perceptron

# https://intoli.com/blog/neural-network-initialization/ 

from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout

model_batch = Sequential()

model_batch.add(Dense(512, activation='relu', input_shape=(input_dim,), kernel_initializer=initializers.he_normal(seed=None)))
#model_batch.add(BatchNormalization())
model_batch.add(Dropout(0.5))


model_batch.add(Dense(420, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
#model_batch.add(BatchNormalization())
model_batch.add(Dropout(0.5))


model_batch.add(Dense(310, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
#model_batch.add(BatchNormalization())
model_batch.add(Dropout(0.5))



model_batch.add(Dense(230, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
#model_batch.add(BatchNormalization())
model_batch.add(Dropout(0.5))




model_batch.add(Dense(100, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
#model_batch.add(BatchNormalization())
model_batch.add(Dropout(0.5))




model_batch.add(Dense(output_dim, activation='softmax'))


model_batch.summary()


# In[33]:


model_batch.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[34]:


history = model_batch.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test)) 


# In[23]:


score = model_batch.evaluate(X_test, Y_test, verbose=0) 
#print('Test score:', score[0]) 
print('Test accuracy:', score[1])


# In[ ]:




