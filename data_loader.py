import numpy as np
import pickle
import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def load_cifar10_data():
   if os.path.exists('cifar10_data.pkl'):
       with open('cifar10_data.pkl', 'rb') as f:
           data = pickle.load(f)
       return data['x_train'], data['y_train'], data['x_test'], data['y_test']
   
   else:
       (x_train, y_train), (x_test, y_test) = cifar10.load_data()
       
       x_train = x_train.astype('float32') / 255.0
       x_test = x_test.astype('float32') / 255.0
       y_train = to_categorical(y_train, 10)
       y_test = to_categorical(y_test, 10)
       
       data = {
           'x_train': x_train,
           'y_train': y_train, 
           'x_test': x_test,
           'y_test': y_test
       }
       with open('cifar10_data.pkl', 'wb') as f:
           pickle.dump(data, f)
       
       return x_train, y_train, x_test, y_test