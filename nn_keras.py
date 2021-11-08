import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

class NN_Keras():

    def __init__(self):
        if not os.path.isdir('models/'):
            os.makedirs('models/')

    def scale_data(self, X, X_linspace):
        ''' Scales data using the keras StandardScaler

        Args:
            X: array of x values of training data
            X_linspace: array of X values within x_min and x_max (specified in app.py) with a step of x_step, e.g. [0, 0.1, 0.2,...,10]
        
        Returns:
            X_scaled, X_linspace_scaled: Scaled version of the input arrays
        '''
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X.reshape(-1,1))
        X_linspace_scaled = self.scaler_X.transform(X_linspace.reshape(-1,1))

        return X_scaled, X_linspace_scaled

    def init_model(self, X, num_neurons, learning_rate):
        ''' Initializes the model

        Args:
            X: array of x values of training data
            num_neurons: number of neurons in the hidden layer
            learning_rate: initial learning rate of the Adam optimizer
        '''
        self.model = Sequential()
        self.model.add(Dense(num_neurons, kernel_initializer='normal', input_dim=X.shape[1], activation='tanh'))
        self.model.add(Dense(1, activation='linear'))

        self.model.compile(loss='mse', 
                           optimizer=Adam(learning_rate=learning_rate), 
                           metrics=['mse'])

    def create_checkpoint_callback(self, save_freq=10):
        '''
        Creates a callback function that saves the model at every save_freq epochs to the folder "models/"

        Args:
            save_freq: Model saving frequency, e.g. saves the model at every 10 trained epochs
        '''
        checkpoint = keras.callbacks.ModelCheckpoint('models/model{epoch:08d}.h5', save_freq=save_freq)
        return checkpoint 

    def fit(self, X, y, epochs):
        ''' Trains the model using the training data
        
        Args:
            X: array of x values of training data
            y: array of y values of training data
            epochs: number of training epochs
        '''
        m = X.shape[0]

        history = self.model.fit(X, 
                                y, 
                                epochs=epochs, 
                                batch_size=m, 
                                verbose=1, 
                                validation_split=0.1,
                                callbacks=[self.create_checkpoint_callback(save_freq=10)])

    def delete_existing_model_checkpoints(self):
        ''' Deletes all the exisitng model checkpoints within folder "models/"
        '''
        ls = os.listdir('models/')
        for model_file in ls:
            os.remove(os.path.join('models/', model_file))


    def predict(self, X):
        ''' Calculates model predictions

        Args:
            X: array of x values
        
        Returns:
            yhat: model predictions at the input values in X
        '''
        yhat = self.model.predict(X)

        return yhat

    def predict_at_checkpoints(self, X):
        ''' Calculates model predictions at every saved checkpoint model

        Args:
            X: array of x values

        Returns
            df: DataFrame (required for plotly animations) with three columns 
                1. X: input data, 
                2. Predictions: model predictions, 
                3. epoch: predictions at specific epoch
        '''
        arr_X = []
        arr_yhat = []
        arr_epoch = []
        for model_file in os.listdir('models/'):
            
            epoch = int(model_file[5:-3])
            
            model = keras.models.load_model(os.path.join('models', model_file))
            yhat = model.predict(self.scaler_X.transform(X.reshape(-1,1)))
            
            arr_X += list(X.squeeze())
            arr_yhat += list(yhat.squeeze())
            arr_epoch += [epoch]*len(X)
    

        df = pd.DataFrame({
            'X': arr_X,
            'Predictions': arr_yhat,
            'epoch': arr_epoch
        })

        return df