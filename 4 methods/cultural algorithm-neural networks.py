import numpy as np
import pandas as pd
import torch

dataset = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df = pd.DataFrame(dataset)
from torch import nn

class CA(torch.optim.Optimizer):
    def create_model(input_dim, hidden1_dim, hidden2_dim, output_dim):
        model = Sequential()
        model.add(Dense(hidden1_dim, input_dim=input_dim, activation='relu'))
        model.add(Dense(hidden2_dim, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))
        return model
    def fitness_function(individual, data):
        weights = individual[0]
        input_dim, hidden1_dim, hidden2_dim, output_dim = data['dimensions']
        X_train, y_train, X_val, y_val = data['train_val_data']
        model = create_model(input_dim, hidden1_dim, hidden2_dim, output_dim)
        model.set_weights(weights)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        return -accuracy
    
(X_train, y_train), (X_test, y_test) = df.load_data()

def preprocess():
    input_dim = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], input_dim).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], input_dim).astype('float32') / 255
    y_train = np.eye(10)[y_train.astype('int32')]
    y_test = np.eye(10)[y_test.astype('int32')]
    return X_train, y_train, X_test, y_test

n_weights = (input_dim * hidden1_dim) + (hidden1_dim * hidden2_dim) + (hidden2_dim * output_dim) + hidden1_dim + hidden2_dim + output_dim
lower_bound = -1
upper_bound = 0


