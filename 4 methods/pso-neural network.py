
import numpy as np
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from pyswarm import pso


dataset = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df = pd.DataFrame(dataset)
def pso(n_input, n_h1, n_h2, n_output):
    model = Sequential()
    model.add(Dense(n_h1, n_input=n_input, activation='relu'))
    model.add(Dense(n_h2, activation='relu'))
    model.add(Dense(n_output, activation='softmax'))
    return model

def fitness_function(weights, n_input, n_h1, n_h2, n_output, X_train, y_train, X_val, y_val):
        model = create_model(n_input, n_h1, n_h2, n_output)
        model.set_weights(weights)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
        accuracy = model.evaluate(X_val, y_val, verbose=0)
        return -accuracy


(X_train, y_train), (X_test, y_test) = df.load_data()



def preprocess():
    input_dim = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], input_dim).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], input_dim).astype('float32') / 255
    y_train = np.eye(10)[y_train.astype('int32')]
    y_test = np.eye(10)[y_test.astype('int32')]
    return X_train, y_train, X_test, y_test


n_weights = (n_input * n_h1) + (n_h1 * n_h2) + (n_h2 * n_output) + n_h1 + n_h2 + n_output
lb = [-1] * n_weights
ub = [1] * n_weights
args = (n_input, n_h1, n_h2, output)

#%%
n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01


x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

