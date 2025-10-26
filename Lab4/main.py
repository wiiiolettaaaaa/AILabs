import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, SimpleRNN, Reshape

x = np.linspace(-5,5,1000)
y = np.linspace(-5,5,1000)
z = np.cos(abs(y)) + np.sin(x+y)

def modelTesting(model):
    model.compile(optimizer='adam', loss='mse')
    model.fit(x,z, epochs=20, batch_size=100)
    z_pred = model.predict(x)

    plt.plot(x,z,label='Actual')
    plt.plot(x, z_pred, label='Predicted')
    plt.legend()
    plt.show()

def feedforwardCreation(layers, neurons):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(1,)))
    for i in range(layers-1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, name='output'))
    return model

def cascadeforwardCreation(layers, neurons):
    inputLayer = Input(shape=(1,), name='input')
    current = Dense(neurons, activation='relu', input_shape=(1,))(inputLayer)
    for i in range(layers-1):
        concatenatedLayer = Concatenate()([inputLayer, current])
        current = Dense(neurons, activation='relu', input_shape=(1,))(concatenatedLayer)
    outputLayer = Dense(1, name='output')(current)
    model = Model(inputs=inputLayer, outputs=outputLayer)
    return model

def elmanCreation(layers, neurons):
    model = Sequential()
    model.add(Reshape((1,1), input_shape=(1,), name='input_reshape'))
    model.add(SimpleRNN(neurons, return_sequences=True, activation='relu',
    input_shape=(1,)))

    for i in range(layers-1):
        model.add(SimpleRNN(neurons, return_sequences=True, activation='relu'))
    model.add(Dense(1, name='output'))
    model.add(Reshape((1,), input_shape=(1,1), name='output_reshape'))
    return model

if __name__=="__main__":
    f1 = feedforwardCreation(1,10)
    f2 = feedforwardCreation(1,20)

    c1 = cascadeforwardCreation(1,20)
    c2 = cascadeforwardCreation(2,10)

    e1 = elmanCreation(1,15)
    e2 = elmanCreation(3,5)

    modelTesting(f1)
    modelTesting(f2)

    modelTesting(c1)
    modelTesting(c2)

    modelTesting(e1)
    modelTesting(e2)