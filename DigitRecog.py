from keras.datasets import mnist
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
Xtrain = Xtrain/255
Xtest = Xtest/255
Xtrain = Xtrain.astype(np.float32)
Xtest = Xtest.astype(np.float32)
x_train_flattened = Xtrain.reshape(len(Xtrain), 28*28)
x_test_flattened = Xtest.reshape(len(Xtest), 28*28)

Ytrain = tf.keras.utils.to_categorical(Ytrain, 10)
Ytest = tf.keras.utils.to_categorical(Ytest, 10)
print(x_train_flattened.shape)
print(Ytrain.shape)
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import InputLayer
model = Sequential([
    InputLayer(input_shape= (784, )),
    Dense(units=20, activation='relu'),
    Dense(units=10, activation='softmax')
    ])
from tensorflow.python.keras.losses import CategoricalCrossentropy
model.compile(loss=CategoricalCrossentropy(), optimizer = "adam", metrics=["accuracy"])
history = model.fit(x_train_flattened, Ytrain, epochs=10)

score = model.evaluate(x_test_flattened, Ytest, verbose=1)
print('Test loss: ', score[0])
print('Test accurace: ', score[1])
Xtest = Xtest*255
pred = model.predict(x_test_flattened)

stop = 1
while stop != 0:
    image_index = random.randint(0, 10000)
    print("hello")
    plt.imshow(Xtest[image_index],cmap='Greys')
    plt.title(np.argmax(pred[image_index], axis=0))
    plt.pause(0.05)
    stop = int(input("input 0 to stop"))
    if stop==0:
        plt.close()
    plt.close()
plt.show()



