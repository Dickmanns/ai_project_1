import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
#import tensorflow.keras as ker
from hydra.utils import get_original_cwd

class model:
    def __init__(self) -> None:
        mnist = keras.datasets.mnist
        (x_train, self.y_train), (x_test, self.y_test) = mnist.load_data()
        self.x_train = keras.utils.normalize(x_train, axis=1)
        self.x_test = keras.utils.normalize(x_test, axis=1)

    def train_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28,28)))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(self.x_train, self.y_train, epochs=3)

        model.save(f'{get_original_cwd()}\handwritten.keras')

    def load_model(self) -> keras.models.Model:
        model = keras.models.load_model(f'{get_original_cwd()}/handwritten.keras')
        return model
    
    def test_model(self, model:keras.models.Model):
        loss, accuracy = model.evaluate(self.x_test, self.y_test)
    
    def test_model_selfData(self, model:keras.models.Model):
        image_number = 1
        while(os.path.isfile(f'{get_original_cwd()}/src/data/digits/digit_{image_number}.png')):
            try: 
                img = cv2.imread(f'{get_original_cwd()}/src/data/digits/digit_{image_number}.png')[:,:,0]
                img = np.invert(np.array([img]))
                prediction = model.predict(img)
                print(prediction)
                print(f"The number is a {np.argmax(prediction)}")
                plt.imshow(img[0],cmap=plt.cm.binary)
                plt.show()
            except:
                print("Error!")
            finally:
                print('next')
                image_number += 1