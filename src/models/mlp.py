import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

from src.utils.evaluate import CustomAccuracy, CustomSparseCategoricalCrossEntropy


class MLP:
    def __init__(self, x, first_y, second_y, test_size=0.2, random_state=40):
        self.model = None
        self.x_train, self.x_test, self.first_y_train, self.first_y_test, self.second_y_train, self.second_y_test = (
            train_test_split(
            x, first_y, second_y, stratify=second_y, test_size=test_size, random_state=random_state
        ))
        self.y = np.stack((first_y, second_y), axis=1)
        self.y_array2d = np.stack((self.first_y_train, self.second_y_train), axis=1)
        self.y_test_array2d = np.stack((self.first_y_test, self.second_y_test), axis=1)
        self.custom_accuracy = CustomAccuracy()
        self.custom_loss = CustomSparseCategoricalCrossEntropy()

    def generate_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(14,),
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(5, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ])

    def run(self, epochs=10, batch_size=64):
        begin = time.time()
        print('------START EXECUTE MLP------')
        self.generate_model()
        self.model.compile(optimizer='adam', loss=self.custom_loss, metrics=[self.custom_accuracy])
        self.model.fit(self.x_train, self.y_array2d, epochs=epochs, batch_size=batch_size)
        predictions = self.model.predict(self.x_test)
        predicted_labels = np.argmax(predictions, axis=1)
        result = {
            "first_y_test": [i for i in self.first_y_test],
            "second_y_test": [i for i in self.second_y_test],
            "predicted_labels": [i for i in predicted_labels]
        }
        df = pd.DataFrame(result)
        df.to_csv('results/mlp.csv')
        print(f'------EXECUTE IN {time.time() - begin}------')