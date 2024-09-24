import json
import time

from sklearn.model_selection import train_test_split
import pandas as pd
from src.utils.evaluate import new_accuracy
from src.utils.labeling_from_score import labeling
import numpy as np

class GradientDescent:
    def __init__(self, x, first_y, second_y, test_size=0.2, random_state=40, num_iterations= 100):
        self.num_iterations = num_iterations
        self.x_train, self.x_test, self.first_y_train, self.first_y_test, self.second_y_train, self.second_y_test = (
            train_test_split(
            x, first_y, second_y, stratify=second_y, test_size=test_size, random_state=random_state
        ))

    @staticmethod
    def new_error_func(y, second_y, y_predict):
        errors = [0] * len(y_predict)
        # Lặp qua từng phần tử trong mảng
        for i in range(len(y_predict)):
            if y[i] == second_y[i]:
                errors[i] = y_predict[i] - y[i]
            else:
                min_value = min(y[i], second_y[i])
                max_value = max(y[i], second_y[i])
                if y_predict[i] <= min_value:
                    errors[i] = y_predict[i] - min_value
                if y_predict[i] >= max_value:
                    errors[i] = y_predict[i] - max_value
        return errors

    @staticmethod
    def mean_squared_error(x, y, w):
        n = len(y)
        y_predict = labeling(np.dot(x, w))
        error = y_predict - y
        mse = np.sum(error ** 2) / n
        return mse

    # Hàm tính gradient của hàm lỗi (MSE)
    def gradient_mean_squared_error(self, x, y, second_y, w):
        n = len(y)
        y_predict = labeling(np.dot(x, w))
        # error = y_pred - y
        error = self.new_error_func(y, second_y, y_predict)
        gradient = 2 * np.dot(x.T, error) / n
        return gradient

    # Gradient descent để tối ưu hóa hàm lỗi (MSE)
    def stochastic_gradient_descent(
            self, x, y, second_y, learning_rate=0.001, num_iterations=10000, decay_rate=0.95
    ):
        w = np.random.uniform(0, 1, x.shape[1])
        for i in range(num_iterations):
            random_index = np.random.randint(0, len(x))
            x_sample = x[random_index: random_index + 1]
            y_sample = y[random_index: random_index + 1]
            second_y_sample = second_y[random_index: random_index + 1]
            grad = self.gradient_mean_squared_error(x_sample, y_sample, second_y_sample, w)
            w -= learning_rate * grad
            learning_rate *= decay_rate  # Decay learning rate

        return w

    def run(self):
        begin = time.time()
        print('------START EXECUTE GD------')
        max1 = 0
        res = None
        for i in range(self.num_iterations):
            learned_weights = self.stochastic_gradient_descent(self.x_train, self.first_y_train, self.second_y_train)
            predict = labeling(self.x_test.dot(learned_weights))
            acc = new_accuracy(self.first_y_test, self.second_y_test, predict)
            if acc > max1:
                max1 = acc
                res = learned_weights
                print(f'new learning weights: {res}')
        predicted_labels = labeling(self.x_test.dot(res))
        result = {
            "first_y_test": [i for i in self.first_y_test],
            "second_y_test": [i for i in self.second_y_test],
            "predicted_labels": [i for i in predicted_labels]
        }
        df = pd.DataFrame(result)
        df.to_csv('results/gd.csv')
        print(f'------EXECUTE IN {time.time() - begin}------')

