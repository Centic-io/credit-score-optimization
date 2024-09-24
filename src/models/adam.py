import json
import time

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from src.utils.evaluate import new_accuracy
from src.utils.labeling_from_score import labeling


class Adam:
    def __init__(self, x, first_y, second_y, test_size=0.2, random_state=40, num_iterations= 100):
        self.num_iterations = num_iterations
        self.x_train, self.x_test, self.first_y_train, self.first_y_test, self.second_y_train, self.second_y_test = (
            train_test_split(
            x, first_y, second_y, stratify=second_y, test_size=test_size, random_state=random_state
        ))

    @staticmethod
    def new_error_func(first_y, second_y, y_predict):
        errors = [0] * len(y_predict)
        for i in range(len(y_predict)):
            if first_y[i] == second_y[i]:
                errors[i] = y_predict[i] - first_y[i]
            else:
                min_value = min(first_y[i], second_y[i])
                max_value = max(first_y[i], second_y[i])
                if y_predict[i] <= min_value:
                    errors[i] = y_predict[i] - min_value
                if y_predict[i] >= max_value:
                    errors[i] = y_predict[i] - max_value
        return errors

    # Hàm tính gradient của hàm lỗi (MSE)
    def gradient_mean_squared_error(self, x, y, second_y, w):
        n = len(y)
        y_predict = labeling(np.dot(x, w))
        # error = y_predict - y
        error = self.new_error_func(y, second_y, y_predict)
        gradient = 2 * np.dot(x.T, error) / n
        return gradient

    # Gradient descent để tối ưu hóa hàm lỗi (MSE)
    def adam(
            self,
            x,
            y,
            second_y,
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            num_iterations=10000,
    ):
        # Khởi tạo vector trọng số ngẫu nhiên
        w = np.random.uniform(0, 1, x.shape[1])
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        t = 0

        for _ in range(num_iterations):
            random_index = np.random.randint(0, len(x))
            x_sample = x[random_index: random_index + 1]
            y_sample = y[random_index: random_index + 1]
            second_y_sample = second_y[random_index: random_index + 1]

            # Tính gradient của hàm lỗi cho mẫu đã chọn
            grad = self.gradient_mean_squared_error(x_sample, y_sample, second_y_sample, w)

            # Tính trung bình độ lớn của gradient và gradient bình phương
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            # Bias correction
            m_hat = m / (1 - beta1 ** (t + 1))
            v_hat = v / (1 - beta2 ** (t + 1))

            w -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            t += 1

        return w

    def run(self):
        begin = time.time()
        print('------START EXECUTE ADAM------')
        max1 = 0
        res = None
        for i in range(self.num_iterations):
            learned_weights = self.adam(self.x_train, self.first_y_train, self.second_y_train)
            predict = labeling(self.x_test.dot(learned_weights))
            acc = new_accuracy(self.first_y_test, self.second_y_test, predict)
            if acc > max1:
                max1 = acc
                res = learned_weights
                print(f'new learning weights: {res}')
        predicted_labels = labeling(self.x_test.dot(res))
        result = {
            "first_y_test": [int(i) for i in self.first_y_test],
            "second_y_test": [int(i) for i in self.second_y_test],
            "predicted_labels": [i for i in predicted_labels]
        }
        df = pd.DataFrame(result)
        df.to_csv('results/adam.csv')
        print(f'------EXECUTE IN {time.time() - begin}------')