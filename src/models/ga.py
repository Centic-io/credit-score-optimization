import json
import random
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.evaluate import normal


class GeneticAlgorithm:
    def __init__(
        self, x, first_y, second_y, num_params, num_solutions=100, mutation_rate=0.1, num_generations = 100,
            test_size=0.2, random_state=40
    ):
        self.num_generations = num_generations
        self.x = x
        self.first_y = first_y
        self.second_y = second_y
        self.num_params = num_params
        self.num_solutions = num_solutions
        self.mutation_rate = mutation_rate
        (
            self.x_train,
            self.x_test,
            self.first_y_train,
            self.first_y_test,
            self.second_y_train,
            self.second_y_test,
        ) = train_test_split(
            x, first_y, second_y, test_size=test_size, stratify=second_y, random_state=random_state
        )

    def initialize_solutions(self):
        solutions = []
        for _ in range(self.num_solutions):
            # Sử dụng np.random.uniform để khởi tạo giá trị trong khoảng từ -500 đến 500
            solution = np.random.uniform(0, 1, size=self.num_params)
            solutions.append(tuple(normal(solution)))
        return solutions

    def mutate(self, child):
        mutated_child = tuple(
            [
                gene * np.random.uniform(1 - self.mutation_rate, 1 + self.mutation_rate)
                for gene in child
            ]
        )
        return mutated_child

    @staticmethod
    def crossover(parent1, parent2, cross_type=None):
        if cross_type is None or cross_type == "one_point":
            crossover_point = np.random.randint(1, len(parent1) - 1)
            child = list(parent1[:crossover_point]) + list(parent2[crossover_point:])
            return tuple(child)
        elif cross_type == "two_point":
            crossover_point1, crossover_point2 = np.sort(
                np.random.choice(range(1, len(parent1)), 2, replace=False)
            )
            child = (
                list(parent1[:crossover_point1])
                + list(parent2[crossover_point1:crossover_point2])
                + list(parent1[crossover_point2:])
            )
            return tuple(normal(child))
        elif cross_type == "mean":
            child = [(parent1[i] + parent2[i]) / 2 for i in range(len(parent1))]
            return tuple(child)

    def evolve(self, solutions, crossover=None):
        ranked_solutions = [(self.fitness(theta), theta) for theta in solutions]
        ranked_solutions = sorted(ranked_solutions, key=lambda x: x[0], reverse=True)
        print(f"fitness:{ranked_solutions[0][0]}")
        best_solutions = ranked_solutions[:20] + ranked_solutions[-5:]

        new_solution = [ranked_solutions[0][1]]

        for _ in range(self.num_solutions - 1):
            parent1, parent2 = (
                random.choice(best_solutions)[1],
                random.choice(best_solutions)[1],
            )
            child1 = self.crossover(parent1, parent2, crossover)
            child1 = self.mutate(child1)
            new_solution.append(normal(child1))
        return new_solution, ranked_solutions[0][0]

    # support function
    def fitness(self, theta):
        y_predict = self.predict(self.x_train, theta)
        return self.accuracy_score(self.first_y_train, self.second_y_train, y_predict)

    @staticmethod
    def accuracy_score(y_train, second_y_train, y_predict):
        y_train = np.array(y_train)
        second_y_train = np.array(second_y_train)
        y_predict = np.array(y_predict)
        condition = np.logical_or(y_predict == second_y_train, y_predict == y_train)
        count = np.sum(condition)
        accuracy = count / len(y_train)
        return accuracy

    @staticmethod
    def predict(matrices, theta):
        list_scores = np.round(np.dot(matrices, theta)).astype(int)
        label = []
        for score in list_scores:
            if score < 580:
                label.append(0)
            elif 580 <= score < 670:
                label.append(1)
            elif 670 <= score < 740:
                label.append(2)
            elif 740 <= score < 800:
                label.append(3)
            elif 800 <= score <= 850:
                label.append(4)
        return np.array(label)

    def run(self, accuracy=None):
        begin = time.time()
        print('------START EXECUTE GA------')
        solution = self.initialize_solutions()
        for _ in range(self.num_generations):
            solution, acc = self.evolve(solution, crossover="two_point")
            if accuracy and acc > accuracy:
                break
        predicted_labels = self.predict(self.x_test, solution[0])
        result = {
            "first_y_test": [i for i in self.first_y_test],
            "second_y_test": [i for i in self.second_y_test],
            "predicted_labels": [i for i in predicted_labels]
        }
        df = pd.DataFrame(result)
        df.to_csv('results/ga.csv')
        print(f'------EXECUTE IN {time.time() - begin}------')

