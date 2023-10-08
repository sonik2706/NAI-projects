import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

from dataclasses import dataclass


@dataclass
class Vector:
    n_columns: int
    value: str
    data: list

    def __init__(self, data):
        self.n_columns = len(data) - 1
        self.value = data[-1].strip()
        self.data = list(map(float, data[:-1]))

    def __str__(self):
        return f"{self.data}"


def distance(vector_1: Vector, vector_2: Vector) -> float:
    sum = 0
    for i in range(vector_1.n_columns):
        sum += (vector_1.data[i] - vector_2.data[i]) ** 2

    return sqrt(sum)


def getClass(vector: Vector, data: list, k: int) -> str:
    distances = list(map(lambda x: distance(vector, x), data))

    idx = np.argpartition(distances, k)[:k]

    result = {data[index].value: 0 for index in idx}

    for index in idx:
        result[data[index].value] += 1

    return max(result)


if __name__ == "__main__":
    with open("iris.data") as data, open("iris.test.data") as testData:
        dt = []
        for line in data:
            tmp = Vector(line.split(","))
            dt.append(tmp)

        test_dt = []
        for line in testData:
            tmp = Vector(line.split(","))
            test_dt.append(tmp)

        accuracy = []

        n = int(input("k:"))
        for k in range(1, n + 1):
            counter = 0
            for vector in test_dt:
                if vector.value == getClass(vector, dt, k):
                    counter += 1

            print(f"for k = {k} -> {counter/len(test_dt)}")
            accuracy.append(counter / len(test_dt))

    plt.plot([n for n in range(1, n + 1)], accuracy)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()
