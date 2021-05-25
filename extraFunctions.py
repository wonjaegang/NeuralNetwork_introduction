import random
import math


def randomList(size, minimum, maximum):
    return [random.uniform(minimum, maximum) for _ in range(size)]


def calculateAverage(lastAverage, n, an):
    return lastAverage * (n - 1) / n + an / n


def analog2digital(analog):
    if analog >= 0.5:
        return 1
    else:
        return 0


def sigmoid(x):
    return 1 / (1 + math.e ** (-1 * x))


def derivativeSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))