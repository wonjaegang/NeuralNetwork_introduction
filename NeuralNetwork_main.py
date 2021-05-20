import random
import math


class Layer:
    def __init__(self, layerIndex, size, lastLayer):
        self.layerIndex = layerIndex
        self.size = size
        # If it is an input layer
        if not lastLayer:
            # 입력값들로 이루어진 열벡터 A
            self.A = randomList(size, 0, 1)
        # If it not
        else:
            self.lastLayer = lastLayer

            # 레이어 노드들의 편향값들로 이루어진 열벡터 B
            self.B = randomList(size, -1, 1)

            # 이전 레이어와 연결되는 신경망들의 가중치들로 이루어진 (thisLayer size) * (lastLayer size) 크기의 행렬 W
            self.W = [randomList(lastLayer.size, -1, 1) for _ in range(size)]

            # 레이어 노드들의 활성화함수 입력값들로 이루어진 열벡터 Z
            self.Z = self.feedForward()

            # 레이어 노드들의 활성화함수 출력값들로 이루어진 열벡터 A
            self.A = self.activateNode()

            # 비용함수의 노드값들에 대한 편미분값으로 이루어진 열벡터 dC_dA
            self.dC_dA = []

    # Z(n) =  W * A(n-1) + B(n)
    def feedForward(self):
        Z = []
        for nodeIndex in range(self.size):
            temp = 0
            for lastNodeIndex in range(self.lastLayer.size):
                temp += self.W[nodeIndex][lastNodeIndex] * self.lastLayer.A[lastNodeIndex]
            z = temp + self.B[nodeIndex]
            Z.append(z)
        return Z

    # A(n) = k(Z(n))
    # k(x)는 활성화 함수
    def activateNode(self):
        return list(map(lambda x: activationFunc(x), self.Z))

    def backPropagation(self, L1, L2):
        learningRate = 0.5
        pass


# 기타 함수들
def randomList(size, minimum, maximum):
    return [random.uniform(minimum, maximum) for _ in range(size)]


def sigmoid(x):
    return 1 / (1 + math.e ** (-1 * x))


def derivativeSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def activationFunc(x):
    y = sigmoid(x)
    return y


# 입력값의 평균값이 0.5 미만이면 [0]을, 이상이면 [1]을 반환한다.
def expectedOutput():
    if sum(inputLayer.A) < inputLayer.size / 2:
        return [0]
    else:
        return [1]


def calculateAverage(lastAverage, n, an):
    return lastAverage * (n - 1) / n + an / n


# C(y) = 1 / n * sum( (Ys - Y)^2 )
# 비용함수는 오차의 제곱들의 평균값
def costFunc():
    errorAverage = 0
    for index in range(outputLayer.size):
        error = (expectedOutput()[index] - outputLayer.A[index]) ** 2
        errorAverage = calculateAverage(errorAverage, index + 1, error)
    return errorAverage


def terminalDisplay():
    print("Input layer node value:", inputLayer.A)

    print("\nlayer1 node bias:", layer1.B)
    print("layer1 neuron weight:", layer1.W)
    print("layer1 Z:", layer1.Z)
    print("layer1 node value:", layer1.A)

    print("\nlayer2 node bias:", layer2.B)
    print("layer2 neuron weight:", layer2.W)
    print("layer2 Z:", layer2.Z)
    print("layer2 node value:", layer2.A)

    print("\nOutput layer node bias:", outputLayer.B)
    print("Output layer neuron weight:", outputLayer.W)
    print("Output layer Z:", outputLayer.Z)
    print("Output layer node value:", outputLayer.A)

    print(expectedOutput())
    print(costFunc())


if __name__ == "__main__":
    inputLayer = Layer(0, 10, False)
    layer1 = Layer(1, 4, inputLayer)
    layer2 = Layer(2, 4, layer1)
    outputLayer = Layer(3, 1, layer2)

    epoch = 1000
    for _ in range(epoch):
        pass

    terminalDisplay()
