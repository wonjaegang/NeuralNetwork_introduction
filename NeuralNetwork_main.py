import random
import math


class Layer:
    def __init__(self, layerIndex, size, lastLayer):
        self.layerIndex = layerIndex
        self.size = size
        # If it is an input layer
        if not lastLayer:
            self.nodeValue = randomList(size, 0, 1)
        # If it not
        else:
            self.lastLayer = lastLayer
            self.nodeBias = randomList(size, -2, 2)

            # 이전 레이어와 연결되는 신경망들의 가중치
            self.neuronWeight = [randomList(lastLayer.size, -2, 2) for _ in range(size)]

            # 레이어의 각 노드들의 z(활성화함수 입력값)와 실제 노드의 값(활성화함수 출력값)을 구한다
            self.Z = self.feedForward()
            self.nodeValue = self.activateNode()

    def feedForward(self):
        Z = []
        for nodeIndex in range(self.size):
            a = 0
            for lastNodeIndex in range(self.lastLayer.size):
                a += self.lastLayer.nodeValue[lastNodeIndex] * self.neuronWeight[nodeIndex][lastNodeIndex]
            z = a + self.nodeBias[nodeIndex]
            Z.append(z)
        return Z

    def activateNode(self):
        return list(map(lambda x: activationFunc(x), self.Z))


def backPropagation(L1, L2):
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


# 입력값의 평균값이 0.5 미만이면 0을, 이상이면 1을 반환한다.
def expectedOutput():
    if sum(inputLayer.nodeValue) < inputLayer.size / 2:
        return 0
    else:
        return 1


def costFunc():
    y = (expectedOutput() - outputLayer.nodeValue[0]) ** 2
    return y


def terminalDisplay():
    print("Input layer node value:", inputLayer.nodeValue)

    print("\nlayer1 node bias:", layer1.nodeBias)
    print("layer1 neuron weight:", layer1.neuronWeight)
    print("layer1 Z:", layer1.Z)
    print("layer1 node value:", layer1.nodeValue)

    print("\nlayer2 node bias:", layer2.nodeBias)
    print("layer2 neuron weight:", layer2.neuronWeight)
    print("layer2 Z:", layer2.Z)
    print("layer2 node value:", layer2.nodeValue)

    print("\nOutput layer node bias:", outputLayer.nodeBias)
    print("Output layer neuron weight:", outputLayer.neuronWeight)
    print("Output layer Z:", outputLayer.Z)
    print("Output layer node value:", outputLayer.nodeValue)

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
