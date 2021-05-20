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

    # Z(n) =  W(n, n-1) * A(n-1) + B(n)
    def feedForward(self):
        Z = []
        for nodeIndex in range(self.size):
            temp = 0
            for lastNodeIndex in range(self.lastLayer.size):
                temp += self.W[nodeIndex][lastNodeIndex] * self.lastLayer.A[lastNodeIndex]
            z = temp + self.B[nodeIndex]
            Z.append(z)
        return Z

    # A(n) = k(Z(n)), k(x)는 활성화 함수
    def activateNode(self):
        return list(map(lambda x: k(x), self.Z))

    # dC/dB = dC/dA * dA/dB = dC/dA * k'(Z)
    # dC/dW(i) = dC/dA * dA/dW = dC/dA * k'(Z) * lastLayer.A(i)
    # dC/d(lastLayer.A) = dC/dA * dA/d(lastLayer.A) = dC/dA * k'(Z) * W
    def feedBackward(self):
        learningRate = 0.5
        pass


# 입력 데이터 파일을 생성
def createInputDataFile():
    dataSize = 10000
    with open("InputData", 'w') as f:
        for _ in range(dataSize):
            inputList = randomList(inputLayer.size, 0, 1)
            for inputNum in inputList:
                f.write("%f " % inputNum)
            f.write(",")
            outputList = expectedOutput(inputList)
            for outputNum in outputList:
                f.write("%f " % outputNum)
            f.write("\n")


def readData(index):
    with open("InputData", 'r') as f:
        for i, line in enumerate(f):
            if i == index:
                split = line.split(',')
                inputData = list(map(lambda x: float(x), split[0].split(' ')[:-1]))
                outputData = list(map(lambda x: float(x), split[1].split(' ')[:-1]))
                return inputData, outputData


# 활성화 함수(Activation function)
def k(x):
    y = sigmoid(x)
    return y


# 활성화 함수의 도함수
def dk(x):
    y = derivativeSigmoid(x)
    return y


# 입력값의 평균값이 0.5 미만이면 [0]을, 이상이면 [1]을 반환한다.
def expectedOutput(inputList):
    if sum(inputList) < inputLayer.size / 2:
        return [0]
    else:
        return [1]


# C(y) = 1 / n * sum( (setPoint - Y)^2 )
# 비용함수는 오차의 제곱들의 평균값
def costFunc():
    errorAverage = 0
    for index in range(outputLayer.size):
        error = (expectedOutput(inputLayer.A)[index] - outputLayer.A[index]) ** 2
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

    print(expectedOutput(inputLayer.A))
    print(costFunc())


# 기타 함수 모음
def randomList(size, minimum, maximum):
    return [random.uniform(minimum, maximum) for _ in range(size)]


def calculateAverage(lastAverage, n, an):
    return lastAverage * (n - 1) / n + an / n


def sigmoid(x):
    return 1 / (1 + math.e ** (-1 * x))


def derivativeSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


if __name__ == "__main__":
    inputLayer = Layer(0, 10, False)
    layer1 = Layer(1, 4, inputLayer)
    layer2 = Layer(2, 4, layer1)
    outputLayer = Layer(3, 1, layer2)

    epoch = 1000
    for _ in range(epoch):
        pass

    terminalDisplay()
    createInputDataFile()
    print(readData(0))
