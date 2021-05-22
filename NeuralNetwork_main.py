import random
import math


class Layer:
    def __init__(self, layerIndex, size, lastLayer):
        self.layerIndex = layerIndex
        self.size = size
        # If it is an input layer
        if not lastLayer:
            # 입력값들로 이루어진 열벡터 A
            self.A = []
        # If it is not
        else:
            self.lastLayer = lastLayer

            # 레이어 노드들의 편향값들로 이루어진 열벡터 B
            self.B = randomList(size, -1, 1)

            # 이전 레이어와 연결되는 신경망들의 가중치들로 이루어진 (thisLayer size) * (lastLayer size) 크기의 행렬 W
            self.W = [randomList(lastLayer.size, -1, 1) for _ in range(size)]

            # 레이어 노드들의 활성화함수 입력값들로 이루어진 열벡터 Z
            self.Z = [0 for _ in range(size)]

            # 레이어 노드들의 활성화함수 출력값들로 이루어진 열벡터 A
            self.A = [0 for _ in range(size)]

            # 비용함수의 노드값들에 대한 편미분값으로 이루어진 열벡터 dC_dA
            self.dC_dA = [0 for _ in range(size)]

            # 비용함수의 가중치값들에 대한 편미분값으로 이루어진 (thisLayer size) * (lastLayer size) 크기의 행렬 dC_dA
            self.dC_dW = [[0 for _ in range(lastLayer.size)] for _ in range(size)]

            # 비용함수의 편향값들에 대한 편미분값으로 이루어진 열벡터 dC_dA
            self.dC_dB = [0 for _ in range(size)]

    # Z(n) =  W(n, n-1) * A(n-1) + B(n)
    # A(n) = k(Z(n)), k(x)는 활성화 함수
    def feedForward(self):
        Z = []
        for nodeIndex in range(self.size):
            temp = 0
            for lastNodeIndex in range(self.lastLayer.size):
                temp += self.W[nodeIndex][lastNodeIndex] * self.lastLayer.A[lastNodeIndex]
            z = temp + self.B[nodeIndex]
            Z.append(z)
        A = list(map(lambda x: k(x), Z))
        self.Z = Z
        self.A = A

    def feedBackward(self):
        # dC/dW(k) = dC/dA * dA/dW(k) = dC/dA * k'(Z) * lastLayer.A[k]
        dC_dW = []
        for nodeIndex in range(self.size):
            dC_dW_k = []
            for lastNodeIndex in range(self.lastLayer.size):
                temp = self.dC_dA[nodeIndex] * dk(self.Z[nodeIndex]) * self.lastLayer.A[lastNodeIndex]
                average = calculateAverage(self.dC_dW[nodeIndex][lastNodeIndex], dataIndex + 1, temp)
                dC_dW_k.append(average)
            dC_dW.append(dC_dW_k)
        self.dC_dW = dC_dW

        # dC/dB = dC/dA * dA/dB = dC/dA * k'(Z)
        dC_dB = []
        for nodeIndex in range(self.size):
            temp = self.dC_dA[nodeIndex] * dk(self.Z[nodeIndex])
            average = calculateAverage(self.dC_dB[nodeIndex], dataIndex + 1, temp)
            dC_dB.append(average)
        self.dC_dB = dC_dB

        # dC/d(lastLayer.A) = dC/dA * dA/d(lastLayer.A) = dC/dA * k'(Z) * W
        last_dC_dA = []
        for lastNodeIndex in range(self.lastLayer.size):
            temp = 0
            for nodeIndex in range(self.size):
                temp += self.dC_dA[nodeIndex] * dk(self.Z[nodeIndex]) * self.W[nodeIndex][lastNodeIndex]
            last_dC_dA.append(temp)
        self.lastLayer.dC_dA = last_dC_dA

    # updated X = X - learningRate * dC/dX
    def updateNeurons(self):
        for nodeIndex in range(self.size):
            for lastNodeIndex in range(self.lastLayer.size):
                self.W[nodeIndex][lastNodeIndex] -= setting.learningRate * self.dC_dW[nodeIndex][lastNodeIndex]

        for nodeIndex in range(self.size):
            self.B[nodeIndex] -= setting.learningRate * self.dC_dB[nodeIndex]


class Setting:
    def __init__(self):
        self.epoch = 1000
        self.inputDataSize = 1000
        self.evaluationDataSize = 100
        self.learningRate = 2


# 하나의 입력-출력 데이터 셋을 파일에 입력
def createDataSet(file):
    inputList = randomList(inputLayer.size, 0, 1)
    for inputNum in inputList:
        file.write("%f " % inputNum)
    file.write(",")
    outputList = expectedOutput(inputList)
    for outputNum in outputList:
        file.write("%f " % outputNum)
    file.write("\n")


# 입력 데이터 파일을 생성
def createInputDataFile():
    with open("InputData", 'w') as f:
        for _ in range(setting.inputDataSize):
            createDataSet(f)


# 평가용 데이터 파일을 생성
def createEvaluationDataFile():
    with open("EvaluationData", 'w') as f:
        for _ in range(setting.evaluationDataSize):
            createDataSet(f)


# 데이터셋 파일로부터 입력/출력 리스트 추출
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


# C(y) = 1/n * sum( (setPoint - Y)^2 )
# 비용함수는 오차의 제곱들의 평균값
def costFunc():
    errorAverage = 0
    for index in range(outputLayer.size):
        error = (expectedOutput(inputLayer.A)[index] - outputLayer.A[index]) ** 2
        errorAverage = calculateAverage(errorAverage, index + 1, error)
    return errorAverage


# dC/dY = -2/n * (setPoint - Y)
# 출력 레이어에서의 dC/dA는 비용함수의 출력값 Y에 대한 편미분 값
def get_dC_dY(setPoint):
    dC_dY = []
    for nodeIndex in range(outputLayer.size):
        temp = -2 / outputLayer.size * (setPoint[nodeIndex] - outputLayer.A[nodeIndex])
        dC_dY.append(temp)
    return dC_dY


def terminalDisplay():
    print("Input layer node value:", inputLayer.A)

    print("\nlayer1 B:", layer1.B)
    print("layer1 W:", layer1.W)
    print("layer1 Z:", layer1.Z)
    print("layer1 A:", layer1.A)
    print("layer1 dC/dA:", layer1.dC_dA)
    print("layer1 dC/dW:", layer1.dC_dW)
    print("layer1 dC/dB:", layer1.dC_dB)

    print("\nlayer2 B:", layer2.B)
    print("layer2 W:", layer2.W)
    print("layer2 Z:", layer2.Z)
    print("layer2 A:", layer2.A)
    print("layer2 dC/dA:", layer2.dC_dA)
    print("layer2 dC/dW:", layer2.dC_dW)
    print("layer2 dC/dB:", layer2.dC_dB)

    print("\nOutput B:", outputLayer.B)
    print("Output W:", outputLayer.W)
    print("Output Z:", outputLayer.Z)
    print("Output A:", outputLayer.A)
    print("Output dC/dA:", outputLayer.dC_dA)
    print("Output dC/dW:", outputLayer.dC_dW)
    print("Output dC/dB:", outputLayer.dC_dB)

    print("\nExpected output:", expectedOutput(inputLayer.A))
    print("Cost:", costFunc())


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
    setting = Setting()

    inputLayer = Layer(0, 10, False)
    layer1 = Layer(1, 4, inputLayer)
    layer2 = Layer(2, 4, layer1)
    outputLayer = Layer(3, 1, layer2)

    # createInputDataFile()
    # createEvaluationDataFile()

    for epochIndex in range(setting.epoch):
        for dataIndex in range(setting.inputDataSize):
            # 데이터 셋에서 이번 인덱스의 데이터 추출
            inputDataList = readData(dataIndex)[0]
            outputDataList = readData(dataIndex)[1]

            # 입력 데이터로 노드값 계산
            inputLayer.A = inputDataList
            layer1.feedForward()
            layer2.feedForward()
            outputLayer.feedForward()

            # 오차역전파
            outputLayer.dC_dA = get_dC_dY(outputDataList)
            outputLayer.feedBackward()
            layer2.feedBackward()
            layer1.feedBackward()

            # print("Data set #%d" % dataIndex)

        # 가중치와 편향값의 gradient 방향 조정
        layer1.updateNeurons()
        layer2.updateNeurons()
        outputLayer.updateNeurons()

        # 테스트 데이터 셋으로 신경망 평가
        averageCost = 0
        for dataIndex in range(setting.evaluationDataSize):
            inputDataList = readData(dataIndex)[0]
            outputDataList = readData(dataIndex)[1]

            inputLayer.A = inputDataList
            layer1.feedForward()
            layer2.feedForward()
            outputLayer.feedForward()
            averageCost = calculateAverage(averageCost, dataIndex + 1, costFunc())

        print("\nEpoch #%d Average Cost:" % (epochIndex + 1), averageCost)
