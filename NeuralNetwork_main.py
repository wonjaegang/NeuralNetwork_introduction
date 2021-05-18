import random


def activationFunc(x):
    y = x
    return y


class Layer:
    def __init__(self, layerIndex, size, lastLayer):
        self.layerIndex = layerIndex
        self.size = size
        # If it is an input layer
        if not lastLayer:
            self.nodeValue = randomList(size)
        # If it not
        else:
            self.lastLayer = lastLayer
            self.nodeBias = randomList(size)
            # 이전 레이어와 연결되는 신경망들의 가중치
            self.neuronWeight = [randomList(lastLayer.size) for _ in range(size)]
            self.nodeValue = self.calculateNodeValue()

    def calculateNodeValue(self):
        valueList = []
        for nodeIndex in range(self.size):
            value = 0
            for lastNodeIndex in range(self.lastLayer.size):
                value = value + self.lastLayer.nodeValue[lastNodeIndex] * self.neuronWeight[nodeIndex][lastNodeIndex]
            value = activationFunc(value + self.nodeBias[nodeIndex])
            valueList.append(value)
        return valueList


def randomList(size):
    return [random.random() for _ in range(size)]


def expectedOutput():
    if sum(inputLayer.nodeValue) > inputLayer.size / 2:
        return 1
    else:
        return 0


if __name__ == "__main__":
    inputLayer = Layer(0, 10, False)
    layer1 = Layer(1, 8, inputLayer)
    layer2 = Layer(2, 8, layer1)
    outputLayer = Layer(3, 1, layer2)

    print("Input layer node value:", inputLayer.nodeValue)

    print("\nlayer1 node bias:", layer1.nodeBias)
    print("layer1 neuron weight:", layer1.neuronWeight)
    print("layer1 node value:", layer1.nodeValue)

    print("\nlayer2 node bias:", layer2.nodeBias)
    print("layer2 neuron weight:", layer2.neuronWeight)
    print("layer2 node value:", layer2.nodeValue)

    print("\nOutput layer node bias:", outputLayer.nodeBias)
    print("Output layer neuron weight:", outputLayer.neuronWeight)
    print("Output layer node value:", outputLayer.nodeValue)
