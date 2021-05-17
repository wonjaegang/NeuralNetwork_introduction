import random


def activationFunc(x):
    return x


class Layer:
    def __init__(self, layerIndex, size, lastLayerSize):
        self.layerIndex = layerIndex
        self.size = size
        self.lastLayerSize = lastLayerSize
        self.nodeBias = [random.random() for _ in range(size)]
        self.nodeValue = self.calculateNodeValue()

    def calculateNodeValue(self):
        Z = 0
        for nodeIndex in range(self.size):
            pass
        return activationFunc(Z)


if __name__ == "__main__":
    inputValue = []
    layer1 = Layer(0, 8, len(inputValue))
    layer2 = Layer(1, 8, layer1.size)
    outputLayer = Layer(2, 1, layer2.size)
    print(layer1.nodeBias)
    print(layer2.nodeBias)
    print(outputLayer.nodeBias)
