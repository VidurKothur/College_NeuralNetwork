from networks import Network
from layers import Layer
from variables import Parameter
from initializations import XavierNormal
from regularizations import L1
from optimizers import SGD
from dissipations import TotalAbsoluteError
from data import Tensor
import numpy as np
import functions as fs
import tensorflow as tf

#np.set_printoptions(threshold=np.inf)

def dense(vector, consts, learns):
    return fs.ReLU(vector @ learns[0].value + learns[1].value)

lr = 0.001
xC = 2e-6

l1 = Layer("Layer1", None)
l1.addParameterVariable(Parameter("W1", "W1G", L1(0.1), 1, XavierNormal((784, 16), xConstant=xC), SGD(lr)))
l1.addParameterVariable(Parameter("B1", "B1G", L1(0.1), 1, XavierNormal((1, 16), xConstant=xC), SGD(lr)))
l1.initializeVariables()
l1.setExecutionFunction(dense)

l2 = Layer("Layer2", None)
l2.addParameterVariable(Parameter("W2", "W2G", L1(0.1), 1, XavierNormal((16, 1), xConstant=xC), SGD(lr)))
l2.addParameterVariable(Parameter("B2", "B2G", L1(0.1), 1, XavierNormal((1, 1), xConstant=xC), SGD(lr)))
l2.initializeVariables()
l2.setExecutionFunction(dense)

net = Network()

net.addLayer(l1)
net.addLayer(l2)

def loss(dissipation, regularization):
    return dissipation

net.setDissipationFunction(TotalAbsoluteError)
net.setLossFunction(loss)

def network(data, layerFunctions, dissipationFunction, regularizationFunction, lossFunction):
    vector = Tensor(data[0])
    for func in layerFunctions:
        vector = func(vector)
    return lossFunction(dissipationFunction(vector, data[1]), regularizationFunction())

net.setNetworkFunction(network)
net.debug(dissipationDebugKey="Dissipation", regularizationDebugKey="Regularization", lossDebugKey="Loss")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, y_train = x_train[:10], y_train[:10]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
train_data = (tuple(map(tuple, x_train)), tuple(y_train))
test_data = (tuple(map(tuple, x_test)), tuple(y_test))

net.train(train_data, 100, 1)