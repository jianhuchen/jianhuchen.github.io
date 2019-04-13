#coding=utf-8
from Network import Network
from mnist_loader import *

model = Network([784, 20, 20, 10])
train, val, test = load_data_wrapper('./data/mnist.pkl.gz')
model.SGD(train, 50, 2000, 0.8, val)
