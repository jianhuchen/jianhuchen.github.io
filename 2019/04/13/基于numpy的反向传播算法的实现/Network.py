#coding=utf-8
import random
import numpy as np

class Network(object):
	def __init__(self, sizes):
		self.sizes = sizes
		self.num_layers = len(sizes)
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

	def sigmoid(self, z):
		'''sigmoid函数
		'''
		return 1.0 / (1.0 + np.exp(-z))

	def sigmoid_prime(self, z):
		'''求sigmoid函数的一阶倒数
		'''
		return self.sigmoid(z) * (1.0 - self.sigmoid(z))

	def feedforward(self, a):
		'''前向传播，得到预测结果并返回
		'''
		for i in range(self.num_layers - 1):
			z = np.dot(self.weights[i], a) + self.biases[i]
			a = self.sigmoid(z)
		return a

	def SGD(self, 
			training_data, 
			epochs, 
			mini_batch_size, 
			eta, 
			test_data=None):
		'''随机梯度下降算法
		eta: 学习率
		'''
		# print(training_data[0][0], training_data[0][1])
		n = len(training_data)
		for i in range(epochs):
			# 随机打乱训练集
			random.shuffle(training_data)
			# 使用传入的数据构造mini_batch
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)
			]
			# 使用一个mini_batch更新参数
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			# 如果有测试集的话，每使用一个mini_batch更新完参数
			# 就在测试集上验证一下正确率
			if test_data:
				print('Epoch {}: accuracy = {}'.format(i, self.accuracy(test_data)))
			else:
				print('Epoch {}: complete!'.format(i))

	def update_mini_batch(self, mini_batch, eta):
		'''用一个mini_batch来更新参数
		'''
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		for x, y in mini_batch:
			delta_nabla_w, delta_nabla_b = self.backprop(x, y)
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
		self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		'''反向传播算法，计算梯度
		'''
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		# feedfoward 前向
		activation = x # 第一层的输入
		activations = [x] # 记录每一层的输入，相当于z对w的梯度
		zs = [] # 经过激活函数前的加权求和值
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = self.sigmoid(z)
			activations.append(activation)
		# backward pass 反向
		# 先算最后一层
		delta = (activations[-1]-y)*self.sigmoid_prime(zs[-1])
		# print(delta.shape)
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		nabla_b[-1] = delta
		# 再算前面几层，这里是从后往前计算的
		for layer in range(2, self.num_layers):
			z = zs[-layer]
			sp = self.sigmoid_prime(z)
			delta = sp * np.dot(self.weights[-layer+1].transpose(), delta)
			# print(delta.shape, activations[-layer-1].shape)
			nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
			nabla_b[-layer] = delta
		return nabla_w, nabla_b
		
	def accuracy(self, test_data):
		'''计算测试数据的正确率
		'''
		n_test = len(test_data)
		# print(test_data[0][0], test_data[0][1])
		test_results = [(np.argmax(self.feedforward(x)), y_) for x, y_ in test_data]
		return sum([int(x == y) for x, y in test_results]) * 1.0 / n_test 

