#%%
import numpy as np


#%%
def sigmoid(Z):
	A = 1 / (1 + np.exp(-Z))
	return A


#%%
def d_sigmoid(Z):
	A = sigmoid(Z)
	dZ = A * (1 - A)
	return dZ


#%%
def relu(Z):
	A = np.maximum(0, Z)
	return A


#%%
def d_relu(Z):
	dZ = np.ones_like(Z)
	dZ[Z <= 0] = 0
	return dZ


#%%
# needs fixing
def leaky_relu(Z, a=0.01):
	return np.maximum(a * Z, Z), Z  # a pbb required for backprop


def d_leaky_relu(dA, Z, a=0.01):
	dx = np.ones_like(Z)
	dx[Z <= 0] = a
	return dx * dA  # a?


#%%
# definitely correct
def cross_entropy_cost(Y_hat, Y, m):
	cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))

	return cost.item()


#%%
def L2_regularization_sum(lambd, W, m):
	return lambd / (2 * m) * np.sum(np.square(W))


def d_L2_regularization(lambd, m, weights):
	return [lambd / m * W for W in weights]


#%%
def L2_cross_entropy_cost(Y_hat, Y, lambd, weights, m):
	cost = cross_entropy_cost(Y_hat, Y, m) + np.sum([L2_regularization_sum(lambd, W, m) for W in weights])

	return cost


#%%
def transpose(dataframes):
	return [df.T for df in dataframes]


#%%
def L2_norm(vector):
	return np.sqrt(np.sum(np.square(vector)))


