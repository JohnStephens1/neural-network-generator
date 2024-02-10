import PIL.Image
import numpy as np
import cv2
import random
from os import listdir
from functions.util import transpose, get_shapes


#%%
def prepare_data(src_dir="Resources/train/", dst_dir="my_resources/"):
	df, labels = [], []
	for file in listdir(src_dir):
		output = 0
		if file.startswith('cat'):
			output = 1
		image = PIL.Image.open(f"{src_dir}{file}").convert(mode="L")
		image_array = np.array(image)
		resized = cv2.resize(image_array, dsize=(100, 100))
		flattened = resized.reshape(-1).T/255

		df.append(flattened)
		labels.append(output)
	df = np.asarray(df)
	labels = np.asarray(labels)
	labels = labels.reshape(labels.shape[0], 1)
	print(df.shape, labels.shape)
	np.save(f'{dst_dir}df.npy', df)
	np.save(f'{dst_dir}df_labels.npy', labels)


#%%
def train_test_split(X, Y, m_axis, test_size=0.3):
	m = X.shape[m_axis]

	indices = [x for x in range(m)]

	train_indices, test_indices = extract_indices(indices, int(m * test_size))

	X_train, Y_train, X_test, Y_test = get_X_Y_set(X, Y, [train_indices, test_indices])

	return X_train, Y_train, X_test, Y_test


#%%
def get_X_Y_set(X, Y, indices):
	sets = []
	for i in indices:
		sets.append(X[:, i])
		sets.append(Y[:, i])
	return sets


#%%
def train_val_test_split(X, Y, m_axis, val_size=0.2, test_size=0.1):
	m = X.shape[m_axis]

	indices = [x for x in range(m)]
	indices, val_indices = extract_indices(indices, int(val_size * m))
	train_indices, test_indices = extract_indices(indices, int(test_size * m))

	# for clarity purposes a little verbose
	X_train, Y_train, X_val, Y_val, X_test, Y_test = get_X_Y_set(X, Y, [train_indices, val_indices, test_indices])

	return X_train, Y_train, X_val, Y_val, X_test, Y_test


#%%
def extract_indices(indices, n_to_extract):
	# random.sample requires a list, so all the casting has to look a bit messy...
	extracted = set(random.sample(indices, n_to_extract))
	remaining = list(set(indices) - extracted)

	return remaining, list(extracted)


#%%
def load_data():
	df = np.load("my_resources/df.npy")
	df_labels = np.load("my_resources/df_labels.npy")
	return df, df_labels


#%%
def get_aligned_dfs(X, Y, m_axis):
	daa = m_axis
	inv_daa = 1 - m_axis
	X_shape, Y_shape = get_shapes([X, Y])

	if X_shape[daa] == Y_shape[daa]:
		return X, Y
	elif X_shape[inv_daa] == Y_shape[inv_daa]:
		return transpose([X, Y])
	elif X_shape[inv_daa] == Y_shape[daa]:
		return *transpose([X]), Y
	elif X_shape[daa] == Y_shape[inv_daa]:
		return X, *transpose([Y])
	else:
		raise ValueError("Dimensions of provided datasets are incompatible!")


#%%
def prepare_dataframes(X, Y, m_axis):
	X, Y = get_aligned_dfs(X, Y, m_axis)
	X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(X, Y, m_axis)

	return X_train, Y_train, X_val, Y_val, X_test, Y_test


#%%
def get_reduced_dataframes(X, Y, m, m_axis):
	samples = random.sample(range(X.shape[m_axis]), m)
	if m_axis == 1:
		return X[:, samples], Y[:, samples]
	elif m_axis == 0:
		return X[samples, :], Y[samples, :]
	else:
		raise ValueError('Invalid axis value provided! Stopping execution...')
