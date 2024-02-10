from functions.quick_maths import *


#%%
def get_d_strings(act_functions):
	return [f"d_{f}" for f in act_functions]


#%%
def get_act_functions(act_functions, act_dict):
	return [act_dict[x] for x in act_functions]


#%%
def get_weights(params, n_layers):
	return [params[f"W{i}"] for i in range(1, n_layers)]


#%%
def get_shapes(dataframes) -> list:
	return [df.shape for df in dataframes]


#%%
def dict_to_vector(dict_Wb, filter_W, filter_b, L):
	vector = []

	for i in range(1, L):
		vector.append(dict_Wb[f"{filter_W}{i}"].reshape(-1, 1))
		vector.append(dict_Wb[f"{filter_b}{i}"].reshape(-1, 1))

	return np.concatenate(vector)


#%%
def vector_to_dict(vector, filler_W, filler_b, layer_dims):
	parameters = {}
	end = 0

	for l in range(1, len(layer_dims)):
		w_size = layer_dims[l] * layer_dims[l-1]
		parameters[f"{filler_W}{l}"] = vector[end:end+w_size].reshape(layer_dims[l], layer_dims[l-1])
		end += w_size
		b_size = layer_dims[l]
		parameters[f"{filler_b}{l}"] = vector[end:end+b_size].reshape(layer_dims[l], 1)
		end += b_size

	return parameters


#%%
def fancy_print(message, colour):
	colour_dict = {
		'red':91,
		'green':92,
		'yellow':93,
		'blue':94,
		'pink':95,
		'cyan':96,
		'white':97
	}
	print(f"\033[{colour_dict[colour]}{message}\033[0m")

