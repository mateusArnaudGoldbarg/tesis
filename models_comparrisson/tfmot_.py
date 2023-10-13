import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import time
from memory_profiler import profile
import sys

model = tf.keras.models.load_model('model.h5')
sparsities = []
names = []
model_mask = []

@profile
def normal_predict(model, data):
	a = model.predict(data)
	return a

@profile
def pruned_predict(model, data):
	a = model.predict(data)
	return a
	
for layer in model.trainable_weights:
	#print(layer.name)
	flat_array = np.array((tf.reshape(layer,[-1])))
	#print(type(flat_array[0]))
	n_zeros = np.count_nonzero(flat_array == 0.0)
	#print(n_zeros)
	
	#print(flat_array.shape[0])
	sparsity = n_zeros/flat_array.shape[0]
	sparsities.append(sparsity)
	
	mask = tf.cast(tf.where(tf.abs(layer)>0,1,0), tf.float32)
	model_mask.append(mask)

for layer in model.layers:
	if ('conv' in layer.name) or ('dense' in layer.name):
		names.append(layer.name)

	
for i in range(len(names)):
	print(names[i], sparsities[i*2])

pruning_params = {
	names[0]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[0], begin_step=0)
	},
	names[1]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[2], begin_step=0)
	},
	names[2]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[4], begin_step=0)
	},
	names[3]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[6], begin_step=0)
	},
	names[4]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[8], begin_step=0)
	},
	names[5]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[10], begin_step=0)
	},
	names[6]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[12], begin_step=0)
	}
}

for i in model_mask:
	print(i.shape)

pruning_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

print(len(pruning_model.layers))

m = 0
for layer in pruning_model.non_trainable_variables:
	if('/mask' in layer.name):
		layer.assign(model_mask[m])
		print(layer.name)
		print(layer.shape, model_mask[m].shape)
		print()
		m+=2

for layer in pruning_model.non_trainable_variables:
	print(layer)

pruning_model.save('p_model.h5')
	
model.summary()
#pruning_model.summary()

print("model loaded!")

#IMPORTAÇÃO E NORRMALIZAÇÃO
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
#x_train = x_train.reshape(-1, 28, 28, 1)
#x_test = x_test.reshape(-1, 28, 28, 1)

x_train = x_train.astype(float)/255
x_test = x_test.astype(float)/255

y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 10)

'''
start = time.time()
pruning_model.predict(x_test[0:64])
end1 = time.time()
model.predict(x_test[0:64])
end2 = time.time()
'''
start = time.time()
normal_predict(model, x_test[0:64])
end1 = time.time()
pruned_predict(pruning_model, x_test[0:64])
end2 = time.time()

normal_size = sys.getsizeof(model)
pruning_size = sys.getsizeof(pruning_model)

print('timeflow of normal model', end1-start)
print('timeflow of pruned model', end2-end1)
print('footprint of normal model', normal_size, 'bytes')
print('footprint of pruned model', pruning_size, 'bytes')
