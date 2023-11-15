import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import time
import numpy as np
import tensorflow.lite as tflite
import zipfile
import sys
import psutil

def count_params(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    total_params = 0
    for i in range(interpreter.tensor_details_count):
        info = interpreter.get_tensor_details(i)
        shape = info['shape']
        
        num_params = 1
        for dim in shape:
            num_params *= dim

        total_params += num_params

    return total_params

def inference(interpreter, data):
	input_tensor = interpreter.get_input_details()[0]['index']
	output_tensor = interpreter.get_output_details()[0]['index']
	start_time = time.time()

	#for sample in dataset:
	interpreter.set_tensor(input_tensor, data)
	interpreter.invoke()

	end_time = time.time()
	elapsed_time = end_time - start_time
	output_data = interpreter.get_tensor(output_tensor)
	#print(output_data)

	return output_data, elapsed_time


model = tf.keras.models.load_model('model_32_100.h5')


sparsities = []
names = []
model_mask = []
ss = []
for i,layer in enumerate(model.trainable_weights):
	#print(layer.name)
	flat_array = np.array((tf.reshape(layer,[-1])))
	#print(type(flat_array[0]))
	n_zeros = np.count_nonzero(flat_array == 0.0)
	#print(n_zeros)
	
	#print(flat_array.shape[0])
	sparsity = n_zeros/flat_array.shape[0]
	sparsities.append(sparsity)
	
	if "kernel" in layer.name:
		ss.append(sparsity)

	mask = tf.cast(tf.where(tf.abs(layer)>0,1,0), tf.float32)
	if("conv" in layer.name) or("dense" in layer.name):
		model_mask.append(mask)
	
	
	#print(i,"-",layer.name,sparsity)


for layer in model.layers:
	
	if ('conv' in layer.name) or ('dense' in layer.name):
		#print(layer.name)
		names.append(layer.name)


pruning_params = {
	names[0]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[0], begin_step=0)
	},
	names[1]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[1], begin_step=0)
	},
	names[2]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[2], begin_step=0)
	},
	names[3]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[3], begin_step=0)
	},
	names[4]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[4], begin_step=0)
	},
	names[5]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[5], begin_step=0)
	},
	names[6]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[6], begin_step=0)
	},
	names[7]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[7], begin_step=0)
	},
	names[8]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[8], begin_step=0)
	},
	names[9]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[9], begin_step=0)
	},
	names[10]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[10], begin_step=0)
	},
	names[11]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[11], begin_step=0)
	},
	names[12]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[12], begin_step=0)
	},
	names[13]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[13], begin_step=0)
	},
	names[14]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[14], begin_step=0)
	},
	names[15]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[15], begin_step=0)
	},
	
}

print("CREATING PRUNED MODEL")
pruning_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

m = 0
for layer in pruning_model.non_trainable_variables:
	if('/mask' in layer.name):
		print(type(layer[0,0]))
		layer.assign(model_mask[m])
		m+=2

		


print("CREATING STRIPED PRUNED MODEL")
stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruning_model)



#print(stripped_pruned_model.get_mask())
'''
for layer in stripped_pruned_model.non_trainable_variables:
	print(layer.name)
'''
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
#x_train = x_train.reshape(-1, 28, 28, 1)
#x_test = x_test.reshape(-1, 28, 28, 1)

x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255


t = np.array([])
for i in range(100):
	s = time.time()
	p = pruning_model.predict(x_train[i].reshape(1,32,32,3))
	e = time.time()
	t = np.append(t,e-s)
	#print("h5 pruned:",e-s)
print("h5 pruned mean:",np.mean(t), "+/-", np.std(t))


t = np.array([])
for i in range(100):
	s = time.time()
	p = stripped_pruned_model.predict(x_train[0].reshape(1,32,32,3))
	e = time.time()
	t = np.append(t,e-s)
	#print("h5 stripped:",e-s)	
print("h5 stripped pruned mean:",np.mean(t), "+/-", np.std(t))


model_ = tf.keras.models.load_model("model_32_000.h5")
t = np.array([])
for i in range(100):
	s = time.time()
	p = model_.predict(x_train[0].reshape(1,32,32,3),verbose=0)
	e = time.time()
	t = np.append(t,e-s)
	#print("h5 complete model:",e-s)
print("h5 complete mean:",np.mean(t), "+/-",np.std(t))


#-----------------------------------------tflite----------------------------------------------
'''
converter = tf.lite.TFLiteConverter.from_keras_model(pruning_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
conv_model = converter.convert()
conv_model_file = 'p_32_100.tflite'
# Save the model.
with open(conv_model_file, 'wb') as f:
    f.write(conv_model)
'''


converter = tf.lite.TFLiteConverter.from_keras_model(stripped_pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
#converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
converter.target_spec.supported_types = [tf.float32]
conv_model = converter.convert()
conv_model_file = 'sp_32_100.tflite'
# Save the model.
with open(conv_model_file, 'wb') as f:
    f.write(conv_model)


# Load the TFLite model.
interpreter = tflite.Interpreter('sp_32_100.tflite')
interpreter.allocate_tensors()

times = np.array([])
for i in range(100):
	p,t = inference(interpreter, x_test[i].reshape(1,32,32,3))
	times = np.append(times, t)
	
print("tflite stripped mean:",np.mean(times), "+/-", np.std(times))


data = (x_train[0].reshape(1,32,32,3)*255).astype(np.uint8)

q_8_000_ = tf.keras.models.load_model("model_8_000.h5")

q_8_000 = tfmot.quantization.keras.quantize_model(q_8_000_)
converter = tf.lite.TFLiteConverter.from_keras_model(q_8_000)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.uint8]
#converter.representative_dataset = data
conv_model = converter.convert()
conv_model_file = 'q_8_000.tflite'
# Save the model.
with open(conv_model_file, 'wb') as f:
    f.write(conv_model)


interpreter = tflite.Interpreter('q_8_000.tflite')
interpreter.allocate_tensors()

times = np.array([])
for i in range(100):
	p,t = inference(interpreter, x_test[i].reshape(1,32,32,3))
	times = np.append(times, t)
	
print("tflite quantized mean:",np.mean(times), "+/-", np.std(times))
    

#-----------------------------------------------------------------------------------------------------
model = tf.keras.models.load_model('model_8_100.h5')


sparsities = []
names = []
model_mask = []
ss = []
for i,layer in enumerate(model.trainable_weights):
	#print(layer.name)
	flat_array = np.array((tf.reshape(layer,[-1])))
	#print(type(flat_array[0]))
	n_zeros = np.count_nonzero(flat_array == 0.0)
	#print(n_zeros)
	
	#print(flat_array.shape[0])
	sparsity = n_zeros/flat_array.shape[0]
	sparsities.append(sparsity)
	
	if "kernel" in layer.name:
		ss.append(sparsity)

	mask = tf.cast(tf.where(tf.abs(layer)>0,1,0), tf.float32)
	if("conv" in layer.name) or("dense" in layer.name):
		model_mask.append(mask)
	
	
	#print(i,"-",layer.name,sparsity)


for layer in model.layers:
	
	if ('conv' in layer.name) or ('dense' in layer.name):
		#print(layer.name)
		names.append(layer.name)


pruning_params = {
	names[0]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[0], begin_step=0)
	},
	names[1]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[1], begin_step=0)
	},
	names[2]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[2], begin_step=0)
	},
	names[3]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[3], begin_step=0)
	},
	names[4]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[4], begin_step=0)
	},
	names[5]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[5], begin_step=0)
	},
	names[6]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[6], begin_step=0)
	},
	names[7]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[7], begin_step=0)
	},
	names[8]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[8], begin_step=0)
	},
	names[9]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[9], begin_step=0)
	},
	names[10]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[10], begin_step=0)
	},
	names[11]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[11], begin_step=0)
	},
	names[12]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[12], begin_step=0)
	},
	names[13]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[13], begin_step=0)
	},
	names[14]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[14], begin_step=0)
	},
	names[15]: {
		'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsities[15], begin_step=0)
	},
	
}

pruning_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

m = 0
for layer in pruning_model.non_trainable_variables:
	if('/mask' in layer.name):
		layer.assign(model_mask[m])
		m+=2
		
stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruning_model)

data = (x_train[0].reshape(1,32,32,3)*255).astype(np.uint8)


#quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(stripped_pruned_model)
#pqat_model = tfmot.quantization.keras.quantize_apply(quant_aware_annotate_model,tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme())


pqat_model = tfmot.quantization.keras.quantize_model(stripped_pruned_model)

#converter = tf.lite.TFLiteConverter.from_keras_model(stripped_pruned_model)
converter = tf.lite.TFLiteConverter.from_keras_model(pqat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
#converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
#converter.target_spec.supported_types = [tf.uint8]
#converter.representative_dataset = data
conv_model = converter.convert()
conv_model_file = 'pq_8_100.tflite'
# Save the model.
with open(conv_model_file, 'wb') as f:
    f.write(conv_model)


interpreter = tflite.Interpreter('pq_8_100.tflite')
interpreter.allocate_tensors()

times = np.array([])
for i in range(100):
	p,t = inference(interpreter, x_test[i].reshape(1,32,32,3))
	times = np.append(times, t)
	
print("tflite P->Q mean:",np.mean(times), "+/-", np.std(times))
