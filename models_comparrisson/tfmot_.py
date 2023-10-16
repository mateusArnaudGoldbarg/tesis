import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import time
from memory_profiler import profile
import sys
import tempfile
import zipfile
import os

print('Loading fake copressed model')
model = tf.keras.models.load_model('model.h5')
sparsities = []
names = []
model_mask = []

@profile
def normal_predict(model, data):
	a = model.predict(data)
	#print(a[0])
	return a

@profile
def pruned_predict(model, data):
	a = model.predict(data)
	#print(a[0])
	return a
	
@profile
def stripped_pruned_predict(model, data):
	a = model.predict(data)
	#print(a[0])
	return a

@profile
def quantized_predict(model,data):
	a = model.predict(data)
	#print(a[0])
	return a
	
@profile
def pq_predict(model,data):
	a = model.predict(data)
	#print(a[0])
	return a
	

def get_gzipped_model_size(file):
  # It returns the size of the gzipped model in kilobytes.

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)/1000
  
  



'''
def quantize_weights(layer):
    #weights = layer.get_weights()
    #quantized_weights = tf.cast(np.round(layer * 2**8), tf.int8)
    #quantized_weights = [tf.cast(np.round(w * 2**8), tf.int8) / 2**num_bits for w in weights]
    l = tf.cast(layer,np.int8),2+10
    #layer.set_weights(quantized_weights)
    print(l.shape)
    #layer.set_weights(l)
    layer.assign(l)
''' 
	
def print_model_weights_sparsity(model):

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Wrapper):
            weights = layer.trainable_weights
        else:
            weights = layer.weights
        for weight in weights:
            # ignore auxiliary quantization weights
            if "quantize_layer" in weight.name:
                continue
            weight_size = weight.numpy().size
            zero_num = np.count_nonzero(weight == 0)
            print(
                f"{weight.name}: {zero_num/weight_size:.2%} sparsity ",
                f"({zero_num}/{weight_size})",
            )


print('Creating pruned model')
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

	
#for i in range(len(names)):
#	print(names[i], sparsities[i*2])

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

#for i in model_mask:
#	print(i.shape)

pruning_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

#print(len(pruning_model.layers))

m = 0
for layer in pruning_model.non_trainable_variables:
	if('/mask' in layer.name):
		layer.assign(model_mask[m])
		#print(layer.name)
		#print(layer.shape, model_mask[m].shape)
		#print()
		m+=2

#for layer in pruning_model.non_trainable_variables:
#	print(layer)

pruning_model.save('p_model.h5')
print("pruned model loaded!")

stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruning_model)
#print_model_weights_sparsity(stripped_pruned_model)


print('creating quantized model')
#quantized_model = create_quantized_model()
q_model = tfmot.quantization.keras.quantize_model(stripped_pruned_model)
print('quantized model created')

print('creating prune_quantized model')
quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
              stripped_pruned_model)
pq_model = tfmot.quantization.keras.quantize_apply(
              quant_aware_annotate_model,
              tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme())
print('prune_quantized model created')



#model.summary()
#pruning_model.summary()



#IMPORTAÇÃO E NORRMALIZAÇÃO
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
#x_train = x_train.reshape(-1, 28, 28, 1)
#x_test = x_test.reshape(-1, 28, 28, 1)

x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255


pq_model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.0), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

#print(pq_model.get_weights())

print("ANTES----------")
for layer in pq_model.trainable_weights:
	print(layer.name)
	print(layer)
	break
		
for layer in pq_model.non_trainable_weights:
	if 'kernel_min' in layer.name or 'kernel_max' in layer.name or 'post_activation_min' in layer.name or 'post_activation_max' in layer.name:
		print(layer.name)
		print(layer)
		break


pq_model.fit(x_train[0:64], y_train[0:64], batch_size=64, epochs=1, validation_split=0.1)

print("DEPOIS----------")
for layer in pq_model.trainable_weights:
	print(layer.name)
	print(layer)
	break
		
for layer in pq_model.non_trainable_weights:
	if 'kernel_min' in layer.name or 'kernel_max' in layer.name or 'post_activation_min' in layer.name or 'post_activation_max' in layer.name:
		print(layer.name)
		print(layer)
		break

#print(pq_model.get_weights())

y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 10)

print('input shape',x_test[0:64].shape)
#================================================================================================
def eval_model(interpreter, test_images):
	input_index = interpreter.get_input_details()[0]["index"]
	output_index = interpreter.get_output_details()[0]["index"]
	
	test_images = tf.data.Dataset.from_tensor_slices(test_images).shuffle(50000).batch(64)
	
	# Run predictions on every image in the "test" dataset.
	prediction_digits = []
	for i, test_image in enumerate(test_images):
		if i % 1000 == 0:
			print(f"Evaluated on {i} results so far.")
		# Pre-processing: add batch dimension and convert to float32 to match with
		# the model's input data format.
		#test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
		interpreter.set_tensor(input_index, test_image)

		# Run inference.
		interpreter.invoke()

		# Post-processing: remove batch dimension and find the digit with highest
		# probability.
		output = interpreter.tensor(output_index)
		digit = np.argmax(output()[0])
		print(output()[0])
		prediction_digits.append(output)

	print(prediction_digits)
	print('\n')
	# Compare prediction results with ground truth labels to calculate accuracy.
	prediction_digits = np.array(prediction_digits)
	accuracy = (prediction_digits == y_test[0:640]).mean()
	return accuracy
#=========================================================================================


# QAT model
converter = tf.lite.TFLiteConverter.from_keras_model(q_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
qat_tflite_model = converter.convert()
qat_model_file = 'qat_model.tflite'
# Save the model.
with open(qat_model_file, 'wb') as f:
    f.write(qat_tflite_model)

# PQAT model
converter = tf.lite.TFLiteConverter.from_keras_model(pq_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
pqat_tflite_model = converter.convert()
pqat_model_file = 'pqat_model.tflite'
# Save the model.
with open(pqat_model_file, 'wb') as f:
    f.write(pqat_tflite_model)

print("QAT model size: ", get_gzipped_model_size(qat_model_file), ' KB')
print("PQAT model size: ", get_gzipped_model_size(pqat_model_file), ' KB')



interpreter = tf.lite.Interpreter(pqat_model_file)
interpreter.allocate_tensors()

pqat_test_accuracy = eval_model(interpreter,x_test[0:640])

print('Pruned and quantized TFLite test_accuracy:', pqat_test_accuracy)
#print('Pruned TF test accuracy:', pruned_model_accuracy)




'''
for layer in stripped_pruned_model.trainable_weights:
	layer.assign(tf.cast(np.round(layer*(2**8)), tf.float16))
	print(layer)
'''	




'''
start = time.time()
pruning_model.predict(x_test[0:64])
end1 = time.time()
model.predict(x_test[0:64])
end2 = time.time()
'''

print('predicting...')
start = time.time()
normal_predict(model, x_test[0:64])
end1 = time.time()
pruned_predict(pruning_model, x_test[0:64])
end2 = time.time()
stripped_pruned_predict(stripped_pruned_model,x_test[0:64])
end3 = time.time()
quantized_predict(q_model,x_test[0:64])
end4 = time.time()
pq_predict(pq_model,x_test[0:64])
end5 = time.time()


normal_size = sys.getsizeof(model)
pruning_size = sys.getsizeof(pruning_model)
stripped_pruned_size = sys.getsizeof(stripped_pruned_model)
quantized_size = sys.getsizeof(q_model)
pq_size = sys.getsizeof(pq_model)

print('timeflow of normal model', end1-start)
print('timeflow of pruned model', end2-end1)
print('timeflow of stripped pruned model', end3-end2)
print('timeflow of quantized model', end4-end3)
print('timeflow of prune_quantized model', end5-end4)
print('footprint of normal model', normal_size, 'bytes')
print('footprint of pruned model', pruning_size, 'bytes')
print('footprint of stripped pruned model', stripped_pruned_size, 'bytes')
print('footprint of quantized model', quantized_size, 'bytes')
print('footprint of prune_quantized model', pq_size, 'bytes')


#print_model_weights_sparsity(q_model)
#print_model_weights_sparsity(pq_model)

#for layer in pq_model.trainable_weights:
#	print(layer)
