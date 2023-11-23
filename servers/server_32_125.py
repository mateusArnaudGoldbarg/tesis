import tensorflow as tf
import os
import zipfile
from flask import Flask, request, jsonify
import numpy as np
import json
import time

app = Flask(__name__)
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def load_model():
	# Caminho para o arquivo ZIP
	caminho_arquivo_zip = 'sp_32_125.zip'

	# Abrir o arquivo ZIP
	with zipfile.ZipFile(caminho_arquivo_zip, 'r') as zip_ref:
	    # Extrair todo o conteúdo para a pasta de destino
	    zip_ref.extractall()

	interpreter = tf.lite.Interpreter('sp_32_125.tflite')
	interpreter.allocate_tensors()
	os.remove('sp_32_125.tflite')
	
	return interpreter
	

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
	
@app.route("/predict", methods=["POST"])
def predict():
	# Deserialize the JSON string back to a numpy array
	data = np.array(json.loads(request.data)).astype(np.float32)

	p,t = inference(interpreter, data)
	
	i = classes[np.argmax(p)]

	s = str(t) +" seconds : "+i
	# Do something with the numpy array

	# Return a JSON response
	return jsonify({"message": str(s)})

@app.route("/")
def index():
	return '''<h1>Mateus Goldbarg Tesis</h1>
		<p>This server contains the pruned model with alpha equal 1.25 model</p>
	'''


if __name__ == "__main__":
	interpreter = load_model()
	app.run(host='0.0.0.0', port=5003,debug=True)

