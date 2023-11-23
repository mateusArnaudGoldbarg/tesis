import requests
import numpy as np
import json
import tensorflow as tf

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

for i in range(100):
	#numpy_array = np.random.rand(1, 32, 32, 3)
	numpy_array = x_test[i].reshape(1,32,32,3)
	json_string = json.dumps(numpy_array.tolist())

	# Serialize the numpy array to JSON
	#json_string = json.dumps(numpy_array.tolist())

	# Send the JSON string to the Flask application using an HTTP POST request
	response = requests.post(
	    "http://127.0.0.1:5000/predict",
	    headers={"Content-Type": "application/json"},
	    data=json_string,
	)

	# Check the response status code
	if response.status_code == 200:
	    # Get the response content
	    response_content = response.content.decode()

	    # Deserialize the response content
	    response_data = json.loads(response_content)

	    # Print the response data
	    
	    print(response_data, classes[y_test[i,0]])
	    
	else:
	    # Handle the error
	    print("Error: {}".format(response.status_code))
