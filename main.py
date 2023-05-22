# import main Flask class and request object
from flask import Flask, request
#import tensorflow as tf
import numpy as np
import h5py

# create the Flask app
app = Flask(__name__)

#data = h5py.File('data.h5', 'r')
#d = data['tensors']
#model = d[()]

#open model
model__ = []
names__ = []

# open the file as 'f'
with h5py.File('data.h5', 'r') as f:
    data = f['tensors']
    #print(data.members)
    members = list(data.keys())
    #print("Members in the h5 file:")
    for member in members:
        #print(member)
        d = data[member]
        for m in list(d.keys()):
          dd = d[m]
          m_ = dd[()]
          model__.append(m_)
          names__.append(member+"/"+m)



#model = tf.keras.models.load_model('test.h5')

@app.route('/')
def home_page():
    return '''<h1>HOME PAGE</h1>'''

@app.route('/model-input')
def model_input():
    # if key doesn't exist, returns None
    #inp = request.args.get('inp')

    #return '''<h1>The model input is: {}, and its type is: {}</h1>'''.format(inp, type(inp))

    data = request.get_json()
    #data['w'] = int(data['w'])
    #my_array = np.array(data['w'],dtype=np.float32)

    #print(my_array.shape)
    #my_array = my_array.reshape(1,32,32,3)
    #print(my_array.shape)
    #p = model.predict(my_array)
    #print(p)
    #return '''<h1>I am {}% sure that this is {}</h1>'''.format(p.max()*100, p.argmax())
    return '''<h1>{}</h1><p>{}</p>'''.format(names__[1],model__[1])


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
