# import main Flask class and request object
from flask import Flask, request
#import tensorflow as tf
import numpy as np
import h5py
from memory_profiler import profile
import sys


"""
n_compressed = 9584032 bytes
compressed =    754577 bytes

"""

# create the Flask app
app = Flask(__name__)

#data = h5py.File('data.h5', 'r')
#d = data['tensors']
#model = d[()]

#open model
model__ = []
names__ = []
index__ = []

'''
# open the file as 'f'
with h5py.File('model_c.h5', 'r') as f:
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
          if "idx" not in m:
          	model__.append(m_)
          	#names__.append(member+"/"+m)
          else:
          	index__.append(m_.astype(np.int16))
          	
#print(names__[1])
print(index__[1].dtype)
print(model__[1].dtype)
'''
'''
file_path = 'model_c.h5'
with h5py.File(file_path, 'w') as hf:
    # Create a group
    group = hf.create_group('tensors')
    for i in range(len(model__)):
      # Create datasets within the group
      group.create_dataset(names__[i], data=model__[i])
      group.create_dataset(names__[i]+"_idx", data=index__[i])
      
      #group.create_dataset(str(model_n[i]+"_idx"), data=model_i[i])
      #break
      #group.create_dataset('tensor2', data=tensor2)


print('H5 file created successfully.')
'''



with h5py.File('n_data.h5', 'r') as f:
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
          #i_ = np.argwhere(m_!=0)
          
          #m__ = m_.flatten()
          #m__ = m__[m__ != 0]
          
          model__.append(m_)
          names__.append(member+"/"+m)
          #index__.append(i)
#print(model__[0])


'''
@profile
def get_model(model):
	model = model__[0]
	index_ = index__
'''


#model = []
#get_model(model)


total_size = sys.getsizeof(model__) + sum(sys.getsizeof(arr) for arr in model__) + sum(sys.getsizeof(arr) for arr in index__)

print('the python object size is:')
print(total_size)


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
