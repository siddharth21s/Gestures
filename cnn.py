# Convolutional Neural Network
# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
# Pickle libs
import pickle
import numpy as np

# Image Part 
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Initialising the CNN
cnn_obj = Sequential()

# step 1
#layer 1
cnn_obj.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
cnn_obj.add(MaxPooling2D(pool_size = (2, 2))) # step 2
# layer 2
cnn_obj.add(Convolution2D(32, (3, 3), activation = 'relu'))
cnn_obj.add(MaxPooling2D(pool_size = (2, 2)))

# layer 3
cnn_obj.add(Convolution2D(32, (3, 3), activation = 'relu'))
cnn_obj.add(MaxPooling2D(pool_size = (2, 2)))


# flatten
cnn_obj.add(Flatten())

# connect flatten layer to ANN
cnn_obj.add(Dense(units = 128, activation = 'relu'))
#cnn_obj.add(Dense(units = 192, activation = 'relu'))
cnn_obj.add(Dense(units = 128, activation = 'relu'))
cnn_obj.add(Dense(units = 5, activation = 'softmax'))

# Compile
cnn_obj.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#rescale images for training
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Data/Train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Data/Test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

m_train = training_set.class_indices
m_test = test_set.class_indices


print(cnn_obj.summary())

cnn_obj.fit_generator(training_set,
                         epochs = 10,
                         verbose = 1,
                         validation_data = test_set)# steps_per_epoch = 2000,


#Save in Model in File

cnn_obj.save('model_demo_ABCDE.h5')
'''
f=open('model_Demo_.txt','wb')

pickle.dump(cnn_obj,f)

f.close()

#Load Model from file
f=open('model_with_2convolutional_4dense_layer_5epoch_predict_ABCDv.1.2.txt','rb')

cnn_obj=pickle.load(f)

f.close()



#test image
test1=[]
test = image.load_img('Q/A_test.jpg',target_size = (64,64))
test = image.img_to_array(test)
test = np.expand_dims(test,axis=0)
test1.append(test)
test = image.load_img('Q/B_test.jpg',target_size = (64,64))
test = np.expand_dims(test,axis=0)
test1.append(test)
test = image.load_img('Q/C_test.jpg',target_size = (64,64))
test = np.expand_dims(test,axis=0)
test1.append(test)
test = image.load_img('Q/D_test.jpg',target_size = (64,64))
test = np.expand_dims(test,axis=0)
test1.append(test)
res=[]
for t in test1:
    res.append(cnn_obj.predict(t))

if res[0]==0:
    print('A')
elif res[1]==1:
    print('B')
elif res[2]==2:
    print('C')
elif res[3]==3:
    print('D')
    

test = image.load_img('Q/del_test.jpg',target_size = (64,64))
test = np.expand_dims(test,axis=0)
r=cnn_obj.predict(test)
r.argmax()
'''
c=training_set.class_indices
v=test_set.class_indices

print("train & test set indices:",c==v)

'''
import os
curr= os.getcwd()
os.chdir(curr+"/ModelTest")
files = [n for n in os.listdir(".") if os.path.isfile(n)]


datagen = ImageDataGenerator(rescale = 1./255)
files = datagen.flow_from_directory('ModelTest',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')
files.filenames
for i in files:
    print(i)


if c==v:
	for i in files:
		#path = i
		#test = image.load_img(path,target_size = (64,64))
		test = np.expand_dims(i,axis=0)
		r=cnn_obj.predict(test)
		print('for ',i,": ",r.argmax())
os.chdir(curr)



-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

weight_file_path = 'path to your keras model'
net_model = load_model(weight_file_path)
sess = K.get_session()

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), 'name of the output tensor')
graph_io.write_graph(constant_graph, 'output_folder_path', 'output.pb', as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join('output_folder_path', 'output.pb'))

------------------------------------------------------------
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

    
    
    
from keras import backend as K
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in cnn_obj.outputs])
#tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)


=========================================================

import tensorflow as tf
from tensorflow.python.platform import gfile
GRAPH_PB_PATH = './model_demo_ABCDE.pb'
with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)

'''
