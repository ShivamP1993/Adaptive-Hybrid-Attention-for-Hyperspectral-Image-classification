# Function to perform one hot encoding of the class labels 

def my_ohc(lab_arr):
    lab_arr_unique =  np.unique(lab_arr)
    r,c = lab_arr.shape
    r_u  = lab_arr_unique.shape
    
    one_hot_enc = np.zeros((r,r_u[0]), dtype = 'float')
    
    for i in range(r):
        for j in range(r_u[0]):
            if lab_arr[i,0] == lab_arr_unique[j]:
                one_hot_enc[i,j] = 1
    
    return one_hot_enc

# Function that takes the confusion matrix as input and
# calculates the overall accuracy, producer's accuracy, user's accuracy,
# Cohen's kappa coefficient and syandard deviation of 
# Cohen's kappa coefficient

def accuracies(cm):
  import numpy as np
  num_class = np.shape(cm)[0]
  n = np.sum(cm)

  P = cm/n
  ovr_acc = np.trace(P)

  p_plus_j = np.sum(P, axis = 0)
  p_i_plus = np.sum(P, axis = 1)

  usr_acc = np.diagonal(P)/p_i_plus
  prod_acc = np.diagonal(P)/p_plus_j

  theta1 = np.trace(P)
  theta2 = np.sum(p_plus_j*p_i_plus)
  theta3 = np.sum(np.diagonal(P)*(p_plus_j + p_i_plus))
  theta4 = 0
  for i in range(num_class):
    for j in range(num_class):
      theta4 = theta4+P[i,j]*(p_plus_j[i]+p_i_plus[j])**2

  kappa = (theta1-theta2)/(1-theta2)

  t1 = theta1*(1-theta1)/(1-theta2)**2
  t2 = 2*(1-theta1)*(2*theta1*theta2-theta3)/(1-theta2)**3
  t3 = ((1-theta1)**2)*(theta4 - 4*theta2**2)/(1-theta2)**4

  s_sqr = (t1+t2+t3)/n

  return ovr_acc, usr_acc, prod_acc, kappa, s_sqr

# Import Relevant libraries and classes
import scipy.io as sio
import numpy as np
import tqdm
from sklearn.decomposition import PCA
import tensorflow as tf
keras = tf.keras
from keras import backend as K
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout
from keras.layers import Conv2D, Flatten, Lambda, Conv3D, Conv3DTranspose,BatchNormalization,Conv1D
from keras.layers import Reshape, Conv2DTranspose, Concatenate, Multiply, Add, MaxPooling2D, MaxPooling3D
from keras import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from sklearn.metrics import confusion_matrix

## read training data

train_patches = np.load('.data/train_patches.npy')
train_labels = np.load('.data/train_labels.npy')-1

test_patches = np.load('.data/test_patches.npy')
test_labels = np.load('.data/test_labels.npy')

## Augmenting the data by rotation
tr90 = np.empty(np.shape(train_patches), dtype = 'float32')
tr180 = np.empty(np.shape(train_patches), dtype = 'float32')

for i in tqdm.tqdm(range(np.shape(train_patches)[0])):
  tr90[i,:,:,:] = np.rot90(train_patches[i,:,:,:])
  tr180[i,:,:,:] = np.rot90(tr90[i,:,:,:])
  tr270[i,:,:,:] = np.rot90(tr180[i,:,:,:])

train_patches = np.concatenate([train_patches, tr90, tr180, tr270], axis = 0)
train_labels = np.concatenate([train_labels,train_labels,train_labels,train_labels], axis = 0)

# Shuffling the training data
from sklearn.utils import shuffle
train_patches, train_labels = shuffle(train_patches, train_labels, random_state = 0)

train_patches = train_patches[:,:,:,0:144]
test_patches = test_patches[:,:,:,0:144]
train_vec = np.expand_dims(train_patches[:,5,5,:], axis = 2)
test_vec = np.expand_dims(test_patches[:,5,5,:], axis = 2)

def spec_att(x):

  conv1 = Conv1D(18, 20, strides=1, padding='valid',
    dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005), 
    name = 'conv1')(x)

  conv1 = BatchNormalization()(conv1)
  
  conv2 = Conv1D(36, 20, strides=1, padding='valid',
    dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005), 
    name ='conv2')(conv1)

  conv2 = BatchNormalization()(conv2)

  conv3 = Conv1D(72, 20, strides=2, padding='valid',
    dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005),
    name = 'conv3')(conv2)

  conv3 = BatchNormalization()(conv3)

  #conv_cat1 = Concatenate(axis = 0)([conv1, conv3])
  
  conv4 = Conv1D(108, 20, strides=2, padding='valid',
    dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005), 
    name = 'conv4')(conv3)

  conv4 = BatchNormalization()(conv4)

  #conv_cat2 = Concatenate(axis = 2)([conv3, conv4])
  
  conv5 = Conv1D(144, 13, strides=1, padding='valid',
    dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005), 
    name ='conv5')(conv4)

  return conv5

def spat_att(x):
  conv1 = Conv2D(256, (3,3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), 
                            activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                            kernel_regularizer=regularizers.l2(0.01), name = 'conva15')(x)

  conv1 = BatchNormalization()(conv1)

  conv2 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                              kernel_regularizer=regularizers.l2(0.01), name = 'conva25')(conv1) 
  
  conv2 = BatchNormalization()(conv2)

  #conv_cat1 = Concatenate(axis = 3)([conv1, conv2])

  conv3 = Conv2D(512, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                              kernel_regularizer=regularizers.l2(0.01), name = 'conva35')(conv2)
  
  conv3 = BatchNormalization()(conv3)

  conv_cat1 = Concatenate(axis = 3)([conv1, conv3])

  mp1 = MaxPooling2D(pool_size = (2,2))(conv_cat1)

  #conv_cat2 = Concatenate(axis = 3)([conv3, conv4]) 

  conv5 = Conv2DTranspose(256, (4,4), strides=(1, 1), padding='valid', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                              kernel_regularizer=regularizers.l2(0.01), name = 'conva55')(mp1)

  conv5 = BatchNormalization()(conv5)
  #conv_cat3 = Concatenate(axis = 3)([conv_cat2, conv5])

  conv6 = Conv2DTranspose(256, (5,5), strides=(1, 1), padding='valid', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                              kernel_regularizer=regularizers.l2(0.01), name = 'conva65')(conv5) 

  conv6 = BatchNormalization()(conv6)   

  conv7 = Conv2DTranspose(144, (1,1), strides=(1, 1), padding='valid', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                              kernel_regularizer=regularizers.l2(0.01), name = 'conva66')(conv6) 
  return conv7

def clf(x):

    conv1 = Conv3D(32, (3,3,36), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), 
                            activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                            kernel_regularizer=regularizers.l2(0.01), name = 'conva1')(x)

    conv1 = BatchNormalization()(conv1)

    mp1 = MaxPooling3D(pool_size=(2, 2, 2), padding="valid")(conv1)

    conv2 = Conv3D(64, (3,3,36), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), 
                            activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                            kernel_regularizer=regularizers.l2(0.01), name = 'conva2')(mp1)

    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.5)(conv2)

    mp2 = MaxPooling3D(pool_size=(2, 2, 2), padding="valid")(conv2)

    conv3 = Conv3D(128, (3,3,36), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), 
                            activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                            kernel_regularizer=regularizers.l2(0.01), name = 'conva3')(mp2)

    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.5)(conv3)

    conv4 = Conv3D(15, (2,2,36), strides=(1, 1, 1), padding='valid', dilation_rate=(1, 1, 1), 
                            activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                            kernel_regularizer=regularizers.l2(0.01), name = 'conva4')(conv3)


    return Reshape([15])(conv4)

# implementation of wasserstein loss
from keras import backend
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

class Wt_Add(keras.layers.Layer):
    def __init__(self, units=1, input_dim=1):
        super(Wt_Add, self).__init__()
        w_init = tf.initializers.GlorotUniform()
        self.w1 = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        self.w2 = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )       

    def call(self, input1, input2, input3):
        return tf.multiply(input1,self.w1) + tf.multiply(input2, self.w2) + tf.multiply(input3, (1-self.w1-self.w2))

xA = Input(shape=(11,11,144), name='inputA') 
xB = Input(shape=(144,1), name='inputB')    
yT = Input(shape=(15,), name = 'inputY')

att_spec = spec_att(xB)

att_spat = spat_att(xA)
att_spat = Reshape([11,11,144,1])(att_spat)

feat_spec = Multiply()([xA, att_spec])
feat_spec = Reshape([11,11,144,1])(feat_spec)

xA2 = Reshape([11,11,144,1])(xA)
feat_spat = Multiply()([xA2, att_spat])

wt_add = Wt_Add(1,1)
input_new = wt_add(xA2, feat_spec, feat_spat)

clsf = clf(input_new)
clsf = Reshape([15])(clsf)

model_att = Model([xA,xB,yT], clsf, name = 'att_clf')
model_att.add_loss(wasserstein_loss(yT, clsf))

optim = keras.optimizers.Nadam(0.00002, beta_1=0.9, beta_2=0.999)
  
# Compiling the model
model_att.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
k=0
import gc
for epoch in range(150): 
  gc.collect()
  model_att.fit(x = [train_patches, train_vec, my_ohc(np.expand_dims(train_labels, axis = 1))], 
                  y = my_ohc(np.expand_dims(train_labels, axis = 1)),
                  epochs=1, batch_size = 64, verbose = 1)
  
  preds2 = model_att.predict([test_patches[:,:,:,0:144], test_vec, my_ohc(np.expand_dims(test_labels, axis = 1))], batch_size = 64)

  conf = confusion_matrix(test_labels, np.argmax(preds2,1))
  ovr_acc, _, _, _, _ = accuracies(conf)

  print(epoch)
  print(np.round(100*ovr_acc,2))
  if ovr_acc>=k:
    model_att.save('.models/model')
    k = ovr_acc
    ep = epoch
  print('acc_max = ', np.round(100*k,2), '% at epoch', ep)