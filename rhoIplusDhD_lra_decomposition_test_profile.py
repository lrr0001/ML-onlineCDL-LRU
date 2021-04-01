import tensorflow as tf
import numpy as np
import matrix_decompositions_tf as fctr
import transforms as transf
import post_process_grad as ppg


fltrSz = (3,3)
fftSz = (8,8)
noc = 5
nof = 4
rho = 1.0
n_components = 4
nots = 100
dtype = tf.complex128

D = tf.random.normal(shape=(1,) + fltrSz + (noc,nof,),dtype=dtype.real_dtype)
#AInv = fctr.dictionary_object2D_init(fftSz=fftSz,D=D,rho=rho,objname='dictObj1',n_components=n_components,cmplxdtype=dtype)
AInv2 = fctr.dictionary_object2D_init_full(fftSz=fftSz,D=D,rho=rho,objname='dictObj2',n_components=n_components,cmplxdtype=dtype)

w = tf.complex(tf.random.normal((nots,) + fftSz + (noc,1),dtype=dtype.real_dtype),tf.random.normal((nots,) + fftSz + (noc,1),dtype = dtype.real_dtype))
w = w + tf.math.conj(tf.reverse(w,axis=(2,3)))
w = w[slice(None),slice(None),slice(0,5,1),slice(None),slice(None)]

B = transf.fft2d_inner(fftSz)(tf.random.normal((1,) + fltrSz + (noc,nof,),dtype=dtype.real_dtype))

x = tf.linalg.matmul(B,w,adjoint_a=True)
xi = tf.keras.layers.Input(x.shape[1:],dtype=dtype)
y = rho*xi + tf.linalg.matmul(B,tf.linalg.matmul(B,xi),adjoint_a=True)
#z = AInv(y)
z2 = AInv2(y)

#my_model = ppg.Model_PostProcess(xi,z)
my_model2 = ppg.Model_PostProcess(xi,z2)

#my_model.compile(optimizer=tf.keras.optimizers.SGD(0.001),loss=tf.keras.losses.MSE,run_eagerly=False)
my_model2.compile(optimizer=tf.keras.optimizers.SGD(0.001),loss=tf.keras.losses.MSE,run_eagerly=False)
import datetime
log_dir = "logs/logs_test/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S.log")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(
#      log_dir = log_dir,
#      histogram_freq = 1,
#      profile_batch = '2,5'
#)
#my_model.fit(x=x,y=x,shuffle=False,batch_size=10,epochs=16)
with tf.profiler.experimental.Profile(log_dir):
    my_model2.fit(x=x,y=x,shuffle=False,batch_size=10,epochs=16)
for tv in my_model2.trainable_variables:
    print(tv.name[:tv.name.index(":")])

#my_model.save_weights('saved_dictionary_object2d-weights')
my_model2.save_weights('second_saved_dictionary_object2d-weights')
