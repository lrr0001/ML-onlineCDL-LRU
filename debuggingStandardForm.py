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
nots = 100
lraParam = {'n_components': 5}
dtype = tf.complex128

D = tf.random.normal(shape=(1,) + fltrSz + (noc,nof,),dtype=dtype.real_dtype)
dictObj = fctr.dictionary_object2D_init(fftSz=fftSz,D=D,rho=rho,lraParam=lraParam)
dictObj2 = fctr.dictionary_object2D_init_full(fftSz=fftSz,D=D,rho=rho,lraParam=lraParam)


w = tf.complex(tf.random.normal((nots,) + fftSz + (noc,1),dtype=dtype.real_dtype),tf.random.normal((nots,) + fftSz + (noc,1),dtype = dtype.real_dtype))
w = w + tf.math.conj(tf.reverse(w,axis=(2,3)))
w = w[slice(None),slice(None),slice(0,5,1),slice(None),slice(None)]

B = transf.fft2d_inner(fftSz)(tf.random.normal((1,) + fltrSz + (noc,nof,),dtype=dtype.real_dtype))

x = tf.linalg.matmul(B,w,adjoint_a=True)
xi = tf.keras.layers.Input(x.shape[1:],dtype=dtype)
y = rho*xi + tf.linalg.matmul(B,tf.linalg.matmul(B,xi),adjoint_a=True)
z = dictObj.qinv(y)
z2 = dictObj2.qinv(y)
modelA = tf.keras.Model(xi,z)
modelB = tf.keras.Model(xi,z2)
xt = x[slice(0,10,1),slice(None),slice(None),slice(None),slice(None)]

with tf.GradientTape() as tape:
    zt = modelA(xt)
    loss = tf.keras.losses.MSE(xt,zt)

grad = tape.gradient(target=loss,sources=modelA.trainable_variables)

with tf.GradientTape() as tape2:
    zt2 = modelB(xt)
    loss2 = tf.keras.losses.MSE(xt,zt2)

grad2 = tape2.gradient(target=loss2,sources=modelB.trainable_variables)

optimizerA=tf.keras.optimizers.SGD(0.00001)
optimizerB=tf.keras.optimizers.SGD(0.00001)

optimizerA.apply_gradients(zip(grad,modelA.trainable_variables))

optimizerB.apply_gradients(zip(grad2,modelB.trainable_variables))

modelB._dict_update()
