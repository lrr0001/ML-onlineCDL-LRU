import tensorflow as tf
import transforms as transf
import util
class QInv_Tight_Frame(tf.keras.layers.Layer):
    def __init__(self,dmul,dhmul,rho,*args,**kwargs):
        # layers are included in the inputs for the purposes of sharing weights.
        super().__init__(*args,**kwargs)
        self.dmul = dmul
        self.dhmul = dhmul
        self.rho = rho

    def get_config(self):
        return {'rho': self.rho}

    def call(self, inputs):
        return (1./self.rho)*(inputs - self.dhmul(self.dmul(inputs))/(self.rho + 1.))

class DMul(tf.keras.layers.Layer):
    def __init__(self,dhmul,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dhmul = dhmul

    def call(self, inputs):
        return tf.matmul(a=self.dhmul.Df,b=inputs)


class DhMul(tf.keras.layers.Layer):
    def __init__(self,Df,*args,dtype=tf.complex64,**kwargs):
        super().__init__(*args,**kwargs)
        with tf.name_scope(self.name):
            self.Dfprev = tf.Variable(initial_value=Df,trainable=False,dtype=dtype,name='Dfreq_previous')
            self.Df = tf.Variable(initial_value=Df,trainable=True,name='Dfreq')
        self.varname = self.Df.name
    def get_config(self):
        return {'varname': self.varname}

    def call(self, inputs):
        return tf.matmul(a=self.Df,b=inputs,adjoint_a=True)

class Coef_Divide_By_R(tf.keras.layers.Layer):
    def __init__(self,D,noc,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with tf.name_scope(self.name):
            self.D = tf.Variable(initial_value = D,trainable=False,name='D')
            self.R = tf.Variable(initial_value = computeR(D,noc),trainable=False,name='R')
    def call(self,inputs):
        R = tf.cast(tf.reshape(self.R,self.R.shape[:3] + (self.R.shape[4],self.R.shape[3],) + self.R.shape[5:]),dtype=self.dtype)
        return inputs/R


def init_dict(self,fftSz,D,name):
    assert(tf.dtypes.as_dtype(self.dtype).is_complex)
    Dnormalized = D/tf.math.sqrt(tf.reduce_sum(input_tensor=D**2,axis=(1,2,3),keepdims=True))
    noc = D.shape[-2]
    self.divide_by_R = Coef_Divide_By_R(Dnormalized,noc,name=name + 'div_by_R',dtype=self.dtype)
    return self.FFT(self.divide_by_R.D)

class dictionary_object2D_init(tf.keras.layers.Layer):
    def __init__(self,fftSz,D,rho,objname,n_components=3,cmplxdtype=tf.complex128,epsilon=1e-6,*args,**kwargs):
        cmplxdtype = util.complexify_dtype(D.dtype)
        tf.keras.layers.Layer.__init__(self,dtype=cmplxdtype,name=objname,*args,**kwargs)
        self.fftSz = fftSz
        self.noc = D.shape[-2]
        self.nof = D.shape[-1]
        self.fltrSz = D.shape[1:3]
        self.epsilon = epsilon
        self.rho = rho
        self.n_components = n_components
        self.FFT = transf.fft2d_inner(self.fftSz)
        Df = self.init_dict(fftSz=fftSz,D=D,name=self.name)

        self.dhmul = DhMul(Df,*args,dtype=self.dtype,name=self.name + '/dhmul',**kwargs)
        self.dmul = DMul(self.dhmul,*args,dtype=self.dtype,name=self.name + '/dmul',**kwargs)
        self.qinv = QInv_Tight_Frame(self.dmul,self.dhmul,rho,*args,dtype=self.dtype,name = self.name + '/qinv',**kwargs)


import numpy as np


fltrSz = (3,3)
fftSz = (8,8)
noc = 5
nof = 4
rho = 1.0
n_components = 4
nots = 100
dtype = tf.complex128

D = tf.random.normal(shape=(1,) + fltrSz + (noc,nof,),dtype=dtype.real_dtype)
AInv2 = dictionary_object2D_init(fftSz=fftSz,D=D,rho=rho,objname='dictObj2',n_components=n_components,cmplxdtype=dtype)

w = tf.complex(tf.random.normal((nots,) + fftSz + (noc,1),dtype=dtype.real_dtype),tf.random.normal((nots,) + fftSz + (noc,1),dtype = dtype.real_dtype))
w = w + tf.math.conj(tf.reverse(w,axis=(2,3)))
w = w[slice(None),slice(None),slice(0,5,1),slice(None),slice(None)]

B = transf.fft2d_inner(fftSz)(tf.random.normal((1,) + fltrSz + (noc,nof,),dtype=dtype.real_dtype))

x = tf.linalg.matmul(B,w,adjoint_a=True)
xi = tf.keras.layers.Input(x.shape[1:],dtype=dtype)
y = rho*xi + tf.linalg.matmul(B,tf.linalg.matmul(B,xi),adjoint_a=True)
z2 = AInv2(y)

#my_model = ppg.Model_PostProcess(xi,z)
#my_model2 = ppg.Model_PostProcess(xi,z2)
my_model2 = tf.keras.Model(xi,z2)

my_model2.compile(optimizer=tf.keras.optimizers.SGD(0.001),loss=tf.keras.losses.MSE,run_eagerly=False)
import datetime
log_dir = "logs/logs_test/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S.log")
with tf.profiler.experimental.Profile(log_dir):
    my_model2.fit(x=x,y=x,shuffle=False,batch_size=10,epochs=16)
for tv in my_model2.trainable_variables:
    print(tv.name[:tv.name.index(":")])

my_model2.save_weights('second_saved_dictionary_object2d-weights')



