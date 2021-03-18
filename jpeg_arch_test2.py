import tensorflow as tf
import jpeg_related_functions as jrf
import transforms as transf

class XUpdate(tf.keras.layers.Layer):
    def __init__(self,lmbda,fftSz,fltr1,fltr2,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lmbda = tf.Variable(initial_value=lmbda,trainable=True)
        self.fft = transf.fft2d_multichannel(fftSz)
        self.ifft = transf.ifft2d_multichannel(fftSz)
        self.fltr1 = self.fft(fltr1)
        self.fltr2 = self.fft(fltr2)
    def call(self,inputs):
        A = 1.0 + self.lmbda*(tf.math.conj(self.fltr1)*self.fltr1 + tf.math.conj(self.fltr2)*self.fltr2)
        return self.ifft(self.fft(inputs)/A)

class Smooth_JPEG_Layer(jrf.Smooth_JPEG):
    def init_fun(self):
        self.W = jrf.RGB2JPEG_Coef(dtype=self.dtype)
        self.Wt = jrf.JPEG_Coef2RGB(dtype=self.dtype)
        self.xupdate = XUpdate(self.lmbda,self.fftSz,tf.reshape(self.fltr,(1,2,1,1)),tf.reshape(self.fltr,(1,1,2,1)),dtype = self.dtype)
        self.relaxlayer = jrf.Relax_SmoothJPEG(self.alpha,dtype=self.dtype)
        self.yupdate = jrf.ZUpdate_JPEG(1.0,self.rho,self.qY,self.qUV,self.W,self.Wt,dtype=self.dtype)
        self.uupdate = jrf.GammaUpdate_JPEG(dtype=self.dtype)
    def get_output(self,s,y,u,By,negC,itstats):
        x,Ax = self.xstep(y,u,By,negC)
        return (x,negC)
