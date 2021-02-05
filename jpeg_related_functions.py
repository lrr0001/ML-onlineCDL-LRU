import tensorflow as tf
import math
import numpy as np
import transforms as transf


class Smooth_JPEG(tf.keras.layers.Layer):
    def __init__(self,q,lmbda,rho,alpha,noi,fftSz,*args,**kwargs):
        self.q = tf.reshape(q,(1,1,1,64))
        self.noi = noi
        self.lmbda = lmbda
        self.rho = rho
        self.alpha = alpha
        self.fftSz = fftSz
        self.fltr = np.asarray([0.,1.])
        super().__init__(*args,**kwargs)
        self.init_fun()
    def init_fun(self):
        self.W = RGB2JPEG_Coef(dtype=self.dtype)
        self.Wt = JPEG_Coef2RGB(dtype=self.dtype)
        self.xupdate = XUpdate_SmoothJPEG(self.lmbda,self.fftSz,tf.reshape(self.fltr,(1,2,1,1)),tf.reshape(self.fltr,(1,1,2,1)),dtype = self.dtype)
        self.relaxlayer = Relax_SmoothJPEG(self.alpha,self.q,dtype=self.dtype)
        self.yupdate = ZUpdate_JPEG(1.0,self.rho,self.q,self.W,self.Wt,dtype=self.dtype)
        self.uupdate = GammaUpdate_JPEG(dtype=self.dtype)

    # These initializations happen once per input (negC,y,By,u):
    def init_vars(self,s):
        negC = self.get_negative_C(s)
        y,By = self.init_y(s,negC)
        u = self.init_u(s)
        return (y,u,By,negC)
    def init_y(self,s,negC):
        return (s,negC)
    def init_u(self,s):
        return [0.,0.,0.]
    def get_negative_C(self,s):
        Ws = self.W(s)
        return [self.q*tf.math.round(Ws[channel]/self.q) for channel in range(len(Ws))]


    # iterative steps:
    def xstep(self,y):
        return (self.xupdate(y),0.)
    def relax(self,Ax,By,negC):
        return self.relaxlayer((negC,By))
    def ystep(self,x,u,AxplusC):
        return self.yupdate((x,AxplusC,u))
    def ustep(self,u,AxplusC,By):
        return self.uupdate((u,AxplusC,By))
    def solvestep(self,y,u,By,negC):
        x,Ax = self.xstep(y)
        AxplusC = self.relax(Ax,By,negC)
        y,By = self.ystep(x,u,AxplusC)
        u = self.ustep(u,AxplusC,By)
        return (y,u,By)

    # Before and After:
    def preprocess(self,s):
        return s
    def get_output(self,s,y,u,By,negC):
        x = self.xstep(y)
        #return (s - x,x)
        return x

    # The Call function    
    def call(self,s):
        s = self.preprocess(s)
        y,u,By,negC = self.init_vars(s)
        for ii in range(self.noi):
            y,u,By = self.solvestep(y,u,By,negC)
        return self.get_output(s,y,u,By,negC)

def generate_dct2D_filters():
    x = tf.reshape(2*tf.range(8.) + 1,(8,1,1,1))
    y = tf.reshape(2*tf.range(8.) + 1,(1,8,1,1))
    u = tf.reshape(math.pi*tf.range(8.)/16.,(1,1,8,1))
    v = tf.reshape(math.pi*tf.range(8.)/16.,(1,1,1,8))
    alpha_u = tf.reshape([1./math.sqrt(2),1.,1.,1.,1.,1.,1.,1.],(1,1,8,1))
    alpha_v = tf.reshape([1./math.sqrt(2),1.,1.,1.,1.,1.,1.,1.],(1,1,1,8))
    cosxu = alpha_u*tf.math.cos(x*u)
    cosyv = alpha_v*tf.math.cos(y*v)
    return tf.reshape(1./4.*cosxu*cosyv,(8,8,1,64))
    

def generate_idct2D_filters():
    return tf.reshape(tf.transpose(generate_dct2D_filters(),perm=(3,0,1,2)),(8,8,1,64))

class RGB2JPEG_Coef(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dct_filters = tf.cast(generate_dct2D_filters(),dtype=self.dtype)
        self.downsample = tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='valid',dtype=self.dtype)
    def call(self,inputs):
        yuv = tf.image.rgb_to_yuv(inputs)
        y,u,v = tf.split(yuv,axis=3,num_or_size_splits=3)
        y_calibrated = y - 1./2.
        u_ds = self.downsample(u)
        v_ds = self.downsample(v)
        ydcc_blks = tf.nn.conv2d(y_calibrated,self.dct_filters,strides=8,padding='VALID')
        udcc_blks = tf.nn.conv2d(u_ds,self.dct_filters,strides=8,padding='VALID')
        vdcc_blks = tf.nn.conv2d(v_ds,self.dct_filters,strides=8,padding='VALID')
        return (ydcc_blks,udcc_blks,vdcc_blks) # discrete cosine coefficients

class JPEG_Coef2RGB(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.idct_filters = tf.cast(generate_idct2D_filters(),dtype=self.dtype)
        self.Upsample = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest',dtype=self.dtype)
    def call(self,inputs):
        ydcc_blks,udcc_blks,vdcc_blks = inputs
        ydcc = tf.nn.depth_to_space(ydcc_blks,block_size=8)
        udcc = tf.nn.depth_to_space(udcc_blks,block_size=8)
        vdcc = tf.nn.depth_to_space(vdcc_blks,block_size=8)
        y_blks = tf.nn.conv2d(ydcc,self.idct_filters,strides=8,padding='VALID')
        u_ds_blks = tf.nn.conv2d(udcc,self.idct_filters,strides=8,padding='VALID')
        v_ds_blks = tf.nn.conv2d(vdcc,self.idct_filters,strides=8,padding='VALID')
        y = tf.clip_by_value(tf.nn.depth_to_space(y_blks,block_size=8) + 1./2.,0.,1.)
        u_ds = tf.nn.depth_to_space(u_ds_blks,block_size = 8)
        v_ds = tf.nn.depth_to_space(v_ds_blks,block_size = 8)      
        u = tf.clip_by_value(self.Upsample(u_ds),-0.5,0.5)
        v = tf.clip_by_value(self.Upsample(v_ds),-0.5,0.5)
        yuv = tf.concat((y,u,v),axis=3)
        return tf.clip_by_value(tf.image.yuv_to_rgb(yuv),0.,1.)
        
def get_JPEG_coef_mask(jpegImages,rgb2jpeg_coef_layer,epsilon):
    ydcc_blks,udcc_blks,vdcc_blks = rgb2jpeg_coef_layer(jpegImages)
    nonzero = lambda x: tf.math.greater(tf.math.abs(x),epsilon)
    return (nonzero(ydcc_blks),nonzero(udcc_blks),nonzero(vdcc_blks))

class Linear_JPEG_Compression_Approx(tf.keras.layers.Layer):
    def __init__(self,jpegImages,epsilon=1e-4,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.jpeg_coef2rgb = JPEG_Coef2RGB(*args,**kwargs)
        self.rgb2jpeg_coef = RGB2JPEG_Coef(*args,**kwargs)
        self.masky,self.masku,self.maskv = get_JPEG_coef_mask(jpegImages,self.rgb2jpeg_coef,epsilon=1e-4)
    def call(self,inputs):
        y_jpeg_coef,u_jpeg_coef,v_jpeg_coef = self.rgb2jpeg_coef(inputs)
        y_jpeg_coef_kept = tf.where(self.masky,y_jpeg_coef,0.)
        u_jpeg_coef_kept = tf.where(self.masku,u_jpeg_coef,0.)
        v_jpeg_coef_kept = tf.where(self.maskv,v_jpeg_coef,0.)
        return self.jpeg_coef2rgb((y_jpeg_coef_kept,u_jpeg_coef_kept,v_jpeg_coef_kept))

def halfqminus(q):
    return q/2 - 1./510.

def halfqplus(q):
    return q/2 + 1./510.

class XUpdate_SmoothJPEG(tf.keras.layers.Layer):
    def __init__(self,lmbda,fftSz,fltr1,fltr2,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lmbda = lmbda
        self.fft = transf.fft2d_multichannel(fftSz)
        self.ifft = transf.ifft2d_multichannel(fftSz)
        self.fltr1 = self.fft(fltr1)
        self.fltr2 = self.fft(fltr2)
    def call(self,inputs):
        A = 1.0 + self.lmbda*(tf.math.conj(self.fltr1)*self.fltr1 + tf.math.conj(self.fltr2)*self.fltr2)
        return self.ifft(self.fft(inputs)/A)

class Relax_SmoothJPEG(tf.keras.layers.Layer):
    def __init__(self,alpha,q,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.alpha = alpha
        self.q = q
    def call(self,inputs):
        QWs,QWz = inputs
        return [(1.0 - self.alpha)*QWz[channel] - (2.0 - self.alpha)*QWs[channel] for channel in range(len(QWs))]

class ZUpdate_JPEG(tf.keras.layers.Layer):
    def __init__(self,mu,rho,q,W,Wt,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.qntzn_adjst = QuantizationAdjustment(mu,rho,q,dtype=self.dtype)
        self.rho = rho
        self.mu = mu
        self.q = q
        self.W = W
        self.Wt = Wt
    def call(self,inputs):
        fx,Axminuss,gamma_over_rho = inputs
        Wx = self.W(fx)
        r = [-Axminuss[channel] - Wx[channel] - gamma_over_rho[channel] for channel in range(len(Wx))]
        Wdeltaz = [self.qntzn_adjst((Wx[channel],r[channel])) for channel in range(len(Wx))]
        return (fx + self.rho/(self.mu + self.rho)*self.Wt(r) + self.Wt(Wdeltaz),[self.q*tf.math.round((Wx[channel] + self.rho/(self.mu + self.rho)*r[channel] + Wdeltaz[channel])/self.q) for channel in range(len(Wx))])

class GammaUpdate_JPEG(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def call(self, inputs):
        gamma_over_rho,Axminuss,QWz = inputs
        return [gamma_over_rho[channel] + Axminuss[channel] + QWz[channel] for channel in range(len(QWz))]
        

class QuantizationAdjustment(tf.keras.layers.Layer):
    def __init__(self,mu,rho,q,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mu = mu
        self.rho = rho
        self.q = q
    def call(self,inputs):
        Wx,r = inputs
        #r = -Axminuss - Wx - gamma_over_rho
        y = Wx + self.rho/(self.mu + self.rho)*r
        dctelemerror = lambda Wdeltaz: self.mu/2*tf.math.square(self.rho/(self.mu + self.rho)*r + Wdeltaz) + self.rho/2*tf.math.square(-self.mu/(self.mu + self.rho) + self.q*tf.math.round((y + Wdeltaz)/self.q) - y)

        candidate1 = - tf.math.sign(r)*tf.math.minimum(self.rho/(self.mu + self.rho)*tf.math.abs(r),tf.math.abs(self.q*tf.math.round(y/self.q) - y - tf.math.sign(r)*self.q*halfqminus(self.q)))
        candidate2 = self.q*tf.math.round(y/self.q) - halfqplus(self.q)*self.q*tf.math.sign(y - self.q*tf.math.round(y/self.q) - self.rho/(self.mu + self.rho)*r) - y
        candidate3 = - tf.math.sign(r)*tf.math.minimum(self.rho/(self.mu + self.rho)*tf.math.abs(r),tf.math.abs(self.q*tf.math.round(y/self.q) - y - tf.math.sign(r)*self.q*(1. + halfqminus(self.q))))
        bestcandidate = tf.where(dctelemerror(candidate2) < dctelemerror(candidate1),candidate2,candidate1)
        bestcandidate = tf.where(dctelemerror(candidate3) < dctelemerror(bestcandidate),candidate3,bestcandidate)
        return bestcandidate

