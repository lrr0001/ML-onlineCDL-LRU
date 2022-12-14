import tensorflow as tf
import math
import numpy as np
import transforms as transf
import optmz

class Smooth_JPEG_Constant(optmz.ADMM):
    ''' This layer computes a smoothed version of a JPEG-compressed image. Input is an uncompressed RGB image.
        Output is a constant image of 0.5, and the compressed JPEG image.''' 
    def __init__(self,rho,alpha,noi,qY,qUV,lmbda,fftSz,*args,**kwargs):
        self.qY = tf.reshape(qY,(1,1,1,64))
        self.qUV = tf.reshape(qUV,(1,1,1,64))
        self.lmbda = lmbda
        self.fftSz = fftSz
        self.fltr = np.asarray([-1.,1.])
        super().__init__(rho,alpha,noi,*args,**kwargs)
        self.init_fun()
    def init_fun(self):
        self.W = RGB2JPEG_Coef(dtype=self.dtype)
        self.Wt = JPEG_Coef2RGB(dtype=self.dtype)

    # These initializations happen once per input (negC,y,By,u):
    def init_x(self,s,negC):
        return (None,None)
    def init_y(self,s,x,Ax,negC):
        #return (self.Wt(negC),negC)
        #return (self.Wt(negC),None)
        return (None,None)
    def init_u(self,s,Ax,By,negC):
        #return [0.,0.,0.]
        return (None,)
    def get_negative_C(self,s):
        Ws = self.W(s)
        Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        return threeChannelQuantize(Ws,self.qY,self.qUV,Yoffset)
    def init_itstats(self,s):
        return []


    # iterative steps:
    def xstep(self,y,u,By,negC):
        #return (self.xupdate(y),None)
        return (None,None)
    def relax(self,Ax,By,negC):
        #return self.relaxlayer((negC,By))
        return (None,)
    def ystep(self,x,u,Ax_relaxed,negC):
        #return (self.yupdate((x,negC)),None)
        return (None,None)
    def ustep(self,u,Ax_relaxed,By,negC):
        return (None,)

    # Before and After:
    def preprocess(self,s):
        #rgb2yuv = RGB2YUV(dtype=self.dtype)
        #return rgb2yuv(s)
        return s
    def get_output(self,s,y,u,By,negC,itstats):
        ''' Outputs:
               Smoothed image (YUV)
               Compressed image (YUV)
               Raw image (YUV)'''
        #x,Ax = self.xstep(y,u,By,negC)
        compressedImg = self.Wt(negC)
        x = tf.zeros(compressedImg.shape,dtype = self.dtype) + 0.5
        return (x,self.Wt(negC))

class Smooth_JPEG_ACTUAL(optmz.ADMM):
    ''' This layer computes a smoothed version of a JPEG-compressed image. Input is an uncompressed RGB image.
        Output is a smoothed version of the image in YUV domain, a JPEG-compressed YUV image, and an uncompressed YUV image.''' 
    def __init__(self,rho,alpha,noi,qY,qUV,lmbda,fftSz,*args,**kwargs):
        self.qY = tf.reshape(qY,(1,1,1,64))
        self.qUV = tf.reshape(qUV,(1,1,1,64))
        self.init_cnsts(rho,alpha,noi,lmbda,fftSz,*args,**kwargs)
    def init_cnsts(self,rho,alpha,noi,lmbda,fftSz,*args,**kwargs):
        self.lmbda = lmbda
        self.fftSz = fftSz
        self.fltr = np.asarray([-1.,1.])
        super().__init__(rho,alpha,noi,*args,**kwargs)
        self.init_fun()
    def init_fun(self):
        #self.W = YUV2JPEG_Coef(dtype=self.dtype)
        self.W = RGB2JPEG_Coef(dtype=self.dtype)
        #self.Wt = JPEG_Coef2YUV(dtype=self.dtype)
        self.Wt = JPEG_Coef2RGB(dtype=self.dtype)
        self.xupdate = XUpdate_SmoothJPEG(self.lmbda,self.fftSz,tf.reshape(self.fltr,(1,2,1,1)),tf.reshape(self.fltr,(1,1,2,1)),dtype = self.dtype)
        #self.relaxlayer = Relax_SmoothJPEG(dtype=self.dtype) # move alpha to uupdate
        #self.yupdate = ZUpdate_JPEG(1.0,self.rho,self.qY,self.qUV,self.W,self.Wt,dtype=self.dtype)
        self.yupdate = ZUpdate_JPEG_Implicit(self.qY,self.qUV,self.W,self.Wt,dtype=self.dtype)
        #self.uupdate = GammaUpdate_JPEG(self.alpha,dtype=self.dtype)

    # These initializations happen once per input (negC,y,By,u):
    def init_x(self,s,negC):
        return (None,None)
    def init_y(self,s,x,Ax,negC):
        #return (self.Wt(negC),negC)
        return (self.Wt(negC),None)
    def init_u(self,s,Ax,By,negC):
        #return [0.,0.,0.]
        return (None,)
    def get_negative_C(self,s):
        Ws = self.W(s)
        Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        return threeChannelQuantize(Ws,self.qY,self.qUV,Yoffset)
    def init_itstats(self,s):
        return []


    # iterative steps:
    def xstep(self,y,u,By,negC):
        return (self.xupdate(y),None)
    def relax(self,Ax,By,negC):
        #return self.relaxlayer((negC,By))
        return (None,)
    def ystep(self,x,u,Ax_relaxed,negC):
        return (self.yupdate((x,negC)),None)
    def ustep(self,u,Ax_relaxed,By,negC):
        return (None,)

    # Before and After:
    def preprocess(self,s):
        #rgb2yuv = RGB2YUV(dtype=self.dtype)
        #return rgb2yuv(s)
        return s
    def get_output(self,s,y,u,By,negC,itstats):
        ''' Outputs:
               Smoothed image (YUV)
               Compressed image (YUV)
               Raw image (YUV)'''
        #x,Ax = self.xstep(y,u,By,negC)
        x = self.xupdate.last_call(y)
        return (x,self.Wt(negC))

class Smooth_JPEGY_ACTUAL(Smooth_JPEG_ACTUAL):
    def __init__(self,rho,alpha,noi,qY,lmbda,fftSz,*args,**kwargs):
        self.qY = tf.reshape(qY,(1,1,1,64))
        self.init_cnsts(rho,alpha,noi,lmbda,fftSz,*args,**kwargs)
    def get_negative_C(self,s):
        Ws = self.W(s)
        Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        return quantize(Ws,self.qY,Yoffset)
    def init_fun(self):
        self.W = Y2JPEG_Coef(dtype=self.dtype)
        self.Wt = JPEG_Coef2Y(dtype=self.dtype)
        self.xupdate = XUpdate_SmoothJPEG(self.lmbda,self.fftSz,tf.reshape(self.fltr,(1,2,1,1)),tf.reshape(self.fltr,(1,1,2,1)),dtype = self.dtype)
        self.yupdate = ZUpdate_JPEGY_Implicit(self.qY,self.W,self.Wt,dtype=self.dtype)
        self.rgb2yuv = RGB2YUV(dtype = self.dtype)
    def preprocess(self,s):
        s_YUV = self.rgb2yuv(s)
        return s_YUV[slice(None),slice(None),slice(None),slice(0,1)]

class Smooth_JPEGY_Constant(Smooth_JPEGY_ACTUAL):
    def init_x(self,s,negC):
        return (None,None)
    def init_y(self,s,x,Ax,negC):
        return (None,None)
    def init_u(self,s,Ax,By,negC):
        return (None,)
    def init_itstats(self,s):
        return []
    # iterative steps:
    def xstep(self,y,u,By,negC):
        return (None,None)
    def relax(self,Ax,By,negC):
        return (None,)
    def ystep(self,x,u,Ax_relaxed,negC):
        return (None,None)
    def ustep(self,u,Ax_relaxed,By,negC):
        return (None,)
    def get_output(self,s,y,u,By,negC,itstats):
        compressedImg = self.Wt(negC)
        x = tf.zeros(compressedImg.shape,dtype = self.dtype) + 0.5
        return (x,self.Wt(negC))

class Smooth_JPEGY(Smooth_JPEGY_ACTUAL):
    pass

class Smooth_JPEG(Smooth_JPEG_ACTUAL):
    pass


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

_rgb_to_yuv_mat = [[0.299,0.587,0.114],
                  [-0.14714119,-0.28886916,0.43601035],
                  [0.61497538, -0.51496512,-0.10001026]]

_yuv_to_rgb_mat = [[1.,0.,1.13988303],
                  [1.,-0.394642334, -0.58062185],
                  [1.,2.03206185,0.]]

class ColorTransform(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def call(self, inputs):
        transf_mat = tf.reshape(self.transf_mat,shape=tf.concat((tf.ones(tf.rank(inputs) - 1,dtype=tf.as_dtype('int32')),[3,3]),axis=0))
        return tf.linalg.matvec(transf_mat,inputs)
    def get_config(self):
        return {'Transform_Matrix': self.transf_mat}

class RGB2YUV(ColorTransform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.transf_mat = tf.convert_to_tensor(_rgb_to_yuv_mat,dtype=self.dtype)

class YUV2RGB(ColorTransform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.transf_mat = tf.convert_to_tensor(_yuv_to_rgb_mat,dtype=self.dtype)

class RGB2YUV_Transpose(ColorTransform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.transf_mat = tf.transpose(tf.convert_to_tensor(_rgb_to_yuv_mat,dtype=self.dtype))

class YUV2RGB_Transpose(ColorTransform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.transf_mat = tf.transpose(tf.convert_to_tensor(_yuv_to_rgb_mat,dtype=self.dtype))

class Color2JPEG_Coef(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dct_filters = tf.cast(generate_dct2D_filters(),dtype=self.dtype)
        self.downsample = tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='valid',dtype=self.dtype)
    def get_config(self):
        return {'dct_filters': self.dct_filters}
    def call(self,inputs):
        yuv = self.colortransform(inputs)
        y,u,v = tf.split(yuv,axis=3,num_or_size_splits=3)
        u_ds = self.downsample(u)
        v_ds = self.downsample(v)
        ydcc_blks = tf.nn.conv2d(y,self.dct_filters,strides=8,padding='VALID')
        udcc_blks = tf.nn.conv2d(u_ds,self.dct_filters,strides=8,padding='VALID')
        vdcc_blks = tf.nn.conv2d(v_ds,self.dct_filters,strides=8,padding='VALID')
        return (ydcc_blks,udcc_blks,vdcc_blks) # discrete cosine coefficients

class Y2JPEG_Coef(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
       super().__init__(*args,**kwargs)
       self.dct_filters = tf.cast(generate_dct2D_filters(),dtype=self.dtype)
    def get_config(self):
        return {'dct_filters':self.dct_filters}
    def call(self,inputs):
        ydcc_blks = tf.nn.conv2d(inputs,self.dct_filters,strides=8,padding='VALID')
        return ydcc_blks

class JPEG_Coef2Color(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.idct_filters = tf.cast(generate_idct2D_filters(),dtype=self.dtype)
        self.Upsample = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest',dtype=self.dtype)
    def get_config(self):
        return {'idct_filters': self.idct_filters}
    def call(self,inputs):
        ydcc_blks,udcc_blks,vdcc_blks = inputs
        ydcc = tf.nn.depth_to_space(ydcc_blks,block_size=8)
        udcc = tf.nn.depth_to_space(udcc_blks,block_size=8)
        vdcc = tf.nn.depth_to_space(vdcc_blks,block_size=8)
        y_blks = tf.nn.conv2d(ydcc,self.idct_filters,strides=8,padding='VALID')
        u_ds_blks = tf.nn.conv2d(udcc,self.idct_filters,strides=8,padding='VALID')
        v_ds_blks = tf.nn.conv2d(vdcc,self.idct_filters,strides=8,padding='VALID')
        #y = tf.clip_by_value(tf.nn.depth_to_space(y_blks,block_size=8),0.,1.)
        y = tf.nn.depth_to_space(y_blks,block_size=8)
        u_ds = tf.nn.depth_to_space(u_ds_blks,block_size = 8)
        v_ds = tf.nn.depth_to_space(v_ds_blks,block_size = 8)      
        #u = tf.clip_by_value(self.Upsample(u_ds),-0.5,0.5)
        u = self.Upsample(u_ds)
        #v = tf.clip_by_value(self.Upsample(v_ds),-0.5,0.5)
        v = self.Upsample(v_ds)
        yuv = tf.concat((y,u,v),axis=3)
        #return tf.clip_by_value(tf.image.yuv_to_rgb(yuv),0.,1.)
        return self.colortransform(yuv)

class JPEG_Coef2Y(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.idct_filters = tf.cast(generate_idct2D_filters(),dtype=self.dtype)
    def get_config(self):
        return {'idct_filters': self.idct_filters}
    def call(self,inputs):
        ydcc = tf.nn.depth_to_space(inputs,block_size=8)
        y_blks = tf.nn.conv2d(ydcc,self.idct_filters,strides=8,padding='VALID')
        y = tf.nn.depth_to_space(y_blks,block_size=8)
        return y

class RGB2JPEG_Coef(Color2JPEG_Coef):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.colortransform = RGB2YUV(dtype=self.dtype)

class JPEG_Coef2RGB(JPEG_Coef2Color):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.colortransform = YUV2RGB(dtype=self.dtype)

class RGB2JPEG_Coef_Transpose(JPEG_Coef2Color):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.colortransform = RGB2YUV_Transpose(dtype=self.dtype)

class YUV2JPEG_Coef(Color2JPEG_Coef):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.colortransform = tf.identity

class JPEG_Coef2YUV(JPEG_Coef2Color):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.colortransform = tf.identity

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
    def get_config(self):
        return {'masky': self.masky,'masku': self.masku, 'maskv': self.maskv}
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

def h(z,y):
    def grad_fun(grad):
        return (tf.identity(grad),tf.identity(grad))
    return z,grad_fun

@tf.custom_gradient # bad gradient in respect to q, but q shouldn't depend on trainable variable
def _quantize(w,q):
    quantized = q*tf.math.round(w/q)
    nonzero = quantized == tf.cast(0.,w.dtype)
    def grad(g):
        return (tf.where(nonzero,g,g-g),None)
    return quantized,grad

#@tf.custom_gradient
def quantize(w,q,offset=None):
    if offset is None:
        #return (q*tf.math.round(w/q),tf.identity)
        return _quantize(w,q)
    else:
        return _quantize(w - offset,q) + offset
        #return (q*(tf.math.round((w - offset)/q)) + offset,tf.identity)

def threeChannelQuantize(w,qY,qUV,Yoffset):
    return [quantize(w[ii],q,offset) for (ii,q,offset) in zip(range(3),(qY,qUV,qUV),(Yoffset,None,None))]

class Inv_muIpluspBtB(ColorTransform):
    def __init__(self,rho,mu,*args,**kwargs):
        super().__init__(*args,**kwargs)
        rgb2yuv = tf.convert_to_tensor(_rgb_to_yuv_mat,dtype=self.dtype)
        self.transf_mat = tf.linalg.inv(mu*tf.eye(3) + rho*tf.linalg.matmul(transpose(rgb2yuv),rgb2yuv))
        # This idea was never fully realized. Full implementation would likely require Cholesky factorization and a Woodbury identity trick.
        raise NotImplementedError

class INV_muIpluspBtB(tf.keras.layers.Layer):
    def __init__(self,rho,mu,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rho = rho
        self.mu = mu
        self.W = tf.convert_to_tensor(_rgb_to_yuv_mat,dtype=self.dtype)
        inv = tf.linalg.inv(self.rho*tf.eye(3,dtype=self.dtype) + self.mu*tf.linalg.matmul(tf.transpose(self.W),self.W))
        self.transf_mat = tf.Variable(initial_value=inv,trainable=False,dtype= self.dtype)
        # This idea was never fully realized. Full implementation would likely require Cholesky factorization and a Woodbury identity trick.
        raise NotImplementedError
    def _update_fun(self):
        self.transf_mat.assign(tf.linalg.inv(self.rho*tf.eye(3) + self.mu*tf.linalg.matmul(tf.transpose(self.W),self.W)))
    def call(self, inputs):
        transf_mat = tf.reshape(self.transf_mat,shape=tf.concat((tf.ones(tf.rank(inputs) - 1,dtype=tf.as_dtype('int32')),[3,3]),axis=0))
        return tf.linalg.matvec(transf_mat,inputs)
    def get_config(self):
        return{'rho': self.rho, 'rgb2yuv_mat': self.W}


class XUpdate_SmoothJPEG(tf.keras.layers.Layer):
    def __init__(self,lmbda,fftSz,fltr1,fltr2,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lmbda = lmbda
        self.fft = transf.fft2d_multichannel(fftSz)
        self.ifft = transf.ifft2d_multichannel(fftSz)
        self.fltr1 = self.fft(fltr1)
        self.fltr2 = self.fft(fltr2)
    def get_config(self):
        return {'lmbda': self.lmdba}
    def call(self,inputs):
        A = tf.cast(1.0,'complex128') + self.lmbda*(tf.math.conj(self.fltr1)*self.fltr1 + tf.math.conj(self.fltr2)*self.fltr2)
        return self.ifft(self.fft(inputs)/A)
    def last_call(self,inputs):
        A = 1.0 + 3*self.lmbda*(tf.math.conj(self.fltr1)*self.fltr1 + tf.math.conj(self.fltr2)*self.fltr2)
        return self.ifft(self.fft(inputs)/A)

class Relax_SmoothJPEG(tf.keras.layers.Layer): # Since constraint does not depend on x, can just use stepsize alpha.
    def __init__(self,alpha,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.alpha = alpha
        print('Relax_SmoothJPEG is outdated code, and incompatible with other jpeg-related functions.')
    def call(self,inputs):
        QWs,QWz = inputs
        print('Relax_SmoothJPEG is outdated code, and incompatible with other jpeg-related functions.')
        return [-(1.0 - self.alpha)*QWz[channel] - self.alpha*QWs[channel] for channel in range(len(QWs))]
        #return None

class ZUpdate_JPEG(tf.keras.layers.Layer):
    def __init__(self,mu,rho,qY,qUV,W,Wt,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        qntzn_adjstY = QuantizationAdjustment(mu,rho,qY,dtype=self.dtype)
        # Factor of 16 comes from downsampling.
        qntzn_adjstUV = QuantizationAdjustment(mu,rho/16.,qUV,dtype=self.dtype)
        self.qntzn_adjst = [qntzn_adjstY,qntzn_adjstUV,qntzn_adjstUV]
        self.rho = rho
        self.mu = mu
        self.qY = qY
        self.qUV = qUV
        self.W = W
        self.Wt = Wt
    def get_config(self):
        return {'Yoffset': self.Yoffset,'rho': self.rho, 'qY': self.qY, 'qUV': self.qUV}
    def call(self,inputs):
        fx,gamma_over_rho,negC = inputs
        Wx = self.W(fx)
        r = [Wx[channel] + gamma_over_rho[channel] - negC[channel] for channel in range(len(Wx))]
        r_factor = [(self.rho/ds_factor)/(self.mu + self.rho/ds_factor) for ds_factor in (1.,16.,16.)]
        Wdeltaz = [tf.stop_gradient(self.qntzn_adjst[channel]((Wx[channel] + offset,r[channel]))) for (channel,offset) in zip(range(len(Wx)),(self.Yoffset,0.,0.))]
        z = fx - self.Wt([r[channel]*r_factor[channel] for channel in range(len(Wx))]) + self.Wt(Wdeltaz)
        #Wz = [Wx[channel] - self.rho/(self.mu + self.rho)*r[channel] for channel in range(len(Wx))]
        Wz = [Wx[channel] - r_factor[channel]*r[channel] + Wdeltaz[channel] for channel in range(len(Wx))]
        #Wz = [Wx[channel] + self.rho/(self.mu + self.rho)*r[channel] for channel in range(len(Wx))]
        QWz = threeChannelQuantize(Wz,self.qY,self.qUV,self.Yoffset)
        #QWz = Wz
        return (z,QWz)

class GammaUpdate_JPEG(tf.keras.layers.Layer):
    #def __init__(self,alpha,qY,qUV,*args,**kwargs):
    def __init__(self,alpha,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.alpha = alpha
        #self.qY = qY
        #self.qUV = qUV
    def get_config(self):
        return {'qY': self.qy,'qUV': self.qUV}
    def call(self, inputs):
        gamma_over_rho,QWz,negC = inputs
        #cdiff = [Wz[channel] - negC[channel] for channel in range(len(Wz))]
        #return [gamma_over_rho[channel] + self.alpha*tf.math.sign(cdiff[channel])*(tf.abs(cdiff[channel]) - tf.math.maximum(tf.abs(cdiff[channel]),halfqminus(q))) for (channel,q) in zip(range(len(Wz)),(self.qY,self.qUV,self.qUV))]
        return [gamma_over_rho[channel] + self.alpha*(QWz[channel] - negC[channel]) for channel in range(len(QWz))]

class QuantizationAdjustment(tf.keras.layers.Layer):
    def __init__(self,mu,rho,q,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mu = mu
        self.rho = rho
        self.q = q
    def get_config(self):
        return {'q': self.q, 'rho': self.rho}
    def call(self,inputs):
        Wx,r = inputs
        #r_scaled = -self.rho/(self.mu + self.rho)*r
        y = Wx - self.rho/(self.mu + self.rho)*r
        firstTerm = lambda Wdeltaz: self.mu/2*tf.math.square(Wdeltaz - self.rho/(self.mu + self.rho)*r)
        secondTerm = lambda Wdelta: self.rho/2*tf.math.square(self.mu/(self.mu + self.rho)*r + quantize(y + Wdelta,self.q) - y)
        dctelemerror = lambda Wdeltaz:  firstTerm(Wdeltaz) + secondTerm(Wdeltaz)

        firstterm_min = self.rho/(self.mu + self.rho)*r
        secondterm_static = quantize(y,self.q) - y + tf.math.sign(r)*halfqminus(self.q)
        secondterm_move = quantize(y,self.q) - y + tf.math.sign(r)*(self.q + halfqminus(self.q))

        # minimize first term, provided second term unchanged
        candidate1 = tf.math.sign(r)*tf.math.minimum(tf.math.abs(firstterm_min),tf.math.abs(secondterm_static))
        # reduce second term, change first term as little as possible
        candidate2 = quantize(y,self.q) - halfqplus(self.q)*tf.math.sign(y - quantize(y,self.q) + firstterm_min) - y
        # change both terms
        candidate3 = tf.math.sign(r)*tf.math.minimum(tf.math.abs(firstterm_min),tf.math.abs(secondterm_move))

        # compare the results of 3 candidates and choose the best one.
        bestcandidate = tf.where(dctelemerror(candidate2) < dctelemerror(candidate1),candidate2,candidate1)
        bestcandidate = tf.where(dctelemerror(candidate3) < dctelemerror(bestcandidate),candidate3,bestcandidate)
        return bestcandidate


class Enforce_JPEG_Constraint(tf.keras.layers.Layer):
    def __init__(self,qY,qUV,W,Wt,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.qY = qY
        self.qUV = qUV
        self.W = W
        self.Wt = Wt
    def get_config(self):
        return {'qY': self.qY, 'qUV': self.qUV}
    def call(self,inputs):
        fx,negC = inputs
        zero = tf.cast(0.,self.dtype)
        max_value = [negC[channel] + q/2. for (channel,q) in zip(range(len(negC)),(self.qY,self.qUV,self.qUV))]
        min_value = [negC[channel] - q/2. for (channel,q) in zip(range(len(negC)),(self.qY,self.qUV,self.qUV))]
        Wx = self.W(fx)
        delta_value = [tf.where(Wx[channel] > max_value[channel],max_value[channel] - Wx[channel],zero) for channel in range(len(negC))]
        delta_value = [tf.where(Wx[channel] < min_value[channel],min_value[channel] - Wx[channel],delta_value[channel]) for channel in range(len(negC))]
        return self.Wt(delta_value)

class Enforce_JPEGY_Constraint(tf.keras.layers.Layer):
    def __init__(self,qY,W,Wt,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.qY = qY
        self.W = W
        self.Wt = Wt
    def get_config(self):
        return {'qY': self.qY}
    def call(self,inputs):
        fx,negC = inputs
        zero = tf.cast(0.,self.dtype)
        max_value = negC + self.qY/2.
        min_value = negC - self.qY/2.
        Wx = self.W(fx)
        delta_value = tf.where(Wx > max_value,max_value - Wx,zero)
        delta_value = tf.where(Wx < min_value,min_value - Wx,delta_value)
        return self.Wt(delta_value)

class ZUpdate_JPEG_Implicit(tf.keras.layers.Layer):
    def __init__(self,qY,qUV,W,Wt,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.enforce_jpeg_constraint = Enforce_JPEG_Constraint(qY,qUV,W,Wt,*args,**kwargs)
    def call(self,inputs):
        fx,negC = inputs
        delta_z = self.enforce_jpeg_constraint((fx,negC))
        z = fx + delta_z
        delta_z = self.enforce_jpeg_constraint((z,negC)) # for precision
        z = z + delta_z
        return z

class ZUpdate_JPEGY_Implicit(tf.keras.layers.Layer):
    def __init__(self,qY,W,Wt,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.enforce_jpeg_constraint = Enforce_JPEGY_Constraint(qY,W,Wt,*args,**kwargs)
    def call(self,inputs):
        fx,negC = inputs
        delta_z = self.enforce_jpeg_constraint((fx,negC))
        z = fx + delta_z
        delta_z = self.enforce_jpeg_constraint((z,negC)) # for precision
        z = z + delta_z
        return z




class I_dont_know_what_to_call_this:
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def get_TL(self,tl,tr,bl,br):
        ''' When blurring across 4 pixels, this computes the top left result. '''
        return ((self.a**2 + 4*self.a*self.b + 2*self.b**2)*tl/(self.a + 2*self.b) + self.b*tr + self.b*bl + 2*self.b**2*br/(self.a + 2*self.b))/(self.a + 4*self.b)
    def get_TR(self,tl,tr,bl,br):
        ''' Computes the top right result. Assumes TL has already been computed. '''
        return (self.b*(self.a + 2*self.b)**2*tl + self.a*(self.a + self.b)*(self.a + 3*self.b)*tr + self.a*self.b**2*bl + self.a*self.b*(self.a + 2*self.b)*br)/(self.a**3 + 6*self.a**2*self.b + 10*self.a*self.b**2 + 4*self.b**3)
    def get_BL(self,tl,tr,bl,br):
        ''' Computes the bottom left result.  Assumes TL and TR have already been computed. '''
        return (self.b*(self.a + 2*self.b)*tl + self.b**2*tr + self.a*(self.a + 2*self.b)*bl + self.a*self.b*br)/((self.a + self.b)*(self.a + 3*self.b))
    def get_BR(self,tl,tr,bl,br):
        ''' Computes the bottom right result. Assumes TL, TR, and BL have all already been computed. '''
        return (self.b*(tr + bl) + self.a*br)/(self.a + 2*self.b)

def smoothPair(x1,x2,a):
    '''Solution minimizing the equation: (z1 - x1)^2 + (z2 - x2)^2 + a(z1 - z2)^2'''
    z1 = ((1 + a)*x1 + a*x2)/(1 + 2*a)
    z2 = (a*x1 + (1 + a)*x2)/(1 + 2*a)
    return z1,z2
    #return x1,x2

def smoothFour(z1,z2,z3,z4,a):
    '''Solution minimizing the equation: sum_i (yi - xi)^2 + a(y1 - y2)^2 + a(y1 - y3)^2 + a(y2 - y4)^2 + a(y3 - y4)^2
       where z1, z2, z3, and z4 are computed from x1, x2, x3, and x4 using smooth pair.
       temp1,temp2 = jrf.smoothPair(x1,x2,a)
       temp3,temp4 = jrf.smoothPair(x3,x4,a)
       z1,z3 = jrf.smoothPair(temp1,temp3,a)
       z2,z4 = jrf.smoothPair(temp2,temp4,a) '''
    y1 = ((a**2 + 4*a + 1)*z1 - a**2*z2 - a**2*z3 + a**2*z4)/(4*a + 1)
    y2 = (-a**2*z1 + (a**2 + 4*a + 1)*z2 + a**2*z3 - a**2*z4)/(4*a + 1)
    y3 = (-a**2*z1 + a**2*z2 + (a**2 + 4*a + 1)*z3 - a**2*z4)/(4*a + 1)
    y4 = (a**2*z1 - a**2*z2  - a**2*z3 + (a**2 + 4*a + 1)*z4)/(4*a + 1)
    return y1,y2,y3,y4
    #dtype = y1.dtype
    #return z1 + tf.ones(y1.shape,dtype=dtype),z2 + 2.*tf.ones(y2.shape,dtype=dtype),z3 + 3.*tf.ones(y3.shape,dtype=dtype),z4 + 4.*tf.ones(y4.shape,dtype=dtype)


class Smooth_Blk_Edges(tf.keras.layers.Layer):
    def __init__(self,xshape,a,blkSz,*args,**kwargs):
        super().__init__(*args,**kwargs)
        blkshape = tf.zeros(shape = (int(xshape[0]/blkSz[0]),tf.cast(blkSz[0],'int32'),int(xshape[1]/blkSz[1]),tf.cast(blkSz[1],'int32'))).shape
        self.sig_2_blks = tf.keras.layers.Reshape(blkshape + xshape[2:],dtype=self.dtype)
        self.blks_2_sig = tf.keras.layers.Reshape(xshape,dtype=self.dtype)
        self.blkSz = blkSz
        self.a = a
    def call(self,inputs):
        blks = self.sig_2_blks(inputs)

        rows_done = smoothXsect(blks,self.a,axis = 2,blkSz=self.blkSz[0],dtype=self.dtype)

        cols_done = smoothXsect(rows_done,self.a,axis = 4,blkSz = self.blkSz[1],dtype = self.dtype)

        corners_done = smoothCorners(cols_done,self.a,axis1 = 2,axis2 = 4,blkSz = self.blkSz,dtype = self.dtype)

        return self.blks_2_sig(corners_done)
    def get_config(self):
        return {'blkSz': self.blkSz, 'a': self.a}




def smoothXsect(blks,a,axis,blkSz,dtype):
    '''This function applies smoothPair to pairs of cross-sections selected by axis and blkSz.'''
    # select last within block
    slices = [slice(None),]*(axis - 1) + [slice(0,-1),slice(blkSz - 1,None)]
    xsect1 = blks[slices[:]]
    # select 1st within block
    slices = [slice(None),]*(axis - 1) + [slice(1,None),slice(0,1)]
    xsect2 = blks[slices[:]]

    # calculate results
    newXsect1,newXsect2 = smoothPair(xsect1,xsect2,a)

    # grab last of block
    temp1 = tf.one_hot(indices = (blkSz - 1)*tf.ones((1,)*(len(blks.shape) - 1),dtype= 'int32'), depth = blkSz, on_value = 1., off_value = 0.,axis = axis, dtype = dtype)

    # leave out last set of blocks
    slices = [slice(None),]*(axis - 1) + [slice(0,-1),slice(None)]
    withXsect1 = (tf.cast(1.,dtype=dtype) - temp1)*blks[slices[:]] + temp1*newXsect1
    # only last set of blocks
    slices = [slice(None),]*(axis - 1) + [slice(-1,None),slice(None)]
    withXsect1 = tf.concat([withXsect1,blks[slices[:]]],axis = axis - 1)

    # grab first of block
    temp2 = tf.one_hot(indices = tf.zeros((1,)*(len(blks.shape) - 1),dtype='int32'), depth = blkSz, on_value = 1., off_value = 0., axis = axis, dtype = dtype)

    # exclude first set of blocks, grab first of block
    slices = [slice(None),]*(axis - 1) + [slice(1,None),slice(None)]
    withXsect2 = (tf.cast(1.,dtype= dtype) - temp2)*withXsect1[slices[:]] + temp2*newXsect2
    # grab first set of blocks
    slices = [slice(None),]*(axis - 1) + [slice(0,1),slice(None)]
    withXsect2 = tf.concat([withXsect1[slices[:]],withXsect2],axis = axis - 1)

    return withXsect2

def smoothCorners(blks,a,axis1,axis2,blkSz,dtype):
    ''' Welcome to an indexing nightmare.  It's a shame tensorflow doesn't allow for slice-based assignment.'''
    # selects this top left corner
    slices = [slice(None),]*(axis1 - 1) + [slice(0,-1),slice(blkSz[0] - 1,None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(0,-1),slice(blkSz[1] - 1,None)]
    zsect1 = blks[slices[:]]
    # selects the bottom left corner
    slices = [slice(None),]*(axis1 - 1) + [slice(1,None),slice(0,1)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(0,-1),slice(blkSz[1] - 1,None)]
    zsect2 = blks[slices[:]]
    # selects the top right corner
    slices = [slice(None),]*(axis1 - 1) + [slice(0,-1),slice(blkSz[0] - 1,None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(1,None),slice(0,1)]
    zsect3 = blks[slices[:]]
    # selects the bottom right corner
    slices = [slice(None),]*(axis1 - 1) + [slice(1,None),slice(0,1)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(1,None),slice(0,1)]
    zsect4 = blks[slices[:]]


    # Wait, is this right?  Or should 2 and 3 be swapped???
    ysect1,ysect2,ysect3,ysect4 = smoothFour(zsect1,zsect2,zsect3,zsect4,a)
    # select last row
    last_row_select = tf.one_hot(indices = (blkSz[0] - 1)*tf.ones((1,)*(len(blks.shape) - 1),dtype= 'int32'), depth = blkSz[0], on_value = 1., off_value = 0.,axis = axis1, dtype = dtype)
    # select first row
    first_row_select = tf.one_hot(indices = tf.zeros((1,)*(len(blks.shape) - 1),dtype='int32'), depth = blkSz[0], on_value = 1., off_value = 0., axis = axis1, dtype = dtype)
    # select last column
    last_col_select = tf.one_hot(indices = (blkSz[1] - 1)*tf.ones((1,)*(len(blks.shape) - 1),dtype= 'int32'), depth = blkSz[1], on_value = 1., off_value = 0.,axis = axis2, dtype = dtype)
    # select first column
    first_col_select = tf.one_hot(indices = tf.zeros((1,)*(len(blks.shape) - 1),dtype='int32'), depth = blkSz[1], on_value = 1., off_value = 0., axis = axis2, dtype = dtype)


    # **** ADDING TOP LEFT ****
    # select last column of all but last row and last column of blocks
    slices = [slice(None),]*(axis1 - 1) + [slice(0,-1),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(0,-1),slice(blkSz[1] - 1,None)]
    last_col = (tf.cast(1.,dtype) - last_row_select)*blks[slices[:]] + ysect1*last_row_select
    # select all except last column of all but last row and last column of blocks.
    slices = [slice(None),]*(axis1 - 1) + [slice(0,-1),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(0,-1),slice(0,-1)]

    temp = tf.concat([blks[slices[:]],last_col],axis2)
    # select bottom row of blocks (excluding last column of blocks)
    slices1 = [slice(None),]*(axis1 - 1) + [slice(-1,None),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(0,-1),slice(None)]
    # select last column of blocks
    slices2 = [slice(None),]*(axis1 - 1) + [slice(None),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(-1,None),slice(None)]
    include_tl = tf.concat([tf.concat([temp,blks[slices1[:]]], axis = axis1 - 1),blks[slices2[:]]],axis = axis2 - 1)


    # **** ADDING BOTTOM LEFT ****
    # select last column of all but first row and last column of blocks
    slices = [slice(None),]*(axis1 - 1) + [slice(1,None),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(0,-1),slice(blkSz[1] - 1,None)]
    last_col = (tf.cast(1.,dtype) - first_row_select)*include_tl[slices[:]] + ysect2*first_row_select

    slices = [slice(None),]*(axis1 - 1) + [slice(1,None),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(0,-1),slice(0,-1)]
    temp = tf.concat([include_tl[slices[:]],last_col],axis2)

    # select top row of blocks (excluding last column of blocks)
    slices1 = [slice(None),]*(axis1 - 1) + [slice(0,1),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(0,-1),slice(None)]
    # select last column of blocks
    slices2 = [slice(None),]*(axis1 - 1) + [slice(None),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(-1,None),slice(None)]
    include_bl = tf.concat([tf.concat([include_tl[slices1[:]],temp], axis = axis1 - 1),include_tl[slices2[:]]],axis = axis2 - 1)


    # **** ADDING TOP Right ****
    # select first column of all but last row and first column of blocks
    slices = [slice(None),]*(axis1 - 1) + [slice(0,-1),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(1,None),slice(0,1)]
    first_col = (tf.cast(1.,dtype) - last_row_select)*include_bl[slices[:]] + ysect3*last_row_select

    slices = [slice(None),]*(axis1 - 1) + [slice(0,-1),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(1,None),slice(1,None)]
    temp = tf.concat([first_col,include_bl[slices[:]]],axis2)

    # select bottom row of blocks (excluding first column of blocks)
    slices1 = [slice(None),]*(axis1 - 1) + [slice(-1,None),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(1,None),slice(None)]
    # select first column of blocks
    slices2 = [slice(None),]*(axis1 - 1) + [slice(None),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(0,1),slice(None)]
    include_tr = tf.concat([include_bl[slices2[:]],tf.concat([temp,include_bl[slices1[:]]], axis = axis1 - 1)],axis = axis2 - 1)

    # **** ADDING BOTTOM RIGHT ****
    # select first column of all but first row and first column of blocks
    slices = [slice(None),]*(axis1 - 1) + [slice(1,None),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(1,None),slice(0,1)]
    first_col = (tf.cast(1.,dtype) - first_row_select)*include_tr[slices[:]] + ysect4*first_row_select

    slices = [slice(None),]*(axis1 - 1) + [slice(1,None),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(1,None),slice(1,None)]
    temp = tf.concat([first_col,include_tr[slices[:]]],axis2)

    # select top row of blocks (excluding first column of blocks)
    slices1 = [slice(None),]*(axis1 - 1) + [slice(0,1),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(1,None),slice(None)]
    # select first column of blocks
    slices2 = [slice(None),]*(axis1 - 1) + [slice(None),slice(None)] + [slice(None),]*(axis2 - axis1 - 2) + [slice(0,1),slice(None)]
    include_br = tf.concat([include_tr[slices2[:]],tf.concat([include_tr[slices1[:]],temp], axis = axis1 - 1)],axis = axis2 - 1)

    return include_br

def XsectDiff(blks,axis,blkSz,dtype):
    '''This function applies smoothPair to pairs of cross-sections selected by axis and blkSz.'''
    # select last within block
    slices = [slice(None),]*(axis - 1) + [slice(0,-1),slice(blkSz - 1,None)]
    xsect1 = blks[slices[:]]
    # select 1st within block
    slices = [slice(None),]*(axis - 1) + [slice(1,None),slice(0,1)]
    xsect2 = blks[slices[:]]

    return xsect1 - xsect2

class Measure_Blk_Edges(tf.keras.layers.Layer):
    def __init__(self,xshape,blkSz,*args,**kwargs):
        super().__init__(*args,**kwargs)
        blkshape = (int(xshape[0]/blkSz[0]),tf.cast(blkSz[0],'int32'),int(xshape[1]/blkSz[1]),tf.cast(blkSz[1],'int32'))
        self.sig_2_blks = tf.keras.layers.Reshape(blkshape + xshape[2:],dtype=self.dtype)
        self.blkSz = blkSz
    def call(self,inputs):
        blks = self.sig_2_blks(inputs)
        row_diff = XsectDiff(blks,axis = 2,blkSz=self.blkSz[0],dtype=self.dtype)

        col_diff = XsectDiff(blks,axis = 4,blkSz = self.blkSz[1],dtype = self.dtype)

        return tf.reduce_sum(row_diff**2) + tf.reduce_sum(col_diff**2)
    def get_config(self):
        return {'blkSz': self.blkSz}
