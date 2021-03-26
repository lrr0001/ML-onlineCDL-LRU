import tensorflow as tf
import math
import numpy as np
import transforms as transf
import optmz

class Smooth_JPEG(optmz.ADMM):
    ''' This layer computes a smoothed version of a JPEG-compressed image. Input is an uncompressed RGB image.
        Output is a smoothed version of the image in YUV domain, a JPEG-compressed YUV image, and an uncompressed YUV image.''' 
    def __init__(self,rho,alpha,noi,qY,qUV,lmbda,fftSz,*args,**kwargs):
        self.qY = tf.reshape(qY,(1,1,1,64))
        self.qUV = tf.reshape(qUV,(1,1,1,64))
        self.lmbda = lmbda
        self.fftSz = fftSz
        self.fltr = np.asarray([-1.,1.])
        super().__init__(rho,alpha,noi,*args,**kwargs)
        self.init_fun()
    def init_fun(self):
        self.W = YUV2JPEG_Coef(dtype=self.dtype)
        self.Wt = JPEG_Coef2YUV(dtype=self.dtype)
        self.xupdate = XUpdate_SmoothJPEG(self.lmbda,self.fftSz,tf.reshape(self.fltr,(1,2,1,1)),tf.reshape(self.fltr,(1,1,2,1)),dtype = self.dtype)
        #self.relaxlayer = Relax_SmoothJPEG(dtype=self.dtype) # move alpha to uupdate
        self.yupdate = ZUpdate_JPEG(1.0,self.rho,self.qY,self.qUV,self.W,self.Wt,dtype=self.dtype)
        self.uupdate = GammaUpdate_JPEG(self.alpha,dtype=self.dtype)

    # These initializations happen once per input (negC,y,By,u):
    def init_x(self,s,negC):
        return (None,None)
    def init_y(self,s,x,Ax,negC):
        return (self.Wt(negC),negC)
    def init_u(self,s,Ax,By,negC):
        return [0.,0.,0.]
    def get_negative_C(self,s):
        Ws = self.W(s)
        Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        return threeChannelQuantize(Ws,self.qY,self.qUV,Yoffset)
    def init_itstats(self,s):
        return []


    # iterative steps:
    def xstep(self,y,u,By,negC):
        return (self.xupdate(y),0.)
    def relax(self,Ax,By,negC):
        #return self.relaxlayer((negC,By))
        return (None,)
    def ystep(self,x,u,Ax_relaxed,negC):
        return self.yupdate((x,u,negC))
    def ustep(self,u,By,negC):
        return self.uupdate((u,By,negC))

    # Before and After:
    def preprocessing(s)
        rgb2yuv = RGB2YUV(dtype=self.dtype)
        return rgb2yuv(s)
    def get_output(self,s,y,u,By,negC,itstats):
        ''' Outputs:
               Smoothed image (YUV)
               Compressed image (YUV)
               Raw image (YUV)'''
        x,Ax = self.xstep(y,u,By,negC)
        return (x,self.Wt(negC),s)

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
    def grad(g):
        return (tf.identity(g),None)
    return q*tf.math.round(w/q),grad

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
        A = 1.0 + self.lmbda*(tf.math.conj(self.fltr1)*self.fltr1 + tf.math.conj(self.fltr2)*self.fltr2)
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
    def __init__(self,mu,rho,inv_IplusWtW,qY,qUV,W,Wt,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        qntzn_adjstY = QuantizationAdjustment(mu,rho,qY,dtype=self.dtype)
        qntzn_adjstUV = QuantizationAdjustment(mu,rho,qUV,dtype=self.dtype)
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
        Wdeltaz = [self.qntzn_adjst[channel]((Wx[channel] + offset,r[channel])) for (channel,offset) in zip(range(len(Wx)),(self.Yoffset,0.,0.))]
        z = fx - self.rho/(self.mu + self.rho)*self.Wt(r)# + self.Wt(Wdeltaz)
        #Wz = [Wx[channel] - self.rho/(self.mu + self.rho)*r[channel] for channel in range(len(Wx))]
        Wz = [Wx[channel] - self.rho/(self.mu + self.rho)*r[channel] + Wdeltaz[channel] for channel in range(len(Wx))]
        #Wz = [Wx[channel] + self.rho/(self.mu + self.rho)*r[channel] for channel in range(len(Wx))]
        QWz = threeChannelQuantize(Wz,self.qY,self.qUV,self.Yoffset)
        #QWz = Wz
        return (z,QWz)

class GammaUpdate_JPEG(tf.keras.layers.Layer):
    def __init__(self,alpha,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.alpha = alpha
    def call(self, inputs):
        gamma_over_rho,QWz,negC = inputs
        return [gamma_over_rho[channel] + alpha*(QWz[channel] - negC[channel]) for channel in range(len(QWz))]

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
        r_scaled = -self.rho/(self.mu + self.rho)*r
        y = Wx + r_scaled
        firstTerm = lambda Wdeltaz: self.mu/2*tf.math.square(r_scaled + Wdeltaz)
        secondTerm = lambda Wdeltaz: self.rho/2*tf.math.square(-self.mu/(self.mu + self.rho) + self.q*tf.math.round((y + Wdeltaz)/self.q) - y)
        dctelemerror = lambda Wdeltaz:  firstTerm(Wdeltaz) + secondTerm(Wdeltaz)

        firstterm_min = -r_scaled
        secondterm_static = quantize(y,self.q) - y - tf.math.sign(r)*self.q*halfqminus(self.q)
        secondterm_move = quantize(y,self.q) - y - tf.math.sign(r)*self.q*(1. + halfqminus(self.q))

        # minimize first term, provided second term unchanged
        candidate1 = -tf.math.sign(r)*tf.math.minimum(tf.math.abs(firstterm_min),tf.math.abs(secondterm_static))
        # reduce second term, change first term as little as possible
        candidate2 = quantize(y,self.q) - halfqplus(self.q)*self.q*tf.math.sign(y - quantize(y,self.q) + firstterm_min) - y
        # change both terms
        candidate3 = -tf.math.sign(r)*tf.math.minimum(tf.math.abs(firstterm_min),tf.math.abs(secondterm_move))

        # compare the results of 3 candidates and choose the best one.
        bestcandidate = tf.where(dctelemerror(candidate2) < dctelemerror(candidate1),candidate2,candidate1)
        bestcandidate = tf.where(dctelemerror(candidate3) < dctelemerror(bestcandidate),candidate3,bestcandidate)
        return bestcandidate

