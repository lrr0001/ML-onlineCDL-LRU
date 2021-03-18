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

class ZUpdate_JPEG(tf.keras.layers.Layer):
    def __init__(self,mu,rho,qY,qUV,W,Wt,*args,**kwargs):
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
    def call(self,inputs):
        fx,Axminuss,gamma_over_rho = inputs
        Wx = self.W(fx)
        r = [-Axminuss[channel] - Wx[channel] - gamma_over_rho[channel] for channel in range(len(Wx))]
        Wdeltaz = [self.qntzn_adjst[channel]((Wx[channel] + offset,r[channel])) for (channel,offset) in zip(range(len(Wx)),(self.Yoffset,0.,0.))]
        z = fx + self.rho/(self.mu + self.rho)*self.Wt(r) + self.Wt(Wdeltaz)
        Wz = [Wx[channel] + self.rho/(self.mu + self.rho)*r[channel] + Wdeltaz[channel] for channel in range(len(Wx))]
        Wz = self.W(z)
        QWz = threeChannelQuantize(Wz,self.qY,self.qUV,self.Yoffset)
        return (z,QWz)

class QuantizationAdjustment(tf.keras.layers.Layer):
    def __init__(self,mu,rho,q,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mu = mu
        self.rho = rho
        self.q = q
    def call(self,inputs):
        Wx,r = inputs
        r_scaled = self.rho/(self.mu + self.rho)*r
        y = Wx + r_scaled
        firstTerm = lambda Wdeltaz: self.mu/2*tf.math.square(r_scaled + Wdeltaz)
        secondTerm = lambda Wdeltaz: self.rho/2*tf.math.square(-self.mu/(self.mu + self.rho) + self.q*tf.math.round((y + Wdeltaz)/self.q) - y)
        dctelemerror = lambda Wdeltaz:  firstTerm(Wdeltaz) + secondTerm(Wdeltaz)

        firstterm_min = -r_scaled
        secondterm_static = quantize(y,self.q) - y - tf.math.sign(r)*self.q*jrf.halfqminus(self.q)
        secondterm_move = quantize(y,self.q) - y - tf.math.sign(r)*self.q*(1. + jrf.halfqminus(self.q))

        # minimize first term, provided second term unchanged
        candidate1 = -tf.math.sign(r)*tf.math.minimum(tf.math.abs(firstterm_min),tf.math.abs(secondterm_static))
        # reduce second term, change first term as little as possible
        candidate2 = quantize(y,self.q) - jrf.halfqplus(self.q)*self.q*tf.math.sign(y - quantize(y,self.q) + firstterm_min) - y
        # change both terms
        candidate3 = -tf.math.sign(r)*tf.math.minimum(tf.math.abs(firstterm_min),tf.math.abs(secondterm_move))

        # compare the results of 3 candidates and choose the best one.
        bestcandidate = tf.where(dctelemerror(candidate2) < dctelemerror(candidate1),candidate2,candidate1)
        bestcandidate = tf.where(dctelemerror(candidate3) < dctelemerror(bestcandidate),candidate3,bestcandidate)
        return bestcandidate


class Smooth_JPEG_Layer(jrf.Smooth_JPEG):
    def init_fun(self):
        self.W = jrf.RGB2JPEG_Coef(dtype=self.dtype)
        self.Wt = jrf.JPEG_Coef2RGB(dtype=self.dtype)
        self.xupdate = XUpdate(self.lmbda,self.fftSz,tf.reshape(self.fltr,(1,2,1,1)),tf.reshape(self.fltr,(1,1,2,1)),dtype = self.dtype)
        self.relaxlayer = jrf.Relax_SmoothJPEG(self.alpha,dtype=self.dtype)
        self.yupdate = ZUpdate_JPEG(1.0,self.rho,self.qY,self.qUV,self.W,self.Wt,dtype=self.dtype)
        self.uupdate = jrf.GammaUpdate_JPEG(dtype=self.dtype)
    def get_output(self,s,y,u,By,negC,itstats):
        x,Ax = self.xstep(y,u,By,negC)
        return (x,negC)

@tf.custom_gradient # bad gradient in respect to q, but q shouldn't depend on trainable variable
def _quantize(w,q):
    def grad(g):
        return tf.identity(g),None
    return q*tf.math.round(w/q),grad
    #return w

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
