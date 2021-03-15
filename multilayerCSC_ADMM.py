import tensorflow as tf
import matrix_decompositions_tf as fctr
import optmz
import jpeg_related_functions as jrf
import transforms as transf
import util
import numpy as np

class MultiLayerCSC(optmz.ADMM):
    '''
    x: list
       x[\ell] = \mR_{\ell}^{-1}\vx_{\ell}
    y: list
        v
        z: list
            z[\ell]: \vz_{\ell}
            z[L - 1]: \mR_{L - 1}^{-1}\vz_{L - 1}
    u: list
        eta
        gamma: list
            gamma[\ell] = \vgamma_{\ell}/\rho\sqrt{\mu_{\ell}}
    Ax: list
        Azero
        Ax: list
            Ax[\ell]
    By: list
        Bv
        Bz: list
            Bz[\ell] 
    negC: list
        s_LF
        QWs
    '''
    def __init__(self,rho,alpha_init,mu_init,qY,qUV,cropAndMerge,fftSz,D,lraParam,noi,noL,cmplxdtype,*args,**kwargs):
        self.cmplxdtype = cmplxdtype
        dtype = cmplxdtype.real_dtype
        rho = tf.cast(rho,dtype = cmplxdtype.real_dtype)
        mu_init = tf.cast(mu_init,dtype=cmplxdtype.real_dtype)
        self.noL = noL
        super().__init__(rho = rho,alpha=alpha_init,noi = noi,dtype=dtype,*args,**kwargs)
        self.alpha = tf.Variable(initial_value=alpha_init,trainable=True,dtype=dtype)
        qY = tf.reshape(qY,(1,1,1,64))
        qUV = tf.reshape(qUV,(1,1,1,64))
        self.cropAndMerge = cropAndMerge
        self.initializeLayers(rho,mu_init,qY,qUV,noL,fftSz,D,lraParam,cmplxdtype)
    def initializeLayers(self,rho,mu_init,qY,qUV,noL,fftSz,D,lraParam,cmplxdtype):
        self.dictObj = []
        self.updateX_layer = []
        self.FFT = []
        self.IFFT = []
        self.FFT_factor = [] 
        for ii in range(noL):
            if ii == 0:
                self.dictObj.append(fctr.dictionary_object2D_full_init(fftSz=fftSz[ii],D[ii],rho=tf.cast(rho,dtype=cmplxdtype),name='dict_layer' + str(ii),lraParam = lraParam,dtype=cmplxdtype))
            else:
                self.dictObj.append(fctr.dictionary_object2D_init(fftSz=fftSz[ii],D=D[ii],rho=tf.cast(rho,dtype=cmplxdtype),name='dict_layer' + str(ii),lraParam = lraParam,dtype=cmplxdtype))
            self.updateX_layer.append(GetNextIterX(tf.cast(rho,dtype=cmplxdtype),self.dictObj[ii],dtype=cmplxdtype))
            self.FFT.append(transf.fft2d_inner(fftSz[ii]))
            self.IFFT.append(transf.ifft2d_inner(fftSz[ii]))
            self.FFT_factor.append(np.prod(fftSz[ii]))
       

        self.relax_layer = GetRelaxedAx(alpha=tf.cast(self.alpha,dtype=cmplxdtype),dtype=cmplxdtype)
        self.updateZ_layer = [[],]*(noL - 1)
        self.updateGamma_layer = GetNextIterGamma(dtype=cmplxdtype)
        self.updateZ_lastlayer = GetNextIterZ_lastlayer(rho,mu_init,self.dictObj[noL - 1],tf.zeros(shape = self.get_z_shape(self.fftSz[noL - 1],D[noL - 1].shape[-1],dtype=cmplxtype.real_dtype),dtype=cmplxdtype.real_dtype)
        mu = self.updateZ_lastlayer.mu
        for ii in range(noL - 2,-1,-1):
            self.updateZ_layer[ii] = GetNextIterZ(rho,mu_init,mu,self.dictObj[ii],self.dictObj[ii + 1],tf.zeros(shape = self.get_z_shape(self.fftSz[ii],D[ii].shape[-1],dtype=cmplxtype.real_dtype),dtype=cmplxdtype.real_dtype)
            mu = self.updateZ_layer[ii].mu
        W = jrf.RGB2JPEG_Coef(dtype=cmplxdtype.real_dtype)
        Wt = jrf.JPEG_Coef2RGB(dtype=cmplxdtype.real_dtype)
        self.updatev = jrf.ZUpdate_JPEG(mu,rho,qY,qUV,W,Wt,dtype = cmplxdtype.real_dtype)
        self.relax0 = jrf.Relax_SmoothJPEG(self.alpha, dtype=cmplxdtype.real_dtype)
        self.updateeta = jrf.GammaUpdate_JPEG(dtype=cmplxdtype.real_dtype)

    def get_output(self,s,y,u,By,negC,itstats):
        x,Ax = self.xstep(y,u,By,negC)
        s_LF,QWs = negC
        return (self.cropAndMerge.crop(self.IFFT[0](x[0])) + s_LF,itstats)

    def get_z_shape(self,fftSz,M):
        return (1) + fftSz + (M,) + (1,)

    # High-level Initialization Functions (x,y,u,Ax,By,C)
    def get_negative_C(self,s):
        s_LF,s_HF,Ws = s
        Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        QWs = jrf.threeChannelQuantize(Ws,self.qY,self.qUV,Yoffset)
        return (s_LF,QWs)
    def init_x(self,s,negC):
        s_LF,s_HF,QWs = s
        x = []
        x.append(self.xinit(self.FFT[0](util.addDim(s_HF)),layer = 0))
        for ii in range(1,self.noL):
            x.append(self.xinit(x[ii - 1],layer = ii))
        Ax = ([-QWs[layer] for layer in range(len(QWs))],x)
        return (x,Ax)
    def init_y(self,s,x,Ax,negC):
        Azero,Ax_layers = Ax
        s_LF,QWs = negC
        v,Bv = self.vinit(x[0],Azero,s_LF)
        z = []
        for ii in range(self.noL - 1):
            z.append(self.zinit(xnext=x[ii + 1],Ax=Ax_layers[ii],layer=ii))
        z.append(self.zinit_last(Ax=Ax_layers[self.noL - 1]))
        y = (v,z)
        By = (Bv,z)
        return (y,By)
    def init_u(self,s,Ax,By,negC):
        Azero,Ax_layers = Ax
        Bv,Bz = By
        gamma = []
        for ii in range(self.noL):
            gamma.append(self.gammainit(Ax_layers[ii],Bz[ii],ii))
        eta = self.etainit(Azero,Bv)
        return (eta,gamma)


    # High-level Update Functions for each iteration (x,y,u,Ax,By)
    def xstep(self,y,u,By,negC):
        v,z = y
        eta,gamma = u
        x = []
        x.append(self.updateX(z_prevlayer=v,z=z[0],gamma_scaled = gamma[0],layer=0))
        for ii in range(1,self.noL):
            x.append(self.updateX(z_prevlayer=z[ii - 1],z = z[ii],gamma_scaled = gamma[ii],layer=ii))
        return (x,(0.,x))
    def relax(self,Ax,By,negC):
        Azero,x = Ax
        s_LF,QWs = negC
        rootmux = []
        Bv,z = By
        for ii in range(self.noL):
            rootmux.append(self.relax_layers(x[ii],z[ii],ii))
        AzeroplusC = self.relax_zero(Bv,QWs)
        return (AzeroplusC,rootmux)
    def ystep(self,x,u,Ax_relaxed,negC):
        eta,gamma = u
        AzeroplusC,Ax_layers = Ax_relaxed
        s_LF,QWs = negC
        z = []
        for ii in range(self.noL - 1):
            z.append(self.updateZ(x_nextlayer=x[ii + 1],Ax_relaxed=Ax_layers[ii],gamma_scaled=gamma[ii],layer=ii))
        z.append(self.updateZ_last(Ax_relaxed=Ax_layers[self.noL - 1],gamma_scaled=gamma[self.noL - 1]))
        v,Bv = self.updateV(x[0],AzeroplusC,eta,s_LF)
        y = (v,z)
        By = (Bv,z)
        return (y,By)
    def ustep(self,u,Ax_relaxed,By,negC):
        eta,gamma = u
        AzeroplusC,Ax_layers = Ax_relaxed
        Bv,z = By
        for ii in range(self.noL):
            gamma[ii] = self.updateGamma(gamma[ii],z[ii],Ax_layers[ii],ii)
        eta = self.updateEta(eta,AzeroplusC,Bv)
        return (eta,gamma)

    #High-level Langrangian Evalaluation Functions
    def evaluateLagrangian(self,x,y,u,By,negC):
        recErr = self.reconstructionErrors(x,y)
        penaltyErr = self.penaltyErrors(y)
        cnstrErr = self.constraintErrors(x,u,By,negC)
        return recErr + penaltyErr + cnstrErr
    def evaluateRelaxedLagrangian(self,x,y,u,Ax,By,negC):
        recErr = self.reconstructionErrors(x,y)
        penaltyErr = self.penaltyErrors(y)
        cnstrErr = self.constraintErrors_relaxed(u,Ax,By)
        return recErr + penaltyErr + cnstrErr
    def reconstructionErrors(self,x,y):
        v,z = y
        mu = self.updateZ_layer[0].mu
        reconErr = (mu/2)*self.reconstructionTerm(v,self.dictObj[0].dmul(x[0]))/self.FFT_factor[0]
        for layer in range(1,self.noL):
            if layer < self.noL - 1:
                mu = self.updateZ_layer[layer].mu
            else:
                mu = self.updateZ_lastlayer.mu
            reconErr += (mu/2)*self.reconstructionTerm(z[layer - 1],self.dictObj[layer].dmul(x[layer]))/self.FFT_factor[layer]
        return reconErr
    def penaltyErrors(self,y):
        v,z = y
        penaltyErr = 0
        for layer in range(self.noL):
            penaltyErr += self.penaltyTerm(self.IFFT[layer](z[layer]),layer)
        return penaltyErr
    def constraintErrors(self,x,u,By,negC):
        s_LF,QWs = negC
        Bv,Bz = By
        eta,gamma = u
        constraintErr = self.jpegConstraint(eta,Bv,QWs)
        for layer in range(self.noL):
            if layer < self.noL - 1:
                mu = self.updateZ_layer[layer].mu
                z_over_R = Bz[layer]/util.complexNum(util.rotate_dims_left(self.dictObj[layer].R,5))
            else:
                mu = self.updateZ_lastlayer.mu
                z_over_R = Bz[layer]
            constraintErr += (mu/2)*self.zxConstraint(gamma[layer],x[layer],z_over_R)/self.FFT_factor[layer]
        return constraintErr
    def constraintErrors_relaxed(self,u,Ax,By):
        Azero,Ax = Ax
        Bv,Bz = By
        eta,gamma = u
        constraintErr = self.jpegConstraint_relaxed(eta,Azero,Bv)
        for layer in range(self.noL):
            if layer < self.noL - 1:
                mu = self.updateZ_layer[layer].mu
                Bz_over_R = Bz[layer]/util.complexNum(util.rotate_dims_left(self.dictObj[layer].R,5))
            else:
                mu = self.updateZ_lastlayer.mu
                Bz_over_R = Bz[layer]
            constraintErr += (mu/2)*self.zxConstraint_relaxed(gamma[layer],Ax[layer],Bz_over_R)/self.FFT_factor[layer]
        return constraintErr

    # Low-level Initialization Functions (x[layer],v,z[layer],eta,gamma[layer],Azero,Ax[layer],Bv,Bz[layer])
    def xinit(self,xprev,layer):
        if layer == 0:
            Dhx = self.dictObj[layer].dhmul(xprev)
        else:
            Rprev = util.rotate_dims_left(self.dictObj[layer - 1].R,5)
            Dhx = self.dictObj[layer].dhmul(xprev*tf.cast(Rprev,self.cmplxdtype))
        Rsquared = util.rotate_dims_left(tf.math.square(self.dictObj[layer].R),5)
        return Dhx/tf.cast(Rsquared,self.cmplxdtype)
    def vinit(self,x_0,Azero,s_LF):
        Dx = self.IFFT[0](self.dictObj[0].dmul(x_0))
        Dx = tf.reshape(Dx,Dx.shape[:-1])
        croppedDx = self.cropAndMerge.crop(Dx)
        vpluss_LF,Bv = self.updatev((croppedDx + s_LF,Azero,[0.,0.,0.]))
        vpremerge = vpluss_LF - s_LF
        v = self.cropAndMerge.merge((vpremerge,Dx))
        v = util.addDim(v)
        return (self.FFT[0](v),Bv) 
    def zinit(self,xnext,Ax,layer):     
        return self.updateZ(xnext,Ax,tf.reshape(tf.cast(0.,tf.as_dtype(self.cmplxdtype)),(1,1,1,1,1)),layer)
    def zinit_last(self,Ax):
        return self.updateZ_last(Ax,tf.reshape(tf.cast(0.,tf.as_dtype(self.cmplxdtype)),(1,1,1,1,1)))
    def gammainit(self,Ax,Bz,layer):
        return self.updateGamma(tf.cast(0.,tf.as_dtype(self.cmplxdtype)),Bz,Ax,layer)
    def etainit(self,Azero,Bv):
        return self.updateEta([0.,0.,0.],Azero,Bv)


    # Low-level Update Functions (x[layer],v,z[layer],eta,gamma[layer],Azero,Ax[layer],Bv,Bz[layer])
    def updateX(self,z_prevlayer,z,gamma_scaled,layer):
        if layer == self.noL - 1:
            return self.updateX_layer[layer]((z_prevlayer,z,gamma_scaled))
        else:
            R = tf.cast(util.rotate_dims_left(self.dictObj[layer].R,5),dtype=self.cmplxdtype)
            return self.updateX_layer[layer]((z_prevlayer,z/R,gamma_scaled))
    def relax_layers(self,x,z,layer):
        if layer < self.noL - 1:
            R = tf.cast(util.rotate_dims_left(self.dictObj[layer].R,5),dtype=self.cmplxdtype)
            z_over_R = z/R
        else:
            z_over_R = z
        return self.relax_layer((x,z_over_R))
    def relax_zero(self,Bv,negC):
        return self.relax0((negC,Bv))
    def updateV(self,x_0,Azero,eta_over_rho,s_LF):
        Dx = self.IFFT[0](self.dictObj[0].dmul(x_0))
        # It might be helpful to build an object to couple cropping and padding so that it is never messed up. Best if this is done outside, because the cropping function can be used to get QWs.
        Dx = tf.reshape(Dx,Dx.shape[:-1]) # add tf.crop_to_bounding_box here.
        vpluss_LF,Bv = self.updatev((Dx + s_LF,Azero,eta_over_rho))
        v = vpluss_LF - s_LF
        v = util.addDim(v)
        return (self.FFT[0](v),Bv) # Need to use tf.pad and tf.where to extend v. Formula for padding: sum_l (prod_i from 1 to l stride_i)(kernel_size_l - 1) + whatever necessary to make divisible by the strides.
    def updateZ(self,x_nextlayer,Ax_relaxed,gamma_scaled,layer):
        assert(layer < self.noL - 1)
        Dx = self.dictObj[layer + 1].dmul(x_nextlayer)
        z = self.updateZ_layer[layer]((self.IFFT[layer](Dx),self.IFFT[layer](Ax_relaxed),self.IFFT[layer](gamma_scaled)))
        return self.FFT[layer](z)
    def updateZ_last(self,Ax_relaxed,gamma_scaled):
        z = self.updateZ_lastlayer((self.IFFT[self.noL - 1](Ax_relaxed),self.IFFT[self.noL - 1](gamma_scaled)))
        return self.FFT[self.noL - 1](z)
    def updateGamma(self,gamma_scaled,z,Ax_relaxed,layer):
        if layer < self.noL - 1:
            z_over_R = z/util.complexNum(util.rotate_dims_left(self.dictObj[layer].R,5))
        else:
            z_over_R = z
        return self.updateGamma_layer((gamma_scaled,z_over_R,Ax_relaxed))
    def updateEta(self,eta_over_rho,AzeroplusC,Bv):
        return self.updateeta((eta_over_rho,AzeroplusC,Bv))


    # Low-level augmented Langrangian evaluation:
    def reconstructionTerm(self,z,Dx):
        zminusDx = z - Dx
        output = tf.reduce_sum(tf.math.conj(zminusDx)*zminusDx)
        return tf.cast(output,output.dtype.real_dtype)
    def nonnegativeCheck(self,z):
        assert(tf.all(z >= 0.))

    def penaltyTerm(self,z,layer):
        if layer < self.noL - 1:
            return tf.math.reduce_sum(tf.math.abs(self.updateZ_layer[layer].b*z))
        else:
            return tf.math.reduce_sum(tf.math.abs(self.updateZ_lastlayer.b*tf.math.maximum(z,0.)))/self.rho/self.updateZ_lastlayer.mu
    def jpegConstraint(self,eta_over_rho,QWz,QWs):
        aug_cnstrnt_term = [QWz[channel] - QWs[channel] + eta_over_rho[channel] for channel in range(len(QWs))]
        return self.rho/2*sum([tf.math.reduce_sum(aug_cnstrnt_term[channel]*tf.math.conj(aug_cnstrnt_term[channel])) for channel in range(len(QWs))])
    def jpegConstraint_relaxed(self,eta_over_rho,Azero,Bv):
        aug_cnstrnt_term = [Azero[channel] + Bv[channel] + eta_over_rho[channel] for channel in range(len(Bv))]
        return self.rho/2*sum([tf.math.reduce_sum(aug_cnstrnt_term[channel]*tf.math.conj(aug_cnstrnt_term[channel])) for channel in range(len(Bv))])
    def zxConstraint(self,gamma_over_rho,x_over_R,z_over_R):
        rho = util.complexNum(self.rho)
        sum_of_terms = z_over_R - x_over_R + gamma_over_rho
        output = rho*tf.reduce_sum(tf.math.conj(sum_of_terms)*sum_of_terms)
        return tf.cast(output,output.dtype.real_dtype)
    def zxConstraint_relaxed(self,gamma_over_rho,Ax_relaxed,Bz):
        rho = util.complexNum(self.rho)
        sum_of_terms = Ax_relaxed + Bz + gamma_over_rho
        output = rho*tf.reduce_sum(tf.math.conj(sum_of_terms)*sum_of_terms)
        return tf.cast(output,output.dtype.real_dtype)



class GetNextIterX(tf.keras.layers.Layer):
    '''
      inputs: All must be in frequency domain.
    
        z_previouslayer: $\vz_{\ell - 1}^{(k)}$
        z_over_R: $\mR^{-1}\vz_{\ell}^{(k)}$
        gamma_scaled: $\frac{\vgamma_{\ell}^{(k)}}{\rho\sqrt{\mu_{\ell}}$

      outputs: Also in frequency domain
        x_scaled: \mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)}
    '''
    def __init__(self,rho,dictObj,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dictObj = dictObj
        self.rho = rho
    def call(self,inputs):
        z_prevlayer,z_over_R,gamma_scaled = inputs
        return self.dictObj.qinv(self.dictObj.dhmul(z_prevlayer) + self.rho*(z_over_R + gamma_scaled))

class GetRelaxedAx(tf.keras.layers.Layer):
    '''
      inputs: All must be in frequency domain

        x_over_R: $\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)}$
        z_over_R: $\mR_{\ell}^{-1}\vz_{\ell}^{(k)}$

      outputs: Also in frequency domain
        Ax_relaxed: $(1 - \alpha_k)\mR_{\ell}^{-1}R_{\ell}^{-1}\vz_{\ell}^{(k)} - \alpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)}$
    '''
    def __init__(self,alpha,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.alpha = alpha #tf.Variable(alpha_init,trainable=True)
    def call(self,inputs):
        x_over_R,z_over_R = inputs
        return  -(1 - self.alpha)*z_over_R - self.alpha*x_over_R



class GetNextIterZ(tf.keras.layers.Layer):
    '''
     inputs: All must be in spatial domain.

       Dx_nextlayer: \mD_{\ell + 1}\vx_{\ell + 1}^{(k + 1)}
       Ax_relaxed: -\alpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} - (1 - \alpha_k)\mR_{\ell}^{-1}\vz_{\ell}^{(k)}
       gamma_scaled: \frac{\vgamma_{\ell}^{(k)}}{\rho\sqrt{\mu_{\ell}}

     outputs: Also in spatial domain
       z: \vz_{\ell}^{(k + 1)}
    '''
    def __init__(self,rho,mu_init,mu_nextlayer,dictObj,dictObj_nextlayer,b_init,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rho = rho
        self.mu = tf.Variable(mu_init,trainable=True,dtype=self.dtype)
        self.mu_nextlayer = mu_nextlayer
        self.dictObj = dictObj
        self.dictObj_nextlayer = dictObj_nextlayer
        self.b = tf.Variable(b_init,trainable=True,dtype=self.dtype)
    def call(self,inputs):
        # inputs are in spatial domain
        Dx_nextlayer,Ax_relaxed,gamma_scaled = inputs
        currR = util.rotate_dims_left(self.dictObj.R,5)
        leadingFactor = 1/(self.mu_nextlayer + self.rho*self.mu/currR**2)
        return leadingFactor*tf.keras.layers.ReLU(dtype=self.dtype)(self.mu_nextlayer*Dx_nextlayer - (self.rho*self.mu/currR)*(Ax_relaxed + gamma_scaled) - self.b)



class GetNextIterZ_lastlayer(tf.keras.layers.Layer):
    '''
      inputs: All must be in spatial domain.

        Ax_relaxed: = -\alpha_k\mR_L^{-1}\vx_L^{(k + 1)} - (1 - \alpha_k)\mR_L^{-1}\vz_L^{(k)}
        gamma_scaled: \frac{\gamma_L^{(k)}}{\rho\sqrt{\mu_L}}

      outputs: Also in spatial domain
        z_over_R: \mR_L^{-1}\vz_L^{(k + 1)}
    '''
    def __init__(self,rho,mu_init,dictObj,b_init,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mu = tf.Variable(mu_init,trainable=True,dtype=self.dtype)
        self.dictObj = dictObj
        self.b = tf.Variable(b_init/(rho*mu_init),trainable=True,dtype=self.dtype) # Is this an active design decision to avoid dependence on mu?
    def call(self,inputs):
        Ax_relaxed,gamma_scaled = inputs
        R = util.rotate_dims_left(self.dictObj.R)
        return tf.keras.layers.ReLU(dtype=self.dtype)(-Ax_relaxed - gamma_scaled - R*self.b)



class GetNextIterGamma(tf.keras.layers.Layer):
    '''
      inputs: All must be in spatial domain.

        gamma_scaled: \frac{\gamma_{\ell}^{(k)}}{\rho\sqrt{\mu_{\ell}}
        z_over_R: \mR_{\ell}^{-1}\vz_{\ell}^{(k + 1)}
        Ax_relaxed: -\valpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} - (1 - \valpha_k)\mR_{\ell}^{-1}\vz_{\ell}^{(k)}

      outputs: Also in spatial domain
        gamma_scaled: \frac{\vgamma_{\ell}^{(k + 1)}}{\rho\sqrt{\mu_{\ell}}
    '''
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def call(self,inputs):
        gamma_scaled,z_over_R,Ax_relaxed = inputs
        return gamma_scaled + Ax_relaxed + z_over_R

class CropPadObject:
    def __init__(self,signalSz,strides,kernelSz,dtype):
        self.dtype = dtype
        padding = 0
        for ii in range(len(kernelSz)):
            stride_factor = 1
            for jj in range(0,ii):
                stride_factor = stride_factor*strides[jj]
            padding = padding + stride_factor*kernelSz[ii]
        stride_factor = stride_factor*strides[len(kernelSz) - 1]
        extra_padding = stride_factor - ((signalSz + padding) % stride_factor)
        self.build_crop_and_merge(signalSz,padding + extra_padding)

    def build_crop_and_merge(self,signal_sz,padding):
        padding_top = (padding/2).astype('int') # don't worry: i'll fix it if it doesn't divide evenly
        padding_bottom = (padding/2).astype('int')
        for ii in range(padding.shape[0]):
            if padding[ii] % 2 == 1:
                padding_top[ii] = ((padding[ii] - 1)/2).astype('int')
                padding_bottom[ii] = ((padding[ii] + 1)/2).astype('int')
        self.paddingTuple = ((padding_top[0],padding_bottom[0]),(padding_top[1],padding_bottom[1]))
        self.crop = tf.keras.layers.Cropping2D(self.paddingTuple,dtype=self.dtype)
        pad = tf.keras.layers.ZeroPadding2D(padding = self.paddingTuple)
        trues = tf.fill((1,signal_sz[0],signal_sz[1],1),1)
        mask = tf.cast(pad(trues),'bool')
        self.merge = Merge(pad,mask,dtype=self.dtype)
    def get_fft_size(self,signalSz,strides):
        fftSz = []
        fftSz.append((signalSz[0] + self.paddingTuple[0][0] + self.paddingTuple[0][1],signalSz[1] + self.paddingTuple[1][0] + self.paddingTuple[1][1]))
        ii = 0
        for stride in strides:
            fftSz.append((fftSz[ii][0]/stride,fftSz[ii][1]/stride))
            ii += 1
        return fftSz

class Merge(tf.keras.layers.Layer):
    def __init__(self,pad,mask,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pad = pad
        self.mask = mask
    def call(self,inputs):
        x,y = inputs
        return tf.where(self.pad(x),y,self.mask)

