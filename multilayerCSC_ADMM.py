import tensorflow as tf
import matrix_decompositions_tf as fctr
class ADMM:
    def __init__(self):
        raise NotImplementedError
    def xstep(self):
        raise NotImplementedError
    def relax(self):
        raise NotImplementedError
    def ystep(self):
        raise NotImplementedError
    def ustep(self):
        raise NotImplementedError
    def solvestep(self):
        self.xstep()
        self.relax()
        self.ystep()
        self.ustep()
    class ReconstructSignal(tf.keras.layers.Layer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
        def call(self,s):
            raise NotImplementedError

class MultiLayerCSC(ADMM):
    def __init__(self,rho,alpha_init,noL,dtype,*args,**kwargs):
        self.noL = noL

    def xstep(self):
        self.x[0] = self.updateX(z_prevlayer=self.y,z=self.z[0],gamma_scaled=gamma[0],layer=0)
        for ii in range(2,self.noL):
            self.x[ii] = self.updateX(z_prevlayer=self.z[ii - 1],gamma_scaled=gamma[0],layer=ii)

    def relax(self):
        for ii in range(self.noL):
            self.Ax[ii] = self.relaxation(self.x[ii],self.z[ii],ii)

    def ystep(self):
        # need to add code for self.y
        for ii in range(self.noL - 1):
            self.z[ii] = self.updateZ(x_nextlayer=self.x[ii + 1],Ax_relaxed=self.Ax[ii],gamma_scaled=self.gamma[ii],layer=ii)
        self.z[self.noL - 1] = self.updateZ_last(Ax_relaxed=self.Ax[self.noL - 1],gamma_scaled=self.gamma[self.nol - 1])

    def ustep(self):
        for ii in range(self.noL):
            self.gamma[ii] = self.updateGamma(self.gamma[ii],self.z[ii],self.Ax[ii],ii)

    def initializeLayers(self,rho,alpha_init,noL,fltrSz,fftSz,noc,nof,rho,rhaParam,dtype):
        self.dictObj = []
        self.updateX_layer = []
        for ii in range(noL):
            self.dictObj.append(fctr.dictionary_object2D(fltrSz= fltrSz[ii],fftSz=fftSz[ii],noc=noc[ii],nof=nof[ii],rho=rho,name='dict_layer' + str(ii),lraParam = lraParam,*args,dtype=dtype,**kwargs))
            self.updateX_layer.append(GetNextIterX(rho,self.dictObj[ii],dtype=dtype))
        self.relax_layer = GetRelaxedAx(rho=rho,dtype=dtype)
        self.updateZ_layer = []
        self.updateGamma_layer = GetNextIterGamma(dtype=dtype)
        self.updateZ_lastlayer = GetNextIterZ_lastlayer(rho,mu_init,self.dictObj[noL - 1],b_init,dtype=dtype.real_dtype)
        mu = self.updateZ_lastlayer.mu
        for ii in range(noL - 2,-1,-1):
            self.updateZ_layer[ii] = GetNextIterZ(rho,mu_init,mu,dictObj,b_init,dtype=dtype.real_dtype)
            mu = self.updateZ_layer[ii].mu

    def updateX(self,z_prevlayer,z,gamma_scaled,layer):
        if layer == self.noL - 1:
            return self.updateX_layer[layer]((z_prevlayer,z,gamma_scaled))
        else:
            return self.updateX_layer[layer]((z_prevlayer,z/self.dictObj[layer].R,gamma_scaled))

    def relaxation(self,x,z,layer):
        if layer < self.noL - 1:
            z_over_R = z/self.dictObj[layer].R
        else
            z_over_R = z
        return self.relax_layer((x,z_over_R))

    def updateZ(self,x_nextlayer,Ax_relaxed,gamma_scaled,layer):
        assert(layer < self.noL - 1)
        Dx = self.dictObj[layer + 1].dmul(x_nextlayer)
        z = self.updateZ_layer[layer]((self.IFFT(Dx),self.IFFT(Ax_relaxed),self.IFFT(gamma_scaled))
        return self.FFT(z)

    def updateZ_last(self,Ax_relaxed,gamma_scaled):
        z = self.updateZ_lastlayer(self.IFFT(Ax_relaxed),self.IFFT(gamma_scaled))
        return self.IFFT(z)

    def updateGamma(self,gamma_scaled,z,Ax_relaxed,layer):
        if layer < self.noL - 1:
            z_over_R = z/self.dictObj[layer]
        else:
            z_over_R = z
        return self.gammaUpdate_layer(gamma_scaled,z_over_R,Ax_relaxed)

class GetNextIterX(tf.keras.layers.Layer):
    '''
      inputs: All must be in frequency domain.
    
        z_previouslayer: $\vz_{\ell - 1}^{(k)}$
        z_over_R: $\mR^{-1}\vz_{\ell}^{(k)}$
        gamma_scaled: $\frac{\vgamma_{\ell}^{(k)}}{\rho\sqrt{\alpha_{\ell}}$

      outputs: Also in frequency domain
        x_scaled: \mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)}
    '''
    def __init__(self,rho,dictObj,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dictObj = dictObj
        self.rho = rho
    def call(self,inputs):
        z_prevlayer,z_over_R,gamma_scaled = inputs
        return dictObj.qinv(dictObj.dhmul(zprevlayer) - self.rho*(z_over_R - gamma_scaled))

class GetRelaxedAx(tf.keras.layers.Layer):
    def __init__(self,alpha_init,*args,**kwargs):
        super().__init__(*args,**kwargs)
        alpha = tf.Variable(alpha_init,trainable=True)
    def call(self,inputs):
        x_over_R,z_over_R = inputs
        return  (1 - alpha)*z_over_R - alpha*x_over_R



class GetNextIterZ(tf.keras.layers.Layer):
    ''' inputs: All must be in spatial domain.

       Dx_nextlayer: \mD_{\ell + 1}\vx_{\ell + 1}^{(k + 1)}
       Ax_relaxed: \alpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} - (1 - \alpha_k)\mR_{\ell}^{-1}\vz_{\ell}^{(k)}
       gamma_scaled: \frac{\vgamma_{\ell}^{(k)}}{\rho\sqrt{\alpha_{\ell}}

     outputs: Also in spatial domain
       z: \vz_{\ell}^{(k + 1)}
    '''
    def __init__(self,rho,mu_init,mu_nextlayer,dictObj,b_init,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rho = rho
        self.mu = tf.Variable(mu_init,trainable=True)
        self.mu_nextlayer = mu_nextlayer
        self.dictObj = dictObj
        self.b = tf.Variable(b_init,trainable=True)
    def call(self,inputs):
        # inputs are in spatial domain
        Dx_nextlayer,Ax_relaxed,gamma_scaled = inputs
        leadingFactor = 1/self.mu_nextlayer + rho*mu/dictObj_nextlayer.R**2
        return tf.keras.layers.activations.RELU(mu_nextlayer*Dx_nextlayer - (self.rho*mu/dictObj.R)*(Ax_relaxed + gamma_scaled) - self.b)



class GetNextIterZ_lastlayer(tf.keras.layers.Layer):
    '''
    inputs: All must be in spatial domain.

       Ax_relaxed: \alpha_k\mR_L^{-1}\vx_L^{(k + 1)} - (1 - \alpha_k)\mR_L^{-1}\vz_L^{(k)}
       gamma_scaled: \frac{\gamma_L^{(k)}}{\rho\sqrt{\alpha_L}}

    outputs: Also in spatial domain
       z_over_R: \mR_L^{-1}\vz_L^{(k + 1)}
     '''
    def __init__(self,rho,mu_init,dictObj,b_init,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mu = tf.Variable(mu_init,trainable=True)
        self.dictObj = dictObj
        self.b = tf.Variable(b_init/(rho*mu_init),trainable=True)
    def call(self,inputs):
        Ax_relaxed,gamma_scaled = inputs
        return tf.keras.layers.activations.RELU(-Ax_relaxed - gamma_scaled - self.dictObj.R*self.b)



class GetNextIterGamma(tf.keras.layers.Layer):
    '''
      inputs: All must be in spatial domain.

        gamma_scaled: \frac{\gamma_{\ell}^{(k)}}{\rho\sqrt{\valpha_{\ell}}
        z_over_R: \mR_{\ell}^{-1}\vz_{\ell}^{(k + 1)}
        Ax_relaxed: \valpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} - (1 - \valpha_k)\mR_{\ell}^{-1}\vz_{\ell}^{(k)}

      outputs: Also in spatial domain
        gamma_scaled: \frac{\vgamma_{\ell}^{(k + 1)}}{\rho\sqrt{\alpha_{\ell}}
    '''
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def call(self,inputs):
        gamma_scaled,z_over_R,Ax_relaxed = inputs
        return gamma_scaled + Ax_relaxed + z_over_R
