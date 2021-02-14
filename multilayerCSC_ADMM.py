import tensorflow as tf
import matrix_decompositions_tf as fctr
import optmz
import jpeg_related_functions as jrf

class MultiLayerCSC(optmz.ADMM):
    def __init__(self,rho,alpha_init,mu_init,q,fltrSz,fftSz,noc,nof,rhaParam,dtype,noi,noL,dtype,*args,**kwargs):
        self.noL = noL
        super().__init__(rho = rho,alpha=alpha_init,noi = noi,dtype=dtype,*args,**kwargs)
        self.alpha = tf.Variable(initial_value=alpha_init,trainable=True)
        self.initializeLayers(rho,mu_init,q,noL,fltrSz,fftSz,noc,nof,rhaParam)
    def initializeLayers(self,rho,mu_init,q,noL,fltrSz,fftSz,noc,nof,rhaParam):
        self.dictObj = []
        self.updateX_layer = []
        for ii in range(noL):
            self.dictObj.append(fctr.dictionary_object2D(fltrSz= fltrSz[ii],fftSz=fftSz[ii],noc=noc[ii],nof=nof[ii],rho=rho,name='dict_layer' + str(ii),lraParam = lraParam,*args,dtype=self.dtype,**kwargs))
            self.updateX_layer.append(GetNextIterX(rho,self.dictObj[ii],dtype=self.dtype))
        self.relax_layer = GetRelaxedAx(rho=rho,dtype=self.dtype)
        self.updateZ_layer = []
        self.updateGamma_layer = GetNextIterGamma(dtype=self.dtype)
        self.updateZ_lastlayer = GetNextIterZ_lastlayer(rho,mu_init,self.dictObj[noL - 1],b_init,dtype=self.dtype.real_dtype)
        mu = self.updateZ_lastlayer.mu
        for ii in range(noL - 2,-1,-1):
            self.updateZ_layer[ii] = GetNextIterZ(rho,mu_init,mu,dictObj,b_init,dtype=self.dtype.real_dtype)
            mu = self.updateZ_layer[ii].mu
        W = jrf.RGB2JPEG_Coef(dtype=self.dtype)
        Wt = jrf.JPEG_Coef2RGB(dtype=self.dtype)
        self.updatev = jrf.Z_Update_JPEG(mu,rho,q,W,Wt,dtype = self.dtype.real_dtype)
        self.relax0 = jrf.Relax_SmoothJPEG(dtype=self.dtype.real_dtype)
        self.updateeta = jrf.GammaUpdate_JPEG(dtype=self.dtype.real_dtype)



    # High-level Initialization Functions (x,y,u,Ax,By,C)
    def get_negative_C(self,s):
        return s
    def init_x(self,s,negC):
        s_LF,QWs = s
        x = []
        x.append(self.xinit(self.FFT(QWs - s_LF),layer = 0))
        for ii in range(1,self.noL):
            x.append(self.xinit(x[ii - 1],layer = ii))
        return (x,(None,x))
    def init_y(self,s,x,Ax,negC):
        Azero,Ax_layers = Ax
        s_LF,QWs = s
        v,Bv = self.vinit(x[0],Azero,s_LF)
        z = []
        for ii in range(self.noL - 1):
            z.append(self.zinit(x_nextlayer=x[ii + 1],Ax=Ax_layers[ii],layer=ii))
        z.append(self.zinit_last(Ax=Ax_layers[self.noL - 1]))
        y = (v,z)
        By = (Bv,z)
        return (y,By)
    def init_u(self,s,Ax,By,negC):
        eta = (0.,0.,0.)
        Azero,Ax_layers = Ax
        Bv,Bz = By
        gamma = []
        for ii in range(self.noL):
            gamma.append(self.gammainit(Ax_layers,Bz,ii))
        return (eta,gamma)


    # High-level Update Functions for each iteration (x,y,u,Ax,By)
    def xstep(self,y,u):
        v,z = y
        eta,gamma = u
        x = []
        x.append(self.updateX(z_prevlayer=v,z=z[0],gamma_scaled = gamma[0],layer=0)
        for ii in range(1,self.noL):
            x.append(self.updateX(z_prevlayer=z[ii - 1],z = z[ii],gamma_scaled = gamma[ii],layer=ii)
        return x,(0.,x)
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
        z.append(self.updateZ_last(Ax_relaxed=Ax_layers[self.noL - 1],gamma_scaled=gamma[self.nol - 1]))
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

    # Low-level Initialization Functions (x[layer],v,z[layer],eta,gamma[layer],Azero,Ax[layer],Bv,Bz[layer])
    def xinit(self,xprev,layer):
        if layer == 0:
            return self.dictObj[layer].dhmul(xprev)
        else:
            return self.dictObj[layer].dhmul(self.dictObj[layer - 1].R*xprev)
    def vinit(self,x_0,Azero,s_LF):
        Dx = self.IFFT(self.dictObj[0].dmul(x_0))
        vpluss_LF,Bv = self.updatev(Dx + s_LF,Azero,0.)
        return (vpluss_LF - s_LF,Bv) 
    def zinit(self,xnext,Ax,layer):
        return self.updateZ(xnext,Ax,0.,layer)
    def zinit_last(self,Ax):
        return updateZ_last(Ax,0.)
    def gammainit(self,Ax,Bz,layer):
        return updateGamma(0.,Bz,Ax,layer)


    # Low-level Update Functions (x[layer],v,z[layer],eta,gamma[layer],Azero,Ax[layer],Bv,Bz[layer])
    def updateX(self,z_prevlayer,z,gamma_scaled,layer):
        if layer == self.noL - 1:
            return self.updateX_layer[layer]((z_prevlayer,z,gamma_scaled))
        else:
            return self.updateX_layer[layer]((z_prevlayer,z/self.dictObj[layer].R,gamma_scaled))
    def relax_layers(self,x,z,layer):
        if layer < self.noL - 1:
            z_over_R = z/self.dictObj[layer].R
        else
            z_over_R = z
        return self.relax_layer((x,z_over_R))
    def relax_zero(self,Bv,negC):
        return self.relax0((negC,Bv))
    def updateV(self,x_0,Azero,eta_over_rho,s_LF):
        Dx = self.IFFT(self.dictObj[0].dmul(x_0))
        vpluss_LF,Bv = self.updatev(Dx + s_LF,Azero,eta_over_rho)
        return (self.FFT(vpluss_LF - s_LF),Bv)
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
    def updateEta(self,eta_over_rho,AzeroplusC,Bv):
        return self.updateeta((eta_over_rho,AzeroplusC,Bv))

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
        return dictObj.qinv(dictObj.dhmul(zprevlayer) - self.rho*(z_over_R - gamma_scaled))

class GetRelaxedAx(tf.keras.layers.Layer):
    '''
      inputs: All must be in frequency domain

        x_over_R: $\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)}$
        z_over_R: $\mR_{\ell}^{-1}\vz_{\ell}^{(k)}$

      outputs: Also in frequency domain
        Ax_relaxed: $(1 - \alpha_k)\mR_{\ell}^{-1}R_{\ell}^{-1}\vz_{\ell}^{(k)} - \alpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)}$
    '''
    def __init__(self,alpha_init,*args,**kwargs):
        super().__init__(*args,**kwargs)
        alpha = tf.Variable(alpha_init,trainable=True)
    def call(self,inputs):
        x_over_R,z_over_R = inputs
        return  (1 - alpha)*z_over_R - alpha*x_over_R



class GetNextIterZ(tf.keras.layers.Layer):
    '''
     inputs: All must be in spatial domain.

       Dx_nextlayer: \mD_{\ell + 1}\vx_{\ell + 1}^{(k + 1)}
       Ax_relaxed: \alpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} - (1 - \alpha_k)\mR_{\ell}^{-1}\vz_{\ell}^{(k)}
       gamma_scaled: \frac{\vgamma_{\ell}^{(k)}}{\rho\sqrt{\mu_{\ell}}

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
        gamma_scaled: \frac{\gamma_L^{(k)}}{\rho\sqrt{\mu_L}}

      outputs: Also in spatial domain
        z_over_R: \mR_L^{-1}\vz_L^{(k + 1)}
    '''
    def __init__(self,rho,mu_init,dictObj,b_init,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mu = tf.Variable(mu_init,trainable=True)
        self.dictObj = dictObj
        self.b = tf.Variable(b_init/(rho*mu_init),trainable=True) # Is this an active design decision to avoid dependence on mu?
    def call(self,inputs):
        Ax_relaxed,gamma_scaled = inputs
        return tf.keras.layers.activations.RELU(-Ax_relaxed - gamma_scaled - self.dictObj.R*self.b)



class GetNextIterGamma(tf.keras.layers.Layer):
    '''
      inputs: All must be in spatial domain.

        gamma_scaled: \frac{\gamma_{\ell}^{(k)}}{\rho\sqrt{\mu_{\ell}}
        z_over_R: \mR_{\ell}^{-1}\vz_{\ell}^{(k + 1)}
        Ax_relaxed: \valpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} - (1 - \valpha_k)\mR_{\ell}^{-1}\vz_{\ell}^{(k)}

      outputs: Also in spatial domain
        gamma_scaled: \frac{\vgamma_{\ell}^{(k + 1)}}{\rho\sqrt{\mu_{\ell}}
    '''
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def call(self,inputs):
        gamma_scaled,z_over_R,Ax_relaxed = inputs
        return gamma_scaled + Ax_relaxed + z_over_R
