import tensorflow as tf



def GetNextIterX(tf.keras.layers.Layer):
    '''
      inputs: All must be in frequency domain.
    
        z_previouslayer: $\vz_{\ell - 1}^{(k)}$
        z_over_R: $\mR^{-1}\vz_{\ell}^{(k)}$
        gamma_scaled: $\frac{\gamma_{\ell}^{(k)}}{\rho\sqrt{\alpha_{\ell}}$
    
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

def GetRelaxedAx(tf.keras.layers.Layer):
    def __init__(self,alpha_init,*args,**kwargs):
        super().__init__(*args,**kwargs)
        alpha = tf.Variable(alpha_init,trainable=True)
    def call(self,inputs):
        z_over_R,x_over_R = inputs
        return  (1 - alpha)*z_over_R - alpha*x_over_R

def GetNextIterZ(tf.keras.layers.Layer):
    ''' inputs: All must be in spatial domain.
    
       Dx_nextlayer: \mD_{\ell + 1}\vx_{\ell + 1}^{(k + 1)}
       Ax_relaxed: \alpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} - (1 - \alpha_k)\mR_{\ell}^{-1}\vz_{\ell}^{(k)}
       gamma_scaled: \frac{\gamma_{\ell}^{(k)}}{\rho\sqrt{\alpha_{\ell}}
    
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

def GetNextIterZ_lastlayer(tf.keras.layers.Layer):
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

def GetNextIterGamma(tf.keras.layers.Layer):
    '''
      inputs: All must be in spatial domain.
    
        gamma_scaled: \frac{\gamma_{\ell}^{(k)}}{\rho\sqrt{\alpha_{\ell}}
        z_over_R: \mR_{\ell}^{-1}\vz_{\ell}^{(k + 1)}
        Ax_relaxed: \alpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} - (1 - \alpha_k)\mR_{\ell}^{-1}\vz_{\ell}^{(k)}
    
      outputs: Also in spatial domain
        gamma_scaled: \frac{\gamma_{\ell}^{(k + 1)}}{\rho\sqrt{\alpha_{\ell}}
    '''
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def call(self,inputs):
        gamma_scaled,z_over_R,Ax_relaxed = inputs
        return gamma_scaled + Ax_relaxed + z_over_R
