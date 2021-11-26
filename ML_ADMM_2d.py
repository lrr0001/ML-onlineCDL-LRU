import tensorflow as tf
import matrix_decompositions_tf as fctr
import optmz
import jpeg_related_functions as jrf
import transforms as transf
import util
import numpy as np
import post_process_grad as ppg
class MultiLayerCSC(optmz.ADMM_Relaxed):
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
        Bzero
        Bz: list
            Bz[\ell] 
    '''
    def __init__(self,rho,alpha_init,mu_init,b_init,cropAndMerge,fftSz,strides,D,n_components,noi,noL,cmplxdtype,longitstat=False,*args,**kwargs):
        rho,mu_init = self.init_param(rho,alpha_init,mu_init,cropAndMerge,noi,noL,cmplxdtype,longitstat,*args,**kwargs)
        self.initializeLayers(rho,mu_init,alpha_init,util.makelist(b_init,noL),noL,fftSz,strides,D,n_components,cmplxdtype)
        self.initializeInputHandlingLayers()
    def init_param(self,rho,alpha_init,mu_init,cropAndMerge,noi,noL,cmplxdtype,longitstat,*args,**kwargs):
        self.longitstat = longitstat
        self.cmplxdtype = cmplxdtype
        dtype = cmplxdtype.real_dtype
        rho = tf.cast(rho,dtype = cmplxdtype.real_dtype)
        mu_init = util.makelist(mu_init,noL)
        self.noL = noL
        optmz.ADMM_Relaxed.__init__(self=self,rho = rho,alpha=alpha_init,noi = noi,dtype=dtype,*args,**kwargs)
        self.cropAndMerge = cropAndMerge
        self.initialize_alpha(alpha_init)
        return rho,mu_init
    def initialize_alpha(self,alpha):
        '''This overwrites the self.alpha from the initialization of ADMM_Relaxed.'''
        self.alpha = tf.Variable(initial_value=alpha,trainable=False,name = 'alpha',dtype=tf.as_dtype(self.cmplxdtype).real_dtype)
    def get_config(self):
        config_dict = {'complex_dtype': self.cmplxdtype,
                       'num_of_Layers': self.noL,
                       'strides': self.strides,
                       'rho': self.rho,
                       'noi': self.noi,
                       'record_iteration_stats': self.longitstat}
        return config_dict

    def initializeLayers(self,rho,mu_init,alpha_init,b_init,noL,fftSz,strides,D,n_components,cmplxdtype):
        self.strides = strides
        self.dictObj = []
        self.updateX_layer = []
        self.FFT,self.IFFT,self.FFT_factor = self.build_fft_layers(fftSz,noL)

        for ii in range(noL):
            self.dictObj.append(self.build_dict_obj(fftSz[ii],D[ii],rho,n_components,cmplxdtype,ii))
            # Changed for frequency domain
            self.updateX_layer.append(GetNextIterX(rho,self.dictObj[ii],self.FFT[ii],self.IFFT[ii],dtype=cmplxdtype.real_dtype))

        
        reversed_updateZ_layer = []
        #self.updateZ_layer = [[],]*(noL - 1)
        self.updateGamma_layer = GetNextIterGamma(alpha=self.alpha,dtype=cmplxdtype.real_dtype)

        self.updateZ_lastlayer,mu = self.build_updateZ_lastlayer(fftSz[noL - 1],D[noL - 1].shape[-1],rho,mu_init[noL - 1],self.dictObj[noL - 1],b_init[noL - 1],cmplxdtype)
        if noL == 1:
            pass
        for ii in range(noL - 2,-1,-1): # appending backwards because need the mu value, and tensorflow doesn't like when I initialize the entire list.  I'll flip it later so the indices make sense.

            zupdate,mu = self.build_updateZ_layer(fftSz[ii],D[ii].shape[-1],rho,mu_init[ii],mu,self.dictObj[ii],self.dictObj[ii + 1],b_init[ii],cmplxdtype,strides[ii],ii)
            #self.updateZ_layer[ii] = zupdate
            reversed_updateZ_layer.append(zupdate)
        
        self.updateZ_layer = []
        for ii in range(noL - 1): # Last shall be first and first shall be last.
            self.updateZ_layer.append(reversed_updateZ_layer[noL - 2 - ii])

    # *** CHANGED FOR RAW TRAINING ***
    def initializeInputHandlingLayers(self):
        if self.noL > 1:
            mu = self.updateZ_layer[0].mu
        else:
            mu = self.updateZ_lastlayer.mu
        self.updateBzero = GetNextIterBZero(cropAndMerge = self.cropAndMerge,mu = mu,rho = self.rho,dtype = self.cmplxdtype.real_dtype)
        self.updateeta = UpdateEta(alpha = self.alpha,dtype = self.cmplxdtype.real_dtype)


    def build_fft_layers(self,fftSz,noL):
        FFT = []
        IFFT = []
        FFT_factor = []
        for ii in range(noL):
            FFT.append(transf.fft2d_inner(fftSz[ii]))
            IFFT.append(transf.ifft2d_inner(fftSz[ii]))
            FFT_factor.append(np.prod(fftSz[ii]))
        return FFT,IFFT,FFT_factor

    def build_dict_obj(self,fftSz,D,rho,n_components,cmplxdtype,layer):
        if layer == 0:
            return fctr.dictionary_object_init_full_sp(fftSz=fftSz,D = tf.convert_to_tensor(D),rho=tf.cast(rho,dtype=cmplxdtype),objname='dict_layer' + str(layer),n_components = n_components)
        else:
            return fctr.dictionary_object_init_sp(fftSz=fftSz,D = tf.convert_to_tensor(D),rho=tf.cast(rho,dtype=cmplxdtype),objname='dict_layer' + str(layer),n_components=n_components)




    def build_updateZ_layer(self,fftSz,nof,rho,mu_init,munext,dictObj,nextdictObj,b_init,cmplxdtype,strides,layer):
        if strides == 2:
            raise NotImplementedError
            zshapes = self.get_b_shape(fftSz,nof)
            #Zupdate = GetNextIterZ(rho,mu_init,munext,dictObj,nextdictObj,tf.fill(zshapes[0],value = tf.cast(b_init,dtype=cmplxdtype.real_dtype)),dtype=cmplxdtype.real_dtype)
            Zupdate = GetNextIterZFreq(rho,self.IFFT[layer + 1],tf.cast(mu_init,self.cmplxdtype.real_dtype),munext,dictObj,nextdictObj,tf.fill(zshapes,value = tf.cast(b_init,dtype=cmplxdtype.real_dtype)),dtype=cmplxdtype)
            mu = Zupdate.mu
            #Z_update_shift = GetNextIterZ_downsampleTossed(rho,mu,dictObj,tf.fill(zshapes[0],value = tf.cast(b_init,dtype=cmplxdtype.real_dtype)),dtype=cmplxdtype.real_dtype)
            Z_update_shift = GetNextIterZFreq_downsampleTossed(rho,self.IFFT[layer + 1],mu,dictObj,Zupdate.b,dtype=cmplxdtype)
            ifft = transf.ifft2d_inner(zshapes[1][1:3])
            Z_update_missed_cols = GetNextIterZFreq_downsampleTossed(rho,ifft,mu,dictObj,Zupdate.b,dtype=cmplxdtype)
            shift_concat = util.AlternateConcat(shape=zshapes[0][1:],num_of_items=2,axis=1,dtype = cmplxdtype.real_dtype)
            cols_concat = util.AlternateConcat(shape=zshapes[1][1:],num_of_items=2,axis=2,dtype = cmplxdtype.real_dtype)
            #return {'downsampled': Zupdate, 'shifted': Z_update_shift, 'missed_cols': Z_update_missed_cols,'shift_concat': shift_concat,'cols_concat': cols_concat},mu 
            return (Zupdate,Z_update_shift,Z_update_missed_cols,shift_concat,cols_concat),mu
        else:
            b = tf.zeros(self.get_b_shape(fftSz,nof),dtype=cmplxdtype.real_dtype) + b_init


            # **** Frequency domain or Spatial Domain???? Changed to Frequency domain****
            #zUpdate = GetNextIterZFreq(rho,self.IFFT[layer],tf.cast(mu_init,cmplxdtype.real_dtype),munext,dictObj,nextdictObj,b,dtype=cmplxdtype,name='Z_layer' + str(layer))
            zUpdate = GetNextIterZ(rho,tf.cast(mu_init,cmplxdtype.realdtype),munext,dictObj,nextdictObj,b,dtype=cmplxdtype.real_dtype,name='Z_layer' + str(layer))


            return zUpdate,zUpdate.mu




    def build_updateZ_lastlayer(self,fftSz,nof,rho,mu_init,dictObj,b_init, cmplxdtype):
        b = tf.zeros(self.get_b_shape(fftSz,nof),dtype=cmplxdtype.real_dtype) + b_init

        # Changed from Frequency to Spatial Domain
        lastlayer = GetNextIterZ_lastlayer(rho,tf.cast(mu_init,self.cmplxdtype.real_dtype),dictObj,b,dtype=cmplxdtype,name='Z_layer' + str(self.noL - 1))

        return lastlayer,lastlayer.mu

    def init_itstats(self,s):
        if not self.longitstat:
            return []
        else:
            raise NotImplementedError

    def itstats_record(self,x,y,u,uhalf,Ax,By,negC,itstats):
        if not self.longitstat:
            return []
        else:
            raise NotImplementedError

    def get_output(self,s,y,u,By,negC,itstats):
        for layer in range(self.noL - 1,0,-1):
            x,Ax = self.xstep_trunc(y,u,By,negC,layer)
            u = self.relax_trunc(u,Ax,By,negC,layer)
            y,By = self.ystep_trunc(x,u,Ax,negC,layer,frozen=False)
            u = self.ustep_trunc(u,Ax,By,negC,layer)

        x,Ax = self.xstep_trunc(y,u,By,negC,layer=0)
        Dx = self.dictObj[0].dmul_sp(x[0])
        return (self.cropAndMerge.crop(tf.squeeze(Dx,axis=-1)),itstats)

    def get_b_shape(self,fftSz,M):
        #return [1,fftSz[0],fftSz[1],M,1,]
        #return [1,1,1,M,1]
        return (1,)



    # High-level Initialization Functions (x,y,u,Ax,By,C)

    # CHANGED FOR RAW TRAINING
    def get_negative_C(self,s):
        s_HF,s_LF,compressed = s
        return (None,self.cropAndMerge.crop(s_HF))

    def init_x(self,s,negC):
        '''Initializes x for all layers. Ax includes an empty slot up front, anticipating its use later. While it would make more sense to add the variable to x as well, naming gets messy, so I'll leave that be. My plan for the added variable has Axzero = -xzero, so storing it with the rest of x is unnecessary, since its value can be accessed from By in the x and u updates. The second element of tuple Ax would more accurately be the negative of what it is in this code, but there's no reason to store the negative version of the same variable. Instead, the equations flip the sign whenever Ax[1] is accessed.'''
        s_HF,temp1,temp2 = s
        temp,s_crop = negC
        x = []
        x.append(self.xinit(tf.expand_dims(s_HF,axis = -1),layer = 0))
        for ii in range(1,self.noL):
            if self.strides[ii - 1] == 2:
                x.append(self.xinit(util.freq_downsample(x[ii - 1]),layer=ii))
            else:
                x.append(self.xinit(x[ii - 1],layer = ii))
        Ax = (None,x)
        return x,Ax


    def init_y(self,s,x,Ax,negC):
        Azero,Ax_layers = Ax
        temp,s_crop = negC
        v,Bv = self.vinit(s,x[0],s_crop)
        z = []
        for ii in range(self.noL - 1):
            z.append(util.rotate_dims_left(self.dictObj[ii].divide_by_R.R,5)*x[ii])
            #z.append(self.zinit(xnext=x[ii + 1],Ax=Ax_layers[ii],layer=ii))
        #z.append(self.zinit_last(Ax=Ax_layers[self.noL - 1]))
        z.append(x[self.noL - 1])
        y = (v,z)
        By = (Bv,z)
        return y,By

    def init_u(self,s,Ax,By,negC):
        Azero,Ax_layers = Ax
        Bzero,Bz = By
        temp,s_crop = negC
        gamma = []
        for ii in range(self.noL):
            gamma.append(self.gammainit(Ax_layers[ii],Bz[ii],ii))
        eta = self.etainit(Azero,Bzero,s_crop)
        return (eta,gamma)


    # High-level Update Functions for each iteration (x,y,u,Ax,By)
    def xstep(self,y,u,By,negC):
        return self.xstep_trunc(y,u,By,negC,layer=self.noL - 1)


    def relax(self,u,Ax,By,negC):
        return self.relax_trunc(u,Ax,By,negC,layer=self.noL)

    def ystep(self,x,uhalf,Ax,negC):
        ''' This code computes the updates for z, v, and Bzero. The other step functions use a truncated step function. However, while it is good practice to avoid code redundancy, here the last layer is distinct from ystep_trunc. ystep_trunc also does not freeze the dictionary weights and does not use updateZlast. If-else logic could resolve these distinctions, but I find it clearer to leave these as two separate functions.'''
        eta,gamma = uhalf
        Azero,Ax_layers = Ax
        temp,s_crop = negC
        z = []
        for ii in range(self.noL - 1):
            z.append(self.updateZ(x_nextlayer=x[ii + 1],Ax=Ax_layers[ii],gamma_scaled=gamma[ii],layer=ii))
        z.append(self.updateZ_last(Ax=Ax_layers[self.noL - 1],gamma_scaled=gamma[self.noL - 1]))
        v,Bzero = self.updateV(x[0],eta,s_crop)
        y = (v,z)
        By = (Bzero,z)
        return y,By

    def ustep(self,uhalf,Ax,By,negC):
        return self.ustep_trunc(uhalf,Ax,By,negC,layer=self.noL)

    def xstep_trunc(self,y,u,By,negC,layer):
        v,z = y
        eta,gamma = u
        x = []
        x.append(self.updateX(z_prevlayer=v,z=z[0],gamma_scaled = gamma[0],layer=0))
        for ii in range(1,layer + 1):
            if self.strides[ii - 1] == 2:
                x.append(self.updateX(z_prevlayer=util.freq_downsample(z[ii - 1]),z = z[ii],gamma_scaled = gamma[ii],layer=ii))
            else:
                x.append(self.updateX(z_prevlayer=z[ii - 1],z = z[ii],gamma_scaled = gamma[ii],layer=ii))
        return x,(None,x)

    def relax_trunc(self,u,Ax,By,negC,layer):
        return self.ustep_trunc(u,Ax,By,negC,layer,relax_bool = True)

    def ystep_trunc(self,x,u,Ax,negC,layer,frozen=True):
        eta,gamma = u
        Azero,Ax_layers = Ax
        temp,s_crop = negC
        z = []
        for ii in range(layer - 1):
            z.append(self.updateZ(x_nextlayer=x[ii + 1],Ax=Ax_layers[ii],gamma_scaled=gamma[ii],layer=ii))
        z.append(self.updateZ(x_nextlayer=x[layer],Ax=Ax_layers[layer - 1],gamma_scaled=gamma[layer - 1],layer=layer - 1,frozen=frozen))
        v,Bzero = self.updateV(x[0],eta,s_crop)
        y = (v,z)
        By = (Bzero,z)
        return y,By

    def ustep_trunc(self,u,Ax,By,negC,layer,relax_bool=False):
        eta,gamma = u
        Azero,Ax_layers = Ax
        temp,s_crop = negC
        Bv,z = By
        for ii in range(layer):
            gamma[ii] = self.updateGamma(gamma[ii],z[ii],Ax_layers[ii],ii,relax_bool)
        eta = self.updateEta(eta,Azero,Bv,s_crop,relax_bool)
        return (eta,gamma) 

    def get_obj(self,y,negC):
        v,z = y
        temp,s_crop = negC
        return self.data_fid_z(y,s_crop) + self.coef_penalty(z)

    def evaluateLagrangian(self,s,x,y,u,Ax,By,negC):
        s_HP,s_LP,compressed = s
        v,z = y
        Azero,Ax_layer = Ax
        Bzero,Bz = By
        eta,gamma = u
        rep_err = self.representation_error(v,x,z)
        print('representation error: ',rep_err)
        coef_pen = self.coef_penalty(z)
        print('coefficient_penalty: ', coef_pen)
        cnstr_pen = self.cnstrPenalty(u,Ax,By,negC)
        print('constraint penalty: ',cnstr_pen)
        u_correction = self.ucorrection(u)
        print('u correction: ',u_correction)
        return rep_err + coef_pen + cnstr_pen - u_correction
        #return self.representation_error(v,x,z) + self.coef_penalty(z) + self.cnstrPenalty(Azero,Bzero,eta,Ax,By,gamma,negC) - self.ucorrection(u)

    def representation_error(self,v,x,z):
        if self.noL > 1:
            mu = self.updateZ_layer[0].mu
        else:
            mu = self.updateZ_lastlayer.mu
        representation_sum = mu/2*self.reconstructionTerm_sp(v,self.dictObj[0].dmul_sp(x[0]))
        for ll in range(1,self.noL - 1):
            mu = self.updateZ_layer[ll].mu
            representation_sum += mu/2*self.reconstructionTerm_sp(z[ll - 1],self.dictObj[ll].dmul_sp(x[ll]))
        if self.noL > 1:
            mu = self.updateZ_lastlayer.mu
            representation_sum += mu/2*self.reconstructionTerm_sp(z[self.noL - 2],self.dictObj[self.noL].dmul_sp(x[self.noL - 1]))
        return representation_sum

    def coef_penalty(self,z):
        if self.noL > 1:
            penalty_sum = self.penaltyTerm(z[0],0)
        else:
            penalty_sum = tf.cast(0.0,self.cmplxdtype.real_dtype)
        for ll in range(1,self.noL - 1):
            penalty_sum += self.penaltyTerm(z[ll],ll)
        R = tf.reshape(self.dictObj[-1].divide_by_R.R,shape=(1,1,1,self.dictObj[-1].divide_by_R.R.shape[4],1))
        penalty_sum += self.penaltyTerm(z[self.noL - 1]*R,self.noL - 1)
        return penalty_sum

    def cnstrPenalty(self,u,Ax,By,negC):
        temp,s_crop = negC
        eta,gamma = u
        temp,Ax_layers = Ax
        Bzero,Bz = By
        cnstr_penalty_sum = (self.rho/2)*self.reconstructionTerm_sp(Bzero + eta,s_crop)
        for ii in range(self.noL - 1):
            R = tf.reshape(self.dictObj[ll].divide_by_R.R,shape=(1,1,1,self.dictObj[ll].divide_by_R.R.shape[4],1))
            mu = self.updateZ_layer[ll].mu
            cnstr_penalty_sum += (mu*self.rho/2)*self.reconstructionTerm_sp(Bz[ll]/R + gamma[ll],Ax_layers[ll])
        mu = self.updateZ_lastlayer.mu
        cnstr_penalty_sum += (mu*self.rho/2)*self.reconstructionTerm_sp(Bz[self.noL - 1] + gamma[self.noL - 1],Ax_layers[self.noL - 1])
        return cnstr_penalty_sum

    def ucorrection(self,u):
        eta,gamma = u
        unorm = (self.rho/2)*tf.reduce_sum(tf.math.square(eta))
        for ll in range(self.noL):
            unorm += (self.rho/2)*tf.reduce_sum(tf.math.square(gamma[ll]))
        return unorm


    def data_fid_z(self,y,negC):
        v,z = y
        if self.noL > 1:
            z_curr = self.dictObj[0].divide_by_R(z[0])
            mu = self.updateZ_layer[0].mu
        else:
            z_curr = z[0]
            mu = self.updateZ_lastlayer.mu
        #tf.print('first mu: ',mu)
        Dx = self.dictObj[0].dmul_sp.freezeD(z_curr)
        reconErr = (mu/2)*self.reconstructionTerm_sp(self.cropAndMerge.crop(tf.squeeze(Dx,axis=-1)),negC)
        for layer in range(1,self.noL):
            if layer < self.noL - 1:
                z_curr = self.dictObj[layer].divide_by_R(z[layer])
                mu = self.updateZ_layer[layer].mu
            else:
                mu = self.updateZ_lastlayer.mu
                z_curr = z[layer]
            reconErr += (mu/2)*self.reconstructionTerm(z[layer - 1],self.dictObj[layer].dmul_sp.freezeD(z_curr),layer)
        return reconErr

    # Low-level Initialization Functions (x[layer],v,z[layer],eta,gamma[layer],Azero,Ax[layer],Bv,Bz[layer])

    def xinit(self,xprev,layer):
        if layer == 0:
            Dhx = self.dictObj[layer].dtmul_sp.freezeD(xprev)
        else:
            Rprev = util.rotate_dims_left(self.dictObj[layer - 1].divide_by_R.R,5) # Fix this later, can be implemented more efficiently
            Dhx = self.dictObj[layer].dtmul_sp.freezeD(xprev*Rprev)
        Rsquared = util.rotate_dims_left(tf.math.square(self.dictObj[layer].divide_by_R.R),5) # Fix this later, can be implemented more efficiently
        #Rsquared = tf.math.square(self.dictObj[layer].divide_by_R.R)
        #print('Rsquared_shape: ',Rsquared.shape)
        Dhx_scaled =  Dhx/Rsquared
        return Dhx_scaled


    def vinit(self,s,x_0,s_crop):
        s_HP,s_LP,compressed = s
        return (tf.expand_dims(s_HP,axis=-1),s_crop)

    #def zinit(self,xnext,Ax,layer):     
    #    return self.updateZ(xnext,Ax,tf.reshape(tf.cast(0.,tf.as_dtype(self.cmplxdtype.real_dtype)),(1,1,1,1,1)),layer)
    #def zinit_last(self,Ax):
    #    return self.updateZ_last(Ax,tf.reshape(tf.cast(0.,tf.as_dtype(self.cmplxdtype.real_dtype)),(1,1,1,1,1)))


    def gammainit(self,Ax,Bz,layer):
        return self.updateGamma(tf.cast(0.,tf.as_dtype(self.cmplxdtype.real_dtype)),Bz,Ax,layer,relax_bool=False)

    def etainit(self,Azero,Bzero,s_crop):
        return tf.cast(0.,self.cmplxdtype.real_dtype)*s_crop #tf.zeros(shape = (1,) + s_crop.shape[1:],dtype = self.cmplxdtype.real_dtype)


    # Low-level Update Functions (x[layer],v,z[layer],eta,gamma[layer],Azero,Ax[layer],Bv,Bz[layer])
    def updateX(self,z_prevlayer,z,gamma_scaled,layer,frozen=True):
        if layer == self.noL - 1:
            if frozen:
                return self.updateX_layer[layer]((z_prevlayer,z,gamma_scaled))
            else:
                return self.updateX_layer[layer].thawD((z_prevlayer,z,gamma_scaled))
        else:
            #R = tf.cast(util.rotate_dims_left(self.dictObj[layer].R,5),dtype=self.cmplxdtype)
            if frozen:
                return self.updateX_layer[layer]((z_prevlayer,self.dictObj[layer].divide_by_R(z),gamma_scaled))
            else:
                return self.updateX_layer[layer].thawD((z_prevlayer,self.dictObj[layer].divide_by_R(z),gamma_scaled))


    def updateV(self,x_0,eta,s_crop,frozen=True):
        if frozen:
            Dx = self.dictObj[0].dmul_sp.freezeD(x_0)
        else:
            Dx = self.dictObj[0].dmul_sp(x_0)
        Dx = tf.squeeze(Dx,axis = -1)
        Bzero = self.updateBzero((Dx,eta,s_crop))
        return (tf.expand_dims(self.cropAndMerge.merge((Bzero,Dx)),axis=-1),Bzero)


    def updateZ(self,x_nextlayer,Ax,gamma_scaled,layer,frozen=True):
        assert(layer < self.noL - 1)
        if self.strides[layer] == 2:
            return self.updateZ_downsample(x_nextlayer,Ax,gamma_scaled,layer,frozen)
        if frozen:
            Dx = self.dictObj[layer + 1].dmul_sp.freezeD(x_nextlayer)
        else:
            Dx = self.dictObj[layer + 1].dmul_sp(x_nextlayer)
        #z = self.updateZ_layer[layer]((self.IFFT[layer](Dx),self.IFFT[layer](Ax_relaxed),self.IFFT[layer](gamma_scaled)))
        z = self.updateZ_layer[layer]((Dx,Ax - gamma_scaled))
        # Changed this to switch from frequency to spatial
        return z

    # Changed to spatial domain from frequency domain.
    def updateZ_last(self,Ax,gamma_scaled):
        z = self.updateZ_lastlayer((Ax,gamma_scaled))
        return z

    def updateGamma(self,gamma_scaled,z,x,layer,relax_bool):
        if layer < self.noL - 1:
            #z_over_R = z/util.complexNum(util.rotate_dims_left(self.dictObj[layer].R,5))
            z_over_R = self.dictObj[layer].divide_by_R(z)
        else:
            z_over_R = z
        if relax_bool:
            return self.updateGamma_layer.halfstep((gamma_scaled,z_over_R,x))
        else:
            return self.updateGamma_layer((gamma_scaled,z_over_R,x))

    def updateEta(self,eta_over_rho,Azero,Bv,s_crop,relax_bool):
        if relax_bool:
            return self.updateeta.halfstep((eta_over_rho,Bv,s_crop))
        else:
            return self.updateeta((eta_over_rho,Bv,s_crop))


    def reconstructionTerm(self,z,Dx,layer):
        zminusDx = z - Dx
        return self.FFT[layer].parseval_sum(zminusDx)
    def reconstructionTerm_sp(self,z,Dx):
        return tf.math.reduce_sum(tf.math.square(z - Dx))

    def penaltyTerm(self,z,layer):
        if layer < self.noL - 1:
            if self.strides[layer] == 2:
                b = self.updateZ_layer[layer][4]((self.updateZ_layer[layer][3]((self.updateZ_layer[layer][0].b,self.updateZ_layer[layer][1].b)),self.updateZ_layer[layer][2].b))
                return tf.math.reduce_sum(tf.math.abs(b*z))
            else:
                return tf.math.reduce_sum(tf.math.abs(self.updateZ_layer[layer].get_lambda(self.rho)*z))
        else:
            return tf.math.reduce_sum(tf.math.abs(self.updateZ_lastlayer.get_lambda(self.rho)*z))


    def get_dict(self):
        D = []
        for ii in range(self.noL):
            D.append(self.dictObj[ii].divide_by_R.get_dict())
        return D

    def get_lambda(self):
        lmbda = []
        for ii in range(self.noL - 1):
             lmbda.append(self.updateZ_layer[ii].get_lambda(self.rho))
        lmbda.append(self.updateZ_lastlayer.get_lambda(self.rho))
        return lmbda

    def get_mu(self):
        mu = []
        for ii in range(self.noL - 1):
            mu.append(self.updateZ_layer[ii].mu)
        mu.append(self.updateZ_lastlayer.mu)
        return mu

    def get_alpha(self):
        return self.alpha


class GetNextIterX(tf.keras.layers.Layer):
    '''
      inputs: All must be in spatial domain.
    
        z_previouslayer: $\vz_{\ell - 1}^{(k)}$
        z_over_R: $\mR^{-1}\vz_{\ell}^{(k)}$
        gamma_scaled: $\frac{\vgamma_{\ell}^{(k)}}{\rho\sqrt{\mu_{\ell}}$

      outputs: Also in spatial domain
        x_scaled: \mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)}
    '''
    def __init__(self,rho,dictObj,FFT,IFFT,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dictObj = dictObj
        self.rho = rho
        self.FFT = FFT
        self.IFFT = IFFT
    def call(self,inputs):
        z_prevlayer,z_over_R,gamma_scaled = inputs
        preQinv = self.FFT(self.dictObj.dtmul_sp.freezeD(z_prevlayer) + self.rho*z_over_R + gamma_scaled)
        
        postQinv = self.IFFT(self.dictObj.freezeD(preQinv))
        return postQinv
    def thawD(self,inputs):
        z_prevlayer,z_over_R,gamma_scaled = inputs
        preQinv = self.FFT(self.dictObj.dtmul_sp(z_prevlayer) + self.rho*z_over_R + gamma_scaled)
        
        postQinv = self.IFFT(self.dictObj(preQinv))
        return postQinv
    def get_config(self):
        return {'rho': self.rho}



class GetNextIterZ(tf.keras.layers.Layer,ppg.PostProcess):
    '''
     inputs: All must be in spatial domain.

       Dx_nextlayer: \mD_{\ell + 1}\vx_{\ell + 1}^{(k + 1)}
       negqnt_Ax_minus_gamma_scaled: \mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} - \frac{\vgamma_{\ell}^{(k)}}{\rho\sqrt{\mu_{\ell}}

     outputs: unlike inputs, output is in spatial domain
       z: \vz_{\ell}^{(k + 1)}
    '''
    def __init__(self,rho,mu_init,mu_nextlayer,dictObj,dictObj_nextlayer,b_init,*args,**kwargs):
        #super().__init__(*args,**kwargs)
        tf.keras.layers.Layer.__init__(self,*args,**kwargs)
        self.rho = rho
        
        self.mu_nextlayer = mu_nextlayer
        self.dictObj = dictObj
        self.dictObj_nextlayer = dictObj_nextlayer
        with tf.name_scope(self.name):
            self.mu = tf.Variable(mu_init,trainable=True,dtype=tf.as_dtype(self.dtype).real_dtype,name='mu')
            self.b = tf.Variable(b_init,trainable=True,dtype=tf.as_dtype(self.dtype).real_dtype,name='b')
        #self.relu = tf.keras.layers.ReLU(dtype=tf.as_dtype(self.dtype).real_dtype)
        self.relu = util.Shrinkage(dtype=tf.as_dtype(self.dtype).real_dtype)
        self.ifft = ifft
        ppg.PostProcess.add_update(self.b.name,self._update_b)
        ppg.PostProcess.add_update(self.mu.name,self._update_mu)

    def _update_b(self):
        return [self.b.assign(tf.where(self.b < 0.,tf.cast(0,dtype=tf.as_dtype(self.dtype).real_dtype),self.b)),]
    def _update_mu(self):
        return [self.mu.assign(tf.where(self.mu < 1e-3,tf.cast(1e-3,dtype=tf.as_dtype(self.dtype).real_dtype),self.mu))]

    def call(self,inputs):
        # Inputs and output are in spatial domain.
        Dx_nextlayer,negqnt_Ax_minus_gamma_scaled = inputs
        currR = tf.reshape(self.dictObj.divide_by_R.R,shape=(1,1,1,self.dictObj.divide_by_R.R.shape[4],1))
        leadingFactor = 1/(self.mu_nextlayer + self.rho*self.mu/currR**2)
        return leadingFactor*self.relu((tf.cast(self.mu_nextlayer,dtype=self.dtype)*Dx_nextlayer + self.dictObj.divide_by_R(self.rho*self.mu*negqnt_Ax_minus_gamma_scaled),self.b))

    def get_lambda(self,rho):
        #tf.print('lambda: ',self.b)
        return self.b

    def get_config(self):
        return {'rho': self.rho}


class GetNextIterZ_lastlayer(tf.keras.layers.Layer,ppg.PostProcess):
    '''
      inputs: All must be in spatial domain.

        Ax: = \mR_L^{-1}\vx_L^{(k + 1)} (Actual Ax would be negative, but no reason to store negative version of same variable.)
        gamma_scaled: \frac{\gamma_L^{(k)}}{\rho\sqrt{\mu_L}}

      outputs: Unlike inputs, output is in spatial domain
        z_over_R: \mR_L^{-1}\vz_L^{(k + 1)}
    '''
    def __init__(self,rho,mu_init,dictObj,b_init,*args,**kwargs):
        #super().__init__(*args,**kwargs)
        tf.keras.layers.Layer.__init__(self,*args,**kwargs)
        
        self.dictObj = dictObj
        with tf.name_scope(self.name):
            self.mu = tf.Variable(mu_init,trainable=True,dtype=tf.as_dtype(self.dtype).real_dtype,name='mu')
            self.b = tf.Variable(b_init/(rho*mu_init),trainable=True,dtype=tf.as_dtype(self.dtype).real_dtype,name='b')
        self.relu = util.Shrinkage(dtype=tf.as_dtype(self.dtype).real_dtype)
        ppg.PostProcess.add_update(self.b.name,self._update_b)
        ppg.PostProcess.add_update(self.mu.name,self._update_mu)

    def _update_b(self):
        return [self.mu.assign(tf.where(self.mu < 0.,tf.cast(0,dtype=tf.as_dtype(self.dtype).real_dtype),self.mu))]

    def _update_mu(self):
        return [self.mu.assign(tf.where(self.mu < 1e-3,tf.cast(1e-3,dtype=tf.as_dtype(self.dtype).real_dtype),self.mu))]

    def call(self,inputs):
        Ax,gamma_scaled = inputs
        R = tf.reshape(self.dictObj.divide_by_R.R,shape=(1,1,1,self.dictObj.divide_by_R.R.shape[4],1))
        return self.relu((Ax - gamma_scaled,R*self.b))

    def get_lambda(self,rho):
        return self.b*(rho*self.mu)



# This code is different from the code in multilayerCSC_ADMM because of the different relaxation scheme. There is not separate computation for Ax_relaxed.
class GetNextIterGamma(tf.keras.layers.Layer):
    '''
      Initialization: Need tf.Variable alpha for overrelaxation.
      inputs: All must be in spatial domain.

        gamma_scaled: \frac{\gamma_{\ell}^{(k)}}{\rho\sqrt{\mu_{\ell}}
        z_over_R: \mR_{\ell}^{-1}\vz_{\ell}^{(k + 1)}
        Ax: \mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} (Actual Ax would be negative, but no reason to store negative version of same variable)

      outputs: Also in spatial domain
        gamma_scaled: \frac{\vgamma_{\ell}^{(k + 1)}}{\rho\sqrt{\mu_{\ell}}
    '''
    def __init__(self,alpha,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.alpha = alpha
    def call(self,inputs):
        gamma_scaled,z_over_R,Ax = inputs
        return gamma_scaled + z_over_R - Ax
    def halfstep(self,inputs):
        ''' same update equation, only with overrelaxation (or under-relaxation).'''
        gamma_scaled,z_over_R,Ax = inputs
        return gamma_scaled + (self.alpha - tf.cast(1.,tf.as_dtype(self.dtype)))*(z_over_R - Ax)

class UpdateEta(tf.keras.layers.Layer):
    def __init__(self,alpha,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.alpha = alpha
    def call(self,inputs):
        eta_scaled,croppedv,negC = inputs
        return eta_scaled + croppedv - negC
    def halfstep(self,inputs):
        eta_scaled,croppedv,negC = inputs
        return eta_scaled + (self.alpha - tf.cast(1.,tf.as_dtype(self.dtype)))*(croppedv - negC)


class GetNextIterBZero(tf.keras.layers.Layer):
    def __init__(self,cropAndMerge,mu,rho,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cropAndMerge = cropAndMerge
        self.mu = mu
        self.rho = rho
    def call(self,inputs):
        Dx_squeezed,eta,negC = inputs
        return (self.mu*self.cropAndMerge.crop(Dx_squeezed) + self.rho*(negC - eta))/(self.mu + self.rho)

class CropPadObject:
    def __init__(self,signalSz,strides,kernelSz,dtype):
        self.dtype = dtype
        padding = 0
        for ii in range(len(kernelSz)):
            stride_factor = 1
            for jj in range(0,ii):
                stride_factor = stride_factor*strides[jj]
            padding = padding + stride_factor*(kernelSz[ii] - 1)
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
        pad = tf.keras.layers.ZeroPadding2D(padding = self.paddingTuple,dtype=self.dtype)
        trues = tf.fill((1,signal_sz[0],signal_sz[1],1),1)
        mask = tf.cast(pad(trues),'bool')
        self.merge = Merge(pad,mask,dtype=self.dtype)
    def get_fft_size(self,signalSz,strides):
        fftSz = []
        fftSz.append((signalSz[0] + self.paddingTuple[0][0] + self.paddingTuple[0][1],signalSz[1] + self.paddingTuple[1][0] + self.paddingTuple[1][1]))
        ii = 0
        for stride in strides:
            fftSz.append((int(fftSz[ii][0]/stride),int(fftSz[ii][1]/stride)))
            ii += 1
        return fftSz

class Merge(tf.keras.layers.Layer):
    def __init__(self,pad,mask,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pad = pad
        self.mask = mask
    def call(self,inputs):
        x,y = inputs
        return tf.where(self.mask,self.pad(x),y)
    def get_config(self):
        return {'mask': self.mask, 'pad': self.pad}
