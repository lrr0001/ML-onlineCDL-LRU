import tensorflow as tf
import matrix_decompositions_tf as fctr
import optmz
import jpeg_related_functions as jrf
import transforms as transf
import util
import numpy as np
import post_process_grad as ppg

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
    def __init__(self,rho,alpha_init,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,D,n_components,noi,noL,cmplxdtype,longitstat=False,*args,**kwargs):
        rho,mu_init = self.init_param(rho,alpha_init,mu_init,cropAndMerge,noi,noL,cmplxdtype,longitstat,*args,**kwargs)
        self.initializeLayers(rho,mu_init,alpha_init,util.makelist(b_init,noL),noL,fftSz,strides,D,n_components,cmplxdtype)
        qY = tf.reshape(qY,(1,1,1,64))
        qUV = tf.reshape(qUV,(1,1,1,64))
        self.initializeJPEGLayers(cmplxdtype,qY,qUV)
        
    def init_param(self,rho,alpha_init,mu_init,cropAndMerge,noi,noL,cmplxdtype,longitstat,*args,**kwargs):
        self.longitstat = longitstat
        self.cmplxdtype = cmplxdtype
        dtype = cmplxdtype.real_dtype
        rho = tf.cast(rho,dtype = cmplxdtype.real_dtype)
        mu_init = util.makelist(mu_init,noL)
        self.noL = noL
        optmz.ADMM.__init__(self=self,rho = rho,alpha=alpha_init,noi = noi,dtype=dtype,*args,**kwargs)
        #self.alpha = tf.Variable(initial_value=alpha_init,trainable=True,name = 'alpha',dtype=dtype)
        self.cropAndMerge = cropAndMerge
        return rho,mu_init
    def get_config(self):
        config_dict = {'complex_dtype': self.cmplxdtype,
                       'num_of_Layers': self.noL,
                       'strides': self.strides,
                       'qY': self.qY,
                       'qUV': self.qUV,
                       'rho': self.rho,
                       'noi': self.noi,
                       'record_iteration_stats': self.longitstat}
        return config_dict

    def initializeLayers(self,rho,mu_init,alpha_init,b_init,noL,fftSz,strides,D,n_components,cmplxdtype):
        self.strides = strides
        self.dictObj = []
        self.updateX_layer = []

        for ii in range(noL):
            self.dictObj.append(self.build_dict_obj(fftSz[ii],D[ii],rho,n_components,cmplxdtype,ii))
            self.updateX_layer.append(GetNextIterX(tf.cast(rho,dtype=cmplxdtype),self.dictObj[ii],dtype=cmplxdtype))

        self.FFT,self.IFFT,self.FFT_factor = self.build_fft_layers(fftSz,noL)

        #self.relax_layer = GetRelaxedAx(alpha=tf.cast(self.alpha,dtype=cmplxdtype),dtype=cmplxdtype)
        self.relax_layer = GetRelaxedAx(alpha=alpha_init,dtype=self.cmplxdtype)
        reversed_updateZ_layer = []
        #self.updateZ_layer = [[],]*(noL - 1)
        self.updateGamma_layer = GetNextIterGamma(dtype=cmplxdtype)

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

    def initializeJPEGLayers(self,cmplxdtype,qY,qUV):
        #self.W = jrf.YUV2JPEG_Coef(dtype=cmplxdtype.real_dtype)
        #self.Wt = jrf.JPEG_Coef2YUV(dtype=cmplxdtype.real_dtype)
        self.W = jrf.RGB2JPEG_Coef(dtype=cmplxdtype.real_dtype)
        self.Wt = jrf.JPEG_Coef2RGB(dtype=cmplxdtype.real_dtype)
        self.updatev = jrf.ZUpdate_JPEG_Implicit(qY,qUV,self.W,self.Wt,dtype = cmplxdtype.real_dtype)
        #self.updatev = jrf.ZUpdate_JPEG(mu,rho,qY,qUV,self.W,self.Wt,dtype = cmplxdtype.real_dtype)
        #self.relax0 = jrf.Relax_SmoothJPEG(self.alpha, dtype=cmplxdtype.real_dtype) # could change these 2 so that eta update gets alpha
        #self.updateeta = jrf.GammaUpdate_JPEG(self.alpha,dtype=cmplxdtype.real_dtype)
        self.qY = qY
        self.qUV = qUV
        

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
            return fctr.dictionary_object2D_init_full(fftSz=fftSz,D = tf.convert_to_tensor(D),rho=tf.cast(rho,dtype=cmplxdtype),objname='dict_layer' + str(layer),n_components = n_components)
        else:
            return fctr.dictionary_object2D_init(fftSz=fftSz,D = tf.convert_to_tensor(D),rho=tf.cast(rho,dtype=cmplxdtype),objname='dict_layer' + str(layer),n_components=n_components)

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
            zUpdate = GetNextIterZFreq(rho,self.IFFT[layer],tf.cast(mu_init,cmplxdtype.real_dtype),munext,dictObj,nextdictObj,b,dtype=cmplxdtype,name='Z_layer' + str(layer))
            return zUpdate,zUpdate.mu

    def build_updateZ_lastlayer(self,fftSz,nof,rho,mu_init,dictObj,b_init, cmplxdtype):
        b = tf.zeros(self.get_b_shape(fftSz,nof),dtype=cmplxdtype.real_dtype) + b_init
        lastlayer = GetNextIterZFreq_lastlayer(rho,self.IFFT[self.noL - 1],tf.cast(mu_init,self.cmplxdtype.real_dtype),dictObj,b,dtype=cmplxdtype,name='Z_layer' + str(self.noL - 1))
        return lastlayer,lastlayer.mu

    def init_itstats(self,s):
        if not self.longitstat:
            return []
        xprev = None
        yprev = None
        uprev = None
        Byprev = None
        prevs = (xprev,yprev,uprev,Byprev)
        x_improvements = []
        v_improvements = []
        vplusz_improvements = []
        y_improvements = (v_improvements,vplusz_improvements)
        improvements = (x_improvements,y_improvements)
        recon_err = []
        cmprssd_recon_err = []
        primal_err = []
        cnstrnt_err = []
        errs = (recon_err,cmprssd_recon_err,primal_err,cnstrnt_err)
        return (prevs,improvements,errs)

    def itstats_record(self,x,y,u,Ax,Ax_relaxed,By,negC,itstats):
        if not self.longitstat:
            return []
        prevs,improvements,errs = itstats
        xprev,yprev,uprev,Byprev = prevs
        x_improvements,y_improvements = improvements
        v_improvements,vplusz_improvements = y_improvements
        recon_err,cmprssd_recon_err,primal_err,cnstrnt_err = errs


        # compute errors
        currReconErr = self.reconstructionErrors(x,y)
        recon_err.append(currReconErr)
        currPrimalErr = currReconErr + self.penaltyErrors(y)
        primal_err.append(currPrimalErr)
        s_LF,QWs = negC
        comprssd_s = self.Wt(QWs)
        Dx = self.dictObj[0].dmul.freezeD(x[0])
        Dx_real = tf.squeeze(self.IFFT[0](Dx),axis=-1)
        Dx_cropped = self.cropAndMerge.crop(Dx_real)
        currCmprssdReconResid = comprssd_s - Dx_cropped
        cmprssd_recon_err.append(tf.math.sqrt(tf.reduce_mean(currCmprssdReconResid**2)))
        cnstrnt_err.append(self.constraintErrors(x,y,Ax,By,negC))
        


        # Compute improvements
        if xprev is not None:
            prevLagrang = self.evaluateLagrangian(xprev,yprev,uprev,Byprev,negC)
            currLagrang = self.evaluateLagrangian(x,yprev,uprev,Byprev,negC)
            x_improvements.append(prevLagrang - currLagrang)
            prevRelaxedLagrang = self.evaluateRelaxedLagrangian(x,yprev,uprev,Ax,Byprev,negC)
            currRelaxedLagrang = self.evaluateRelaxedLagrangian(x,y,uprev,Ax,By,negC)
            vplusz_Lchange = prevRelaxedLagrang - currRelaxedLagrang
            if self.noL > 1:
                if self.strides[0] == 2:
                    mu = self.updateZ_layer[0][0].mu
                    #mu = self.updateZ_layer[0]['downsampled'].mu
                else:
                    mu = self.updateZ_layer[0].mu
            else:
                mu = self.updateZ_lastlayer.mu
            vprev,zprev = yprev
            Bzprev = Byprev
            etaprev,gammaprev = uprev
            vprev_cropped = self.cropAndMerge.crop(tf.squeeze(self.IFFT[0](vprev),axis=-1))
            augLangVprev = mu/2*self.reconstructionTerm(vprev_cropped,Dx_cropped,0)*self.FFT[0].fft_factor + self.jpegConstraint_relaxed(etaprev,Bvprev,negC)
            v,z = y
            Bz = By
            eta,gamma = u
            v_cropped = self.cropAndMerge.crop(tf.squeeze(self.IFFT[0](v),axis=-1))
            augLangVcurr = mu/2*self.reconstructionTerm(v_cropped,Dx_cropped,0)*self.FFT[0].fft_factor + self.jpegConstraint_relaxed(etaprev,Bv,negC)
            v_Lchange = augLangVprev - augLangVcurr
            v_improvements.append(v_Lchange)
            vplusz_improvements.append(vplusz_Lchange)
            y_improvements = (v_improvements,vplusz_improvements)

        prevs = (x,y,u,By)
        improvements = (x_improvements,y_improvements)
        errs = (recon_err,cmprssd_recon_err,primal_err,cnstrnt_err)

        itstats = (prevs,improvements,errs)
        return itstats


    def get_output(self,s,y,u,By,negC,itstats):
        s_LF,QWs = negC
        for layer in range(self.noL - 1,0,-1):
            x,Ax = self.xstep_trunc(y,u,By,negC,layer)
            #u = self.relax_trunc(u,Ax,By,negC,layer)
            Ax_relaxed = self.relax_trunc(Ax,By,negC,layer)
            y,By = self.ystep_trunc(x,u,Ax_relaxed,negC,layer,frozen=False)
            #y,By = self.ystep_trunc(x,u,Ax,negC,layer,frozen=False)
            #u = self.ustep_trunc(u,Ax,By,negC,layer)
            u = self.ustep_trunc(u,Ax_relaxed,By,negC,layer)

        x,Ax = self.xstep_trunc(y,u,By,negC,layer=0)
        #Ax_relaxed = self.relax(Ax,By,negC)
        #y,By = self.ystep(x,u,Ax_relaxed,negC)
        v,z = y
        #s_LF,QWs = negC
        Dx = self.IFFT[0](self.dictObj[0].dmul(x[0]))
        vreal = self.IFFT[0](v)
        return (self.cropAndMerge.crop(tf.squeeze(Dx,axis=-1)) + s_LF,self.cropAndMerge.crop(tf.squeeze(vreal,axis=-1)) + s_LF,itstats)

    def get_b_shape(self,fftSz,M):
        #return [1,fftSz[0],fftSz[1],M,1,]
        #return [1,1,1,M,1]
        return (1,)

    # Function is no longer necessary
    #def get_downsampled_b_shape(self,fftSz,M):
        # This is an error
    #    print('Stride is currently broken, and the downsampled z-updates within the same layer should probably share the same b.')
    #    raise NotImplementedError
    #    return ((1,int(fftSz[0]/2),int(fftSz[1]/2),M,1),(1,fftSz[0],int(fftSz[1]/2),M,1))

    # High-level Initialization Functions (x,y,u,Ax,By,C)
    def get_negative_C(self,s):
        s_HF,s_LF,compressed = s
        Ws = self.W(compressed)
        Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        QWs = jrf.threeChannelQuantize(Ws,self.qY,self.qUV,Yoffset)
        return (self.cropAndMerge.crop(s_LF),QWs)
    def init_x(self,s,negC):
        s_HF,temp1,temp2 = s
        s_LF,QWs = negC
        x = []
        x.append(self.xinit(self.FFT[0](util.addDim(s_HF)),layer = 0))
        for ii in range(1,self.noL):
            if self.strides[ii - 1] == 2:
                x.append(self.xinit(util.freq_downsample(x[ii - 1]),layer=ii))
            else:
                x.append(self.xinit(x[ii - 1],layer = ii))
        Ax = ([-QWs[ii] for ii in range(len(QWs))],x)
        return x,Ax
    def init_y(self,s,x,Ax,negC):
        Azero,Ax_layers = Ax
        v = self.vinit(x[0],negC)
        z = []
        for ii in range(self.noL - 1):
            z.append(tf.cast(util.rotate_dims_left(self.dictObj[ii].divide_by_R.R,5),self.cmplxdtype)*x[ii])
            #z.append(self.zinit(xnext=x[ii + 1],Ax=Ax_layers[ii],layer=ii))
        #z.append(self.zinit_last(Ax=Ax_layers[self.noL - 1]))
        z.append(x[self.noL - 1])
        y = (v,z)
        By = z
        return y,By
    def init_u(self,s,Ax,By,negC):
        Azero,Ax_layers = Ax
        Bz = By
        gamma = []
        for ii in range(self.noL):
            gamma.append(self.gammainit(Ax_layers[ii],Bz[ii],ii))
        s_LF,QWs = negC
        #eta = self.etainit(Bv,QWs)
        #return eta,gamma
        return gamma


    # High-level Update Functions for each iteration (x,y,u,Ax,By)
    def xstep(self,y,u,By,negC):
        return self.xstep_trunc(y,u,By,negC,layer=self.noL - 1)
        #v,z = y
        #eta,gamma = u
        #x = []
        #x.append(self.updateX(z_prevlayer=v,z=z[0],gamma_scaled = gamma[0],layer=0))
        #for ii in range(1,self.noL):
        #    if self.strides[ii - 1] == 2:
        #        x.append(self.updateX(z_prevlayer=util.freq_downsample(z[ii - 1]),z = z[ii],gamma_scaled = gamma[ii],layer=ii))
        #    else:
        #        x.append(self.updateX(z_prevlayer=z[ii - 1],z = z[ii],gamma_scaled = gamma[ii],layer=ii))
        #return x,(0.,x)
    def relax(self,Ax,By,negC):
        return self.relax_trunc(Ax,By,negC,layer=self.noL)

    #def relax(self,u,Ax,By,negC):
    #    return self.relax_trunc(u,Ax,By,negC,layer=self.noL)
        #Azero,x = Ax
        #s_LF,QWs = negC
        #rootmux = []
        #Bv,z = By
        #for ii in range(self.noL):
        #    rootmux.append(self.relax_layers(x[ii],z[ii],ii))
        #AzeroplusC = self.relax_zero(Bv,QWs)
        #AzeroplusC = None
        #return AzeroplusC,rootmux
    def ystep(self,x,u,Ax_relaxed,negC):
        # It is good practice to avoid code redundancy, but here the last layer is distinct from ystep_trunc. y_step trunc does not freeze the dictionary weights and also does not use updateZlast. If-else logic could resolve these distinctions, but I find it clearer to leave these as two separate functions.
        #eta,gamma = u
        gamma = u
        AzeroplusC,Ax_layers = Ax_relaxed
        z = []
        for ii in range(self.noL - 1):
            z.append(self.updateZ(x_nextlayer=x[ii + 1],Ax_relaxed=Ax_layers[ii],gamma_scaled=gamma[ii],layer=ii))
        z.append(self.updateZ_last(Ax_relaxed=Ax_layers[self.noL - 1],gamma_scaled=gamma[self.noL - 1]))
        #v,Bv = self.updateV(x[0],eta,negC)
        v = self.updateV(x[0],negC)
        y = (v,z)
        #By = (Bv,z)
        By = z
        return y,By
    def ustep(self,u,Ax_relaxed,By,negC):
        return self.ustep_trunc(u,Ax_relaxed,By,negC,layer=self.noL)




        #eta,gamma = u
        #AzeroplusC,Ax_layers = Ax_relaxed
        #Bv,z = By
        #for ii in range(self.noL):
        #    gamma[ii] = self.updateGamma(gamma[ii],z[ii],Ax_layers[ii],ii)
        #s_LF,QWs = negC
        #eta = self.updateEta(eta,Bv,QWs)
        #return eta,gamma

    def xstep_trunc(self,y,u,By,negC,layer):
        v,z = y
        #eta,gamma = u
        gamma = u
        x = []
        x.append(self.updateX(z_prevlayer=v,z=z[0],gamma_scaled = gamma[0],layer=0))
        for ii in range(1,layer + 1):
            if self.strides[ii - 1] == 2:
                x.append(self.updateX(z_prevlayer=util.freq_downsample(z[ii - 1]),z = z[ii],gamma_scaled = gamma[ii],layer=ii))
            else:
                x.append(self.updateX(z_prevlayer=z[ii - 1],z = z[ii],gamma_scaled = gamma[ii],layer=ii))
        return x,(0.,x)
    def relax_trunc(self,Ax,By,negC,layer):
        Azero,x = Ax
        s_LF,QWs = negC
        rootmux = []
        #Bv,z = By
        z = By
        for ii in range(layer):
            rootmux.append(self.relax_layers(x[ii],z[ii],ii))
            
        #AzeroplusC = self.relax_zero(Bv,QWs)
        AzeroplusC = None
        return AzeroplusC,rootmux

    #def relax_trunc(self,u,Ax,By,negC,layer):
    #    #eta,gamma = u
    #    gamma = u
    #    AzeroplusC,Ax_layers = Ax
    #    #Bv,z = By
    #    z = By
    #    for ii in range(layer):
    #        gamma[ii] = self.relax_layers(gamma[ii],z[ii],Ax_layers[ii],ii)
    #    #s_LF,QWs = negC
    #    #eta = self.updateEta(eta,Bv,QWs)
    #    #return eta,gamma
    #    return gamma        

    def ystep_trunc(self,x,u,Ax_relaxed,negC,layer,frozen=True):
        #eta,gamma = u
        gamma = u
        AzeroplusC,Ax_layers = Ax_relaxed
        z = []
        for ii in range(layer - 1):
            z.append(self.updateZ(x_nextlayer=x[ii + 1],Ax_relaxed=Ax_layers[ii],gamma_scaled=gamma[ii],layer=ii))
        z.append(self.updateZ(x_nextlayer=x[layer],Ax_relaxed=Ax_layers[layer - 1],gamma_scaled=gamma[layer - 1],layer=layer - 1,frozen=frozen))
        #z.append(self.updateZ_last(Ax_relaxed=Ax_layers[self.noL - 1],gamma_scaled=gamma[self.noL - 1]))
        #v,Bv = self.updateV(x[0],eta,negC)
        v= self.updateV(x[0],negC)
        y = (v,z)
        #By = (Bv,z)
        By = z
        return y,By
    def ustep_trunc(self,u,Ax_relaxed,By,negC,layer):
        #eta,gamma = u
        gamma = u
        AzeroplusC,Ax_layers = Ax_relaxed
        #Bv,z = By
        z = By
        for ii in range(layer):
            gamma[ii] = self.updateGamma(gamma[ii],z[ii],Ax_layers[ii],ii)
        s_LF,QWs = negC
        #eta = self.updateEta(eta,Bv,QWs)
        #return eta,gamma
        return gamma 

    #High-level Langrangian Evalaluation Functions
    def evaluateLagrangian(self,x,y,u,By,negC):
        recErr = self.reconstructionErrors(x,y)
        penaltyErr = self.penaltyErrors(y)
        cnstrErr = self.augConstraintErrors(x,u,By,negC)
        return recErr + penaltyErr + cnstrErr
    def evaluateRelaxedLagrangian(self,x,y,u,Ax,By,negC):
        recErr = self.reconstructionErrors(x,y)
        penaltyErr = self.penaltyErrors(y)
        cnstrErr = self.augConstraintErrors_relaxed(u,Ax,By,negC)
        return recErr + penaltyErr + cnstrErr
    def reconstructionErrors(self,x,y):
        v,z = y
        if self.noL > 1:
            if self.strides[0] == 2:
                mu = self.updateZ_layer[0][0].mu
            #mu = self.updateZ_layer[0]['downsampled'].mu
            else:
                mu = self.updateZ_layer[0].mu
        else:
            mu = self.updateZ_lastlayer.mu
        reconErr = (mu/2)*self.reconstructionTerm(v,self.dictObj[0].dmul.freezeD(x[0]),0)
        for layer in range(1,self.noL):
            if layer < self.noL - 1:
                if self.strides[layer] == 2:
                    mu = self.updateZ_layer[layer][0].mu
                    #mu = self.updateZ_layer[layer]['downsampled'].mu
                else:
                    mu = self.updateZ_layer[layer].mu
            else:
                mu = self.updateZ_lastlayer.mu
            if self.strides[layer - 1] == 2:
                reconErr += (mu/2)*self.reconstructionTerm(util.freq_downsample(z[layer - 1]),self.dictObj[layer].dmul.freezeD(x[layer]), layer)
            else:
                reconErr += (mu/2)*self.reconstructionTerm(z[layer - 1],self.dictObj[layer].dmul.freezeD(x[layer]), layer)
        return reconErr

    #def get_obj(self,y):
    #    return self.penaltyErrors_z(y)

    def get_obj(self,y,negC):
        return self.data_fid_z(y,negC) + self.penaltyErrors_z(y)

    def data_fid_z(self,y,negC):
        v,z = y
        if self.noL > 1:
            z_curr = self.dictObj[0].divide_by_R(z[0])
            if self.strides[0] == 2:
                mu = self.updateZ_layer[0][0].mu
            else:
                mu = self.updateZ_layer[0].mu
        else:
            z_curr = z[0]
            mu = self.updateZ_lastlayer.mu
        #tf.print('first mu: ',mu)
        Dx = self.IFFT[0](self.dictObj[0].dmul.freezeD(z_curr))
        reconErr = (mu/2)*self.reconstructionTerm_sp(self.updateV_subfun(Dx,negC),Dx)
        for layer in range(1,self.noL):
            if layer < self.noL - 1:
                z_curr = self.dictObj[layer].divide_by_R(z[layer])
                if self.strides[layer] == 2:
                    mu = self.updateZ_layer[layer][0].mu
                    #mu = self.updateZ_layer[layer]['downsampled'].mu
                else:
                    mu = self.updateZ_layer[layer].mu
            else:
                mu = self.updateZ_lastlayer.mu
                z_curr = z[layer]
            if self.strides[layer - 1] == 2:
                reconErr += (mu/2)*self.reconstructionTerm(util.freq_downsample(z[layer - 1]),self.dictObj[layer].dmul.freezeD(z_curr), layer)
            else:
                reconErr += (mu/2)*self.reconstructionTerm(z[layer - 1],self.dictObj[layer].dmul.freezeD(z_curr),layer)
            #tf.print('mu: ',mu)
        return reconErr

    def data_fid_x(self,xnegC):
        if self.noL > 1:
            if self.strides[0] == 2:
                mu = self.updateZ_layer[0][0].mu
            #mu = self.updateZ_layer[0]['downsampled'].mu
            else:
                mu = self.updateZ_layer[0].mu
        else:
            mu = self.updateZ_lastlayer.mu
        Dx = self.dictObj[0].dmul.freezeD(x[0])
        reconErr = (mu/2)*self.reconstructionTerm(self.updateV_subfun(Dx,negC),Dx,layer=0)
        for layer in range(1,self.noL):
            if layer < self.noL - 1:
                if self.strides[layer] == 2:
                    mu = self.updateZ_layer[layer][0].mu
                    #mu = self.updateZ_layer[layer]['downsampled'].mu
                else:
                    mu = self.updateZ_layer[layer].mu
            else:
                mu = self.updateZ_lastlayer.mu
            if self.strides[layer - 1] == 2:
                reconErr += (mu/2)*self.reconstructionTerm(util.freq_downsample(x[layer - 1]),self.dictObj[layer].dmul.freezeD(x[layer]),layer) # Multiply x[layer - 1] by R
            else:
                reconErr += (mu/2)*self.reconstructionTerm(x[layer - 1],self.dictObj[layer].dmul.freezeD(x[layer]),layer)
        return reconErr
    def penaltyErrors(self,y):
        v,z = y
        penaltyErr = 0
        for layer in range(self.noL):
            penaltyErr += self.penaltyTerm(self.IFFT[layer](z[layer]),layer)
        return penaltyErr

    def penaltyErrors_x(self,x):
        penaltyErr = 0
        for layer in range(self.noL):
            R = tf.reshape(self.dictObj[layer].divide_by_R.R,shape=(1,1,1,self.dictObj[layer].divide_by_R.R.shape[4],1))
            penaltyErr += self.penaltyTerm(self.IFFT[layer](R*x[layer]),layer)
        R = tf.reshape(self.dictObj[-1].divide_by_R.R,shape=(1,1,1,self.dictObj[-1].divide_by_R.R.shape[4],1))
        penaltyErr += self.penaltyTerm(self.IFFT[-1](R*x[-1]),self.noL - 1)
        return penaltyErr

    def penaltyErrors_z(self,y):
        v,z = y
        #return tf.math.reduce_sum(tf.math.abs(v))
        penaltyErr = 0
        for layer in range(self.noL - 1):
            penaltyErr += self.penaltyTerm(self.IFFT[layer](z[layer]),layer)
        R = tf.reshape(self.dictObj[-1].divide_by_R.R,shape=(1,1,1,self.dictObj[-1].divide_by_R.R.shape[4],1))
        penaltyErr += self.penaltyTerm(self.IFFT[-1](tf.cast(R,z[-1].dtype)*z[-1]),self.noL - 1)
        return penaltyErr

    def constraintErrors(self,x,y,Ax,By,negC):
        s_LF,QWs = negC
        Bv,Bz = By
        constraintErr = self.jpegConstraint((0.,0.,0.),Bv,QWs)
        for layer in range(self.noL):
            if layer < self.noL - 1:
                if self.strides[layer] == 2:
                    mu = self.updateZ_layer[layer][0].mu
                else:
                    mu = self.updateZ_layer[layer].mu
                #z_over_R = Bz[layer]/util.complexNum(util.rotate_dims_left(self.dictObj[layer].R,5))
                z_over_R = self.dictObj[layer].divide_by_R(Bz[layer])
            else:
                mu = self.updateZ_lastlayer.mu
                z_over_R = Bz[layer]
            constraintErr += (mu/2)*self.zxConstraint(tf.cast(0.,self.cmplxdtype),x[layer],z_over_R)/self.FFT_factor[layer]
        return constraintErr        

    def augConstraintErrors(self,x,u,By,negC):
        s_LF,QWs = negC
        Bv,Bz = By
        eta,gamma = u
        constraintErr = self.jpegConstraint(eta,Bv,QWs)
        for layer in range(self.noL):
            if layer < self.noL - 1:
                if self.strides[layer] == 2:
                    mu = self.updateZ_layer[layer][0].mu
                else:
                    mu = self.updateZ_layer[layer].mu
                #z_over_R = Bz[layer]/util.complexNum(util.rotate_dims_left(self.dictObj[layer].R,5))
                z_over_R = self.dictObj[layer].divide_by_R(Bz[layer])
            else:
                mu = self.updateZ_lastlayer.mu
                z_over_R = Bz[layer]
            constraintErr += (mu/2)*self.zxConstraint(gamma[layer],x[layer],z_over_R)/self.FFT_factor[layer]
        return constraintErr
    def augConstraintErrors_relaxed(self,u,Ax,By,negC):
        Azero,Ax = Ax
        Bv,Bz = By
        eta,gamma = u
        s_LF,QWs = negC
        constraintErr = self.jpegConstraint(eta,Bv,QWs)
        for layer in range(self.noL):
            if layer < self.noL - 1:
                if self.strides[layer] == 2:
                    mu = self.updateZ_layer[layer][0].mu
                    #mu = self.updateZ_layer[layer]['downsampled'].mu
                else:
                    mu = self.updateZ_layer[layer].mu
                Bz_over_R = self.dictObj[layer].divide_by_R(Bz[layer])
            else:
                mu = self.updateZ_lastlayer.mu
                Bz_over_R = Bz[layer]
            constraintErr += (mu/2)*self.zxConstraint_relaxed(gamma[layer],Ax[layer],Bz_over_R)/self.FFT_factor[layer]
        return constraintErr

    # Low-level Initialization Functions (x[layer],v,z[layer],eta,gamma[layer],Azero,Ax[layer],Bv,Bz[layer])
    def xinit(self,xprev,layer):
        if layer == 0:
            Dhx = self.dictObj[layer].dhmul.freezeD(xprev)
        else:
            Rprev = util.rotate_dims_left(self.dictObj[layer - 1].divide_by_R.R,5) # Fix this later, can be implemented more efficiently
            Dhx = self.dictObj[layer].dhmul.freezeD(xprev*tf.cast(Rprev,self.cmplxdtype))
        Rsquared = util.rotate_dims_left(tf.math.square(self.dictObj[layer].divide_by_R.R),5) # Fix this later, can be implemented more efficiently
        return Dhx/tf.cast(Rsquared,self.cmplxdtype)
    def vinit(self,x_0,negC):
        Dx = self.IFFT[0](self.dictObj[0].dmul.freezeD(x_0))
        Dx = tf.squeeze(Dx,axis=-1)
        croppedDx = self.cropAndMerge.crop(Dx)
        s_LF,QWs = negC
        #vpluss_LF,Bv = self.updatev((croppedDx + s_LF,[0.,0.,0.],QWs))
        vpluss_LF = self.updatev((croppedDx + s_LF,QWs))
        vpremerge = vpluss_LF - s_LF
        v = self.cropAndMerge.merge((vpremerge,Dx))
        v = util.addDim(v)
        return self.FFT[0](v)#(self.FFT[0](v),Bv) 
    def zinit(self,xnext,Ax,layer):     
        return self.updateZ(xnext,Ax,tf.reshape(tf.cast(0.,tf.as_dtype(self.cmplxdtype)),(1,1,1,1,1)),layer)
    def zinit_last(self,Ax):
        return self.updateZ_last(Ax,tf.reshape(tf.cast(0.,tf.as_dtype(self.cmplxdtype)),(1,1,1,1,1)))
    def gammainit(self,Ax,Bz,layer):
        return self.updateGamma(tf.cast(0.,tf.as_dtype(self.cmplxdtype)),Bz,Ax,layer)
    #def etainit(self,Bv,QWs):
    #    return self.updateEta([0.,0.,0.],Bv,QWs)


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
    def relax_layers(self,x,z,layer):
        if layer < self.noL - 1:
            #R = tf.cast(util.rotate_dims_left(self.dictObj[layer].R,5),dtype=self.cmplxdtype)
            z_over_R = self.dictObj[layer].divide_by_R(z)
        else:
            z_over_R = z
        return self.relax_layer((x,z_over_R))


    #def relax_layers(self,u,x,z,layer):
    #    gamma_scaled = u
    #    if layer < self.noL - 1:
    #        z_over_R = self.dictObj[layer].divide_by_R(z)
    #    else:
    #        z_over_R = z
    #    return self.updateGamma_layer((gamma_scaled,tf.cast(self.alpha - 1.,self.cmplxdtype)*z_over_R,tf.cast(self.alpha - 1.,self.cmplxdtype)*x))



    def relax_zero(self,Bv,negC):
        raise NotImplementedError
        return self.relax0((negC,Bv))
    #def updateV(self,x_0,eta_over_rho,negC,frozen=True):
    def updateV(self,x_0,negC,frozen=True):
        if frozen:
            Dx = self.dictObj[0].dmul.freezeD(x_0)
        else:
            Dx = self.dictObj[0].dmul(x_0)
        # It might be helpful to build an object to couple cropping and padding so that it is never messed up. Best if this is done outside, because the cropping function can be used to get QWs.
        
        return self.FFT[0](self.updateV_subfun(self.IFFT[0](Dx),negC))#,Bv # Need to use tf.pad and tf.where to extend v. Formula for padding: sum_l (prod_i from 1 to l stride_i)(kernel_size_l - 1) + whatever necessary to make divisible by the strides.

    def updateV_subfun(self,Dx,negC):
        Dx = tf.squeeze(Dx,axis=-1) # add tf.crop_to_bounding_box here.
        s_LF,QWs = negC
        #vpluss_LF,Bv = self.updatev((self.cropAndMerge.crop(Dx) + s_LF,eta_over_rho,QWs))
        vpluss_LF = self.updatev((self.cropAndMerge.crop(Dx) + s_LF,QWs))
        v = self.cropAndMerge.merge((vpluss_LF - s_LF,Dx))
        v = util.addDim(v)
        return v


    def updateZ(self,x_nextlayer,Ax_relaxed,gamma_scaled,layer,frozen=True):
        assert(layer < self.noL - 1)
        if self.strides[layer] == 2:
            return self.updateZ_downsample(x_nextlayer,Ax_relaxed,gamma_scaled,layer,frozen)
        if frozen:
            Dx = self.dictObj[layer + 1].dmul.freezeD(x_nextlayer)
        else:
            Dx = self.dictObj[layer + 1].dmul(x_nextlayer)
        #z = self.updateZ_layer[layer]((self.IFFT[layer](Dx),self.IFFT[layer](Ax_relaxed),self.IFFT[layer](gamma_scaled)))
        z = self.updateZ_layer[layer]((Dx,Ax_relaxed + gamma_scaled))
        return self.FFT[layer](z)
    def updateZ_downsample(self,x_nextlayer,Ax_relaxed,gamma_scaled,layer,frozen=True):
        assert(layer < self.noL - 1)

        # downsampled
        if frozen:
            Dx = self.dictObj[layer + 1].dmul.freezeD(x_nextlayer)
        else:
            Dx = self.dictObj[layer + 1].dmul(x_nextlayer)
        #Ax_relaxed_spatial = self.IFFT[layer](Ax_relaxed)
        #gamma_spatial = self.IFFT[layer](gamma_scaled)
        #Dx_spatial = self.IFFT[layer + 1](Dx)
        Ax_plus_gamma_ds = util.freq_downsample(Ax_relaxed + gamma_scaled)

        z_downsampled = self.updateZ_layer[layer][0]((Dx,Ax_plus_gamma_ds))
        #z_downsampled = self.updateZ_layer[layer][0]((Dx_spatial,Ax_ds,gamma_ds))
        #z_downsampled = self.updateZ_layer[layer]['downsampled']((Dx_spatial,Ax_ds,gamma_ds))

        #shift downsampled
        #z_shift_downsampled = self.updateZ_layer[layer]['shifted']((util.shift_downsample(Ax_relaxed_spatial),util.shift_downsample(gamma_spatial)))
        z_shift_downsampled = self.updateZ_layer[layer][1]((util.freq_shift_downsample(Ax_relaxed + gamma_scaled)))

        #z_missed_cols = self.updateZ_layer[layer]['missed_cols']((util.col_downsample(Ax_relaxed_spatial),util.col_downsample(gamma_spatial)))
        Ax_plus_gamma = Ax_relaxed + gamma_scaled
        Ax_plus_gamm_col_shift_ds = util.freq_col_shift_downsample(Ax_relaxed + gamma_scaled)
        z_missed_cols = self.updateZ_layer[layer][2]((Ax_plus_gamm_col_shift_ds))

        # Need to build reconstruction from the three Zs
        #z_downsample_cols = self.updateZ_layer[layer]['shift_concat']((z_downsampled,z_shift_downsampled))
        z_downsample_cols = self.updateZ_layer[layer][3]((z_downsampled,z_shift_downsampled))
        #z = self.updateZ_layer[layer]['cols_concat']((z_downsample_cols,z_missed_cols))
        z = self.updateZ_layer[layer][4]((z_downsample_cols,z_missed_cols))
        return self.FFT[layer](z)

    def updateZ_last(self,Ax_relaxed,gamma_scaled):
        z = self.updateZ_lastlayer((Ax_relaxed,gamma_scaled))
        return self.FFT[self.noL - 1](z)
    def updateGamma(self,gamma_scaled,z,Ax_relaxed,layer):
        if layer < self.noL - 1:
            #z_over_R = z/util.complexNum(util.rotate_dims_left(self.dictObj[layer].R,5))
            z_over_R = self.dictObj[layer].divide_by_R(z)
        else:
            z_over_R = z
        return self.updateGamma_layer((gamma_scaled,z_over_R,Ax_relaxed))
    #def updateEta(self,eta_over_rho,Bv,QWs):
        #return self.updateeta((eta_over_rho,Bv,QWs))


    # Low-level augmented Langrangian evaluation:
    def reconstructionTerm(self,z,Dx,layer):
        zminusDx = z - Dx
        return self.FFT[layer].parseval_sum(zminusDx)
    def reconstructionTerm_sp(self,z,Dx):
        return tf.math.reduce_sum(tf.math.square(z - Dx))
    def nonnegativeCheck(self,z):
        assert(tf.all(z >= 0.))

    def penaltyTerm(self,z,layer):
        if layer < self.noL - 1:
            if self.strides[layer] == 2:
                b = self.updateZ_layer[layer][4]((self.updateZ_layer[layer][3]((self.updateZ_layer[layer][0].b,self.updateZ_layer[layer][1].b)),self.updateZ_layer[layer][2].b))
                return tf.math.reduce_sum(tf.math.abs(b*z))
            else:
                return tf.math.reduce_sum(tf.math.abs(self.updateZ_layer[layer].get_lambda(self.rho)*z))
        else:
            return tf.math.reduce_sum(tf.math.abs(self.updateZ_lastlayer.get_lambda(self.rho)*tf.math.maximum(z,0.)))
    def jpegConstraint(self,eta_over_rho,Wv,QWs):
        #aug_cnstrnt_term = [QWz[channel] - QWs[channel] + eta_over_rho[channel] for channel in range(len(QWs))]
        #return self.rho/2*sum([tf.math.reduce_sum(aug_cnstrnt_term[channel]*tf.math.conj(aug_cnstrnt_term[channel])) for channel in range(len(QWs))])
        #Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        #QWz = jrf.threeChannelQuantize(Wz,self.qY,self.qUV,Yoffset)
        QWv = Wv
        aug_cnstrnt_term = self.Wt([(QWv[channel] - QWs[channel] + eta_over_rho[channel])/ds_factor for (channel,ds_factor) in zip(range(len(QWs)),(1.,4.,4.))])
        return self.rho/2*tf.math.reduce_sum(aug_cnstrnt_term*tf.math.conj(aug_cnstrnt_term))
    def jpegConstraint_relaxed(self,eta_over_rho,Bv,negC):
        s_LF,QWs = negC
        #aug_cnstrnt_term = [Azero[channel] + Bv[channel] + eta_over_rho[channel] for channel in range(len(Bv))]
        #return self.rho/2*sum([tf.math.reduce_sum(aug_cnstrnt_term[channel]*tf.math.conj(aug_cnstrnt_term[channel])) for channel in range(len(Bv))])
        QWv = Bv
        #Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        #QWv = jrf.threeChannelQuantize(Bv,self.qY,self.qUV,Yoffset)
        aug_cnstrnt_term = self.Wt([(-QWs[channel] + QWv[channel] + eta_over_rho[channel])/ds_factor for (channel,ds_factor) in zip(range(len(Bv)),(1.,4.,4.))])
        return self.rho/2*tf.math.reduce_sum(aug_cnstrnt_term*tf.math.conj(aug_cnstrnt_term))
    def zxConstraint(self,gamma_over_rho,x_over_R,z_over_R): # There is an error here. The nature of the representation needs to be taken into account when applying Parseval's theorem; the DC component is fine, but everything else is halfed...
        rho = util.complexNum(self.rho)
        sum_of_terms = z_over_R - x_over_R + gamma_over_rho
        output = rho*tf.reduce_sum(tf.math.conj(sum_of_terms)*sum_of_terms)
        return tf.cast(output,output.dtype.real_dtype)
    def zxConstraint_relaxed(self,gamma_over_rho,Ax_relaxed,Bz):  # See zxConstraint error
        rho = util.complexNum(self.rho)
        sum_of_terms = Ax_relaxed + Bz + gamma_over_rho
        output = rho*tf.reduce_sum(tf.math.conj(sum_of_terms)*sum_of_terms)
        return tf.cast(output,output.dtype.real_dtype)

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
        return self.relax_layer.alpha
class Wrap_ML_ADMM(tf.keras.layers.Layer):
    def __init__(self,rho,alpha_init,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,D,n_components,noi,noL,cmplxdtype,longitstat=False,*args,**kwargs):
        self.admm = MultiLayerCSC(rho,alpha_init,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,D,n_components,noi,noL,cmplxdtype,longitstat,*args,**kwargs)
        super().__init__(dtype = self.admm.cmplxdtype.real_dtype,*args,**kwargs)
    def call(self,inputs):
        return self.admm.solve_coef(inputs)

class Wrap_ML_ADMM_SC(tf.keras.layers.Layer):
    def __init__(self,rho,alpha_init,mu_init,b_init,qY,cropAndMerge,fftSz,strides,D,n_components,noi,noL,cmplxdtype,longitstat=False,*args,**kwargs):
        self.admm = MultiLayerCSC_SC(rho,alpha_init,mu_init,b_init,qY,cropAndMerge,fftSz,strides,D,n_components,noi,noL,cmplxdtype,longitstat,*args,**kwargs)
        super().__init__(dtype = self.admm.cmplxdtype.real_dtype,*args,**kwargs)
    def call(self,inputs):
        return self.admm.solve_coef(inputs)

class Get_Obj(tf.keras.layers.Layer):
    def __init__(self,ml_csc,*args,**kwargs):
        self.ml_csc = ml_csc
        super().__init__(dtype = self.ml_csc.admm.cmplxdtype,*args,**kwargs)
    def call(self,inputs):
        x,negC = inputs
        return self.ml_csc.admm.get_obj(x,negC)

class MultiLayerCSC_SC(MultiLayerCSC):
    def __init__(self,rho,alpha_init,mu_init,b_init,qY,cropAndMerge,fftSz,strides,D,n_components,noi,noL,cmplxdtype,longitstat=False,*args,**kwargs):
        rho,mu_init = self.init_param(rho,alpha_init,mu_init,cropAndMerge,noi,noL,cmplxdtype,longitstat,*args,**kwargs)
        self.initializeLayers(rho,mu_init,alpha_init,util.makelist(b_init,noL),noL,fftSz,strides,D,n_components,cmplxdtype)
        qY = tf.reshape(qY,(1,1,1,64))
        self.initializeJPEGLayers(cmplxdtype,qY)
    def get_config(self):
        config_dict = {'complex_dtype': self.cmplxdtype,
                       'num_of_Layers': self.noL,
                       'strides': self.strides,
                       'qY': self.qY,
                       'rho': self.rho,
                       'noi': self.noi,
                       'record_iteration_stats': self.longitstat}
        return config_dict
    def initializeJPEGLayers(self,cmplxdtype,qY):
        #self.W = jrf.YUV2JPEG_Coef(dtype=cmplxdtype.real_dtype)
        #self.Wt = jrf.JPEG_Coef2YUV(dtype=cmplxdtype.real_dtype)
        self.W = jrf.Y2JPEG_Coef(dtype=cmplxdtype.real_dtype)
        self.Wt = jrf.JPEG_Coef2Y(dtype=cmplxdtype.real_dtype)
        self.updatev = jrf.ZUpdate_JPEGY_Implicit(qY,self.W,self.Wt,dtype = cmplxdtype.real_dtype)
        self.qY = qY
    def get_negative_C(self,s):
        s_HF,s_LF,compressed = s
        Ws = self.W(compressed)
        Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        QWs = jrf.quantize(Ws,self.qY,Yoffset)
        return (self.cropAndMerge.crop(s_LF),QWs)
    def init_x(self,s,negC):
        s_HF,temp1,temp2 = s
        s_LF,QWs = negC
        x = []
        x.append(self.xinit(self.FFT[0](util.addDim(s_HF)),layer = 0))
        for ii in range(1,self.noL):
            if self.strides[ii - 1] == 2:
                x.append(self.xinit(util.freq_downsample(x[ii - 1]),layer=ii))
            else:
                x.append(self.xinit(x[ii - 1],layer = ii))
        Ax = (-QWs,x)
        return x,Ax

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
        return self.dictObj.freezeD(self.dictObj.dhmul.freezeD(z_prevlayer) + self.rho*(z_over_R + gamma_scaled))
    def thawD(self,inputs):
        z_prevlayer,z_over_R,gamma_scaled = inputs
        return self.dictObj(self.dictObj.dhmul(z_prevlayer) + self.rho*(z_over_R + gamma_scaled))
    def get_config(self):
        return {'rho': self.rho}

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
        self.alpha = tf.Variable(initial_value=alpha,trainable=False,name = 'alpha',dtype=tf.as_dtype(self.dtype).real_dtype)#alpha #tf.Variable(alpha_init,trainable=True)
    def call(self,inputs):
        x_over_R,z_over_R = inputs
        return  -(1 - tf.cast(self.alpha,self.dtype))*z_over_R - tf.cast(self.alpha,self.dtype)*x_over_R



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
        self.relu = tf.keras.layers.ReLU(dtype=self.dtype)
    def call(self,inputs):
        # inputs are in spatial domain
        Dx_nextlayer,Ax_relaxed,gamma_scaled = inputs
        currR = util.rotate_dims_left(self.dictObj.divide_by_R.R,5)
        leadingFactor = 1/(self.mu_nextlayer + self.rho*self.mu/currR**2)
        return leadingFactor*self.relu(self.mu_nextlayer*Dx_nextlayer - (self.rho*self.mu/currR)*(Ax_relaxed + gamma_scaled) - self.b)
    def get_config(self):
        return {'rho': self.rho}

class GetNextIterZFreq(tf.keras.layers.Layer,ppg.PostProcess):
    '''
     inputs: All must be in frequency domain.

       Dx_nextlayer: \mD_{\ell + 1}\vx_{\ell + 1}^{(k + 1)}
       Ax_relaxed: -\alpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} - (1 - \alpha_k)\mR_{\ell}^{-1}\vz_{\ell}^{(k)}
       gamma_scaled: \frac{\vgamma_{\ell}^{(k)}}{\rho\sqrt{\mu_{\ell}}

     outputs: unlike inputs, output is in spatial domain
       z: \vz_{\ell}^{(k + 1)}
    '''
    def __init__(self,rho,ifft,mu_init,mu_nextlayer,dictObj,dictObj_nextlayer,b_init,*args,**kwargs):
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
        # Inputs are in frequency domain, but output is in spatial domain.
        Dx_nextlayer,Ax_relaxed_plus_gamma_scaled = inputs
        currR = tf.reshape(self.dictObj.divide_by_R.R,shape=(1,1,1,self.dictObj.divide_by_R.R.shape[4],1))
        leadingFactor = 1/(self.mu_nextlayer + self.rho*self.mu/currR**2)
        return leadingFactor*self.relu((self.ifft(tf.cast(self.mu_nextlayer,dtype=self.dtype)*Dx_nextlayer - self.dictObj.divide_by_R(tf.cast(self.rho*self.mu,dtype=self.dtype)*Ax_relaxed_plus_gamma_scaled)),self.b))

    def get_lambda(self,rho):
        #tf.print('lambda: ',self.b)
        return self.b

    def get_config(self):
        return {'rho': self.rho}


class GetNextIterZ_lastlayer(tf.keras.layers.Layer,ppg.PostProcess):
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
        #self.relu = tf.keras.layers.ReLU(dtype=self.dtype)
        self.relu = util.Shrinkage(dtype=self.dtype)

    def call(self,inputs):
        Ax_relaxed,gamma_scaled = inputs
        R = util.rotate_dims_left(self.dictObj.divide_by_R.R)
        return self.relu(-Ax_relaxed - gamma_scaled - R*self.b)

class GetNextIterZFreq_lastlayer(tf.keras.layers.Layer,ppg.PostProcess):
    '''
      inputs: All must be in frequency domain.

        Ax_relaxed: = -\alpha_k\mR_L^{-1}\vx_L^{(k + 1)} - (1 - \alpha_k)\mR_L^{-1}\vz_L^{(k)}
        gamma_scaled: \frac{\gamma_L^{(k)}}{\rho\sqrt{\mu_L}}

      outputs: Unlike inputs, output is in spatial domain
        z_over_R: \mR_L^{-1}\vz_L^{(k + 1)}
    '''
    def __init__(self,rho,ifft,mu_init,dictObj,b_init,*args,**kwargs):
        #super().__init__(*args,**kwargs)
        tf.keras.layers.Layer.__init__(self,*args,**kwargs)
        
        self.dictObj = dictObj
        with tf.name_scope(self.name):
            self.mu = tf.Variable(mu_init,trainable=True,dtype=tf.as_dtype(self.dtype).real_dtype,name='mu')
            self.b = tf.Variable(b_init/(rho*mu_init),trainable=True,dtype=tf.as_dtype(self.dtype).real_dtype,name='b') # Is this an active design decision to avoid dependence on mu?
            self.bprev = tf.Variable(b_init/(rho*mu_init),trainable=False,dtype=tf.as_dtype(self.dtype).real_dtype,name='bprev')
            self.deltab = tf.Variable(0, trainable = False,dtype = tf.as_dtype(self.dtype).real_dtype,name = 'deltab')
            self.count = tf.Variable(0,trainable = False,dtype = tf.as_dtype(self.dtype).real_dtype,name = 'count')
        #self.relu = util.BiasedReLU(dtype=tf.as_dtype(self.dtype).real_dtype)
        #self.relu = tf.keras.layers.ReLU(dtype=tf.as_dtype(self.dtype).real_dtype)
        self.relu = util.Shrinkage(dtype=tf.as_dtype(self.dtype).real_dtype)
        self.ifft = ifft
        self.maxcount = 32
        ppg.PostProcess.add_update(self.b.name,self._update_b)
        ppg.PostProcess.add_update(self.mu.name,self._update_mu)

    def _update_b(self):
        deltab = (self.count*self.deltab + self.b - self.bprev)/(self.b + 1)
        update_cond = self.count == self.maxcount
        b = tf.cond(update_cond,lambda: self.bprev + deltab,lambda: self.bprev)
        b = tf.where(b < 0.,tf.cast(0,dtype=tf.as_dtype(self.dtype).real_dtype),b)
        count = tf.cond(update_cond,lambda: 0,lambda: self.count + 1)
        return [self.b.assign(b),self.count.assign(count),self.bprev.assign(b),self.deltab.assign(tf.cond(update_cond,tf.cast(0,dtype=tf.as_dtype(self.dtype).real_dtype),deltab))]

    def _update_mu(self):
        return [self.mu.assign(tf.where(self.mu < 1e-3,tf.cast(1e-3,dtype=tf.as_dtype(self.dtype).real_dtype),self.mu))]

    def call(self,inputs):
        Ax_relaxed,gamma_scaled = inputs
        R = tf.reshape(self.dictObj.divide_by_R.R,shape=(1,1,1,self.dictObj.divide_by_R.R.shape[4],1))
        return self.relu((-self.ifft(Ax_relaxed + gamma_scaled),R*self.b))

    def get_lambda(self,rho):
        #tf.print('last lambda: ',self.b*(rho*self.mu))
        return self.b*(rho*self.mu)

class GetNextIterZ_downsampleTossed(tf.keras.layers.Layer):
    def __init__(self,rho,mu,dictObj,b,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mu = mu
        self.dictObj = dictObj
        self.b = b
        self.rho = rho
        #self.relu = tf.keras.layers.ReLU(dtype=self.dtype)
        self.relu = util.Shrinkage(dtype=tf.as_dtype(self.dtype).real_dtype)
    def call(self,inputs):
        Ax_relaxed,gamma_scaled = inputs
        R = util.rotate_dims_left(self.dictObj.divide_by_R.R)
        return R*self.relu((-Ax_relaxed - gamma_scaled,R*self.b/(self.rho*self.mu)))
    def get_config(self):
        return {'rho': self.rho}

class GetNextIterZFreq_downsampleTossed(tf.keras.layers.Layer):
    def __init__(self,rho,ifft,mu,dictObj,b,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mu = mu
        self.dictObj = dictObj
        self.b = b
        self.rho = rho
        #self.relu = tf.keras.layers.ReLU(dtype=tf.as_dtype(self.dtype).real_dtype)
        self.relu = util.Shrinkage(dtype=tf.as_dtype(self.dtype).real_dtype)
        self.ifft = ifft
    def call(self,inputs):
        Ax_relaxed_plus_gamma_scaled = inputs
        R = tf.reshape(self.dictObj.divide_by_R.R,shape=(1,1,1,self.dictObj.divide_by_R.R.shape[4],1))
        #return R*self.relu(self.ifft(Ax_relaxed_plus_gamma_scaled) - R*self.b/(self.rho*self.mu))
        return R*self.relu((self.ifft(Ax_relaxed_plus_gamma_scaled),R*self.b/(self.rho*self.mu)))
    def get_config(self):
        return {'rho': self.rho}

class GetNextIterGamma(tf.keras.layers.Layer):
    '''
      inputs: All must be in frequency domain.

        gamma_scaled: \frac{\gamma_{\ell}^{(k)}}{\rho\sqrt{\mu_{\ell}}
        z_over_R: \mR_{\ell}^{-1}\vz_{\ell}^{(k + 1)}
        Ax_relaxed: -\valpha_k\mR_{\ell}^{-1}\vx_{\ell}^{(k + 1)} - (1 - \valpha_k)\mR_{\ell}^{-1}\vz_{\ell}^{(k)}

      outputs: Also in frequency domain
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
