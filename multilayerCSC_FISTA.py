import optmz
import tensorflow as tf
import post_process_grad as ppg
import jpeg_related_functions as jrf
import transforms as transf
import numpy as np
import util

class ML_FISTA(optmz.FISTA):
    def __init__(self,lpstz,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,D,noi,noL,cmplxdtype,longitstat=False,*args,**kwargs):

        self.noL = noL
        self.lpstz = lpstz
        self.strides = strides
        self.relu = tf.keras.layers.ReLU(dtype = cmplxdtype.real_dtype)
        self.noi = noi
        self.stepSz = 1/lpstz
        self.cmplxdtype = cmplxdtype
        dtype = cmplxdtype.real_dtype
        qY = tf.reshape(qY,(1,1,1,64))
        qUV = tf.reshape(qUV,(1,1,1,64))
        self.qY = qY
        self.qUV = qUV
        self.cropAndMerge = cropAndMerge
        self.FFT,self.IFFT,self.FFT_factor = self.build_fft_layers(fftSz,noL)
        mu_init = util.makelist(mu_init,noL)
        b_init = util.makelist(b_init,noL)
        lmbd = []
        for ii in range(noL):
            lmbd.append(mu_init[ii]*(b_init[ii] + tf.zeros(self.get_b_shape(fftSz[ii],D[ii].shape[-1]),dtype= cmplxdtype.real_dtype)))
        self.dmul = []
        self.dhmul = []
        for ll in range(noL):
            Df = self.FFT[ll](D[ll]/tf.math.sqrt(tf.reduce_sum(input_tensor=D[ll]**2,axis=(1,2,3),keepdims=True)))
            fltrSz = D[ll].shape[1:3]
            noc = D[ll].shape[-2]
            self.dhmul.append(DhMul(Df,fltrSz,self.IFFT[ll],self.FFT[ll],noc,dtype=self.cmplxdtype,name='Dh_mul_' + str(ll)))
            self.dmul.append(DMul(self.dhmul[ll],dtype=self.cmplxdtype,name='D_mul_' + str(ll))) 
        self.gradfLastLayer,mu = self.build_grad_last_layer(mu_init[noL - 1],self.dmul[noL - 1],self.dhmul[noL - 1])
        reversedGradLayers = []
        for ll in range(noL - 2,-1,-1):
            grad_layer,mu = self.build_grad_layer(mu_init[ll],mu,self.dmul,self.dhmul,ll)
            reversedGradLayers.append(grad_layer)
        self.gradfY = Gradf_y(mu,dtype=self.cmplxdtype)
        self.gradfLayers = []
        for ll in range(noL - 1):
            self.gradfLayers.append(reversedGradLayers[noL - ll - 2])
        

        self.shrinkLayer = []
        for ll in range(noL - 1):
            self.shrinkLayer.append(Shrink(tf.cast(lmbd[ll]/self.gradfLayers[ll].mu*self.stepSz,cmplxdtype.real_dtype),self.IFFT[ll],self.FFT[ll],dtype = self.cmplxdtype,name = 'shrink_' + str(ll)))
        self.shrinkLayer.append(BiasedReLU(tf.cast(lmbd[noL - 1]/self.gradfLastLayer.mu*self.stepSz,cmplxdtype.real_dtype),self.IFFT[noL - 1],self.FFT[noL - 1],dtype = self.cmplxdtype,name = 'shrink_' + str(noL - 1)))
        self.W = jrf.RGB2JPEG_Coef(dtype=cmplxdtype.real_dtype)
        self.Wt = jrf.JPEG_Coef2RGB(dtype=cmplxdtype.real_dtype)
        self.proxy = jrf.ZUpdate_JPEG_Implicit(qY,qUV,self.W,self.Wt,dtype = self.cmplxdtype.real_dtype)
        super().__init__(lpstz,noi,dtype=self.cmplxdtype.real_dtype,*args,**kwargs)
    def get_config(self):
        config_dict = {'complex_dtype': self.cmplxdtype,
                       'num_of_Layers': self.noL,
                       'strides': self.strides,
                       'qY': self.qY,
                       'qUV': self.qUV,
                       'lpstz': self.lpstz,
                       'noi': self.noi,
                       'record_iteration_stats': self.longitstat}
        return config_dict

    def get_b_shape(self,fftSz,M):
        #return [1,fftSz[0],fftSz[1],M,1,]
        #return [1,1,1,M,1]
        return (1,)

    def build_fft_layers(self,fftSz,noL):
        FFT = []
        IFFT = []
        FFT_factor = []
        for ii in range(noL):
            FFT.append(transf.fft2d_inner(fftSz[ii]))
            IFFT.append(transf.ifft2d_inner(fftSz[ii]))
            FFT_factor.append(np.prod(fftSz[ii]))
        return FFT,IFFT,FFT_factor

    def build_grad_layer(self,mu_init,munext,dmul,dhmul,layer):
        grad_layer = Gradf(mu_init,munext,dmul[layer],dhmul[layer],dmul[layer + 1],dtype=self.cmplxdtype,name='grad_layer_' + str(layer)) # add names
        return grad_layer,grad_layer.mu

    def build_grad_last_layer(self,mu_init,dmul,dhmul): 
        grad_layer = Gradf_Last(mu_init,dmul,dhmul,dtype=self.cmplxdtype,name='last_grad_layer') # add names
        return grad_layer,grad_layer.mu


    def get_negative_C(self,s): # creates a cropped version of the low-frequency image computes the corresponding JPEG coef
        s_HF,s_LF,compressed = s
        Ws = self.W(compressed)
        Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        QWs = jrf.threeChannelQuantize(Ws,self.qY,self.qUV,Yoffset)
        return (self.cropAndMerge.crop(s_LF),QWs)

    def init_r(self):
        r = 1.
        r = tf.cast(r,self.cmplxdtype)
        return r

    def init_z(self,s,negC):
        s_HF,temp1,temp2 = s
        s_LF,QWs = negC
        z = []
        z_sig = self.FFT[0](util.addDim(s_HF))
        z.append(self.zinit(z_sig,layer = 0))
        for ii in range(1,self.noL):
            if self.strides[ii - 1] == 2:
                z.append(self.zinit(util.freq_downsample(x[ii - 1]),layer=ii))
            else:
                z.append(self.zinit(z[ii - 1],layer = ii))
        return z_sig,z
    def init_x(self,s,z,negC):
        z_sig,z_coef = z
        x = []
        for ii in range(self.noL):
            x.append(self.xinit(z=z_coef[ii],layer=ii))
        return self.prox_sigz(z_sig,negC),x
    def init_itrstats(self,r,s,x,z):
        return []

    def zinit(self,xprev,layer):
        #return self.shrinkLayer[layer](self.dhmul[layer].freezeD(xprev)) # Doesn't work since shink layer is intended to be applied to freq domain.
        return self.dhmul[layer].freezeD(xprev)

    def xinit(self,z,layer):
        return z

    # iterative steps:
    #def gradstep(self,x):
    def gradstep(self,x):
        return self.gradstep_trunc(x,frzD = True,layer=self.noL - 1)
    def gradstep_trunc(self,x,frzD,layer):
        x_sig,x_coef = x

        z_coef = []
        if layer < 0:
            if frzD:
                z_sig = self.updateW(x_sig,self.dmul[0].freezeD(x_coef[0]))
            else:
                z_sig = self.updateW(x_sig,self.dmul[0](x_coef[0]))
            return z_sig,z_coef

        z_sig = self.updateW(x_sig,self.dmul[0](x_coef[0]))

        
        if self.noL == 1:
            z_coef.append(self.updateZ_lastlayer(x_prevlayer=x_sig,x = x_coef))
        else:
            z_coef.append(self.updateZ(x_prevlayer=x_sig,x=x_coef[0],x_nextlayer = x_coef[1],frzD = True,layer=0))
        for ii in range(1,layer - 1):
            z_coef.append(self.updateZ(x_prevlayer=x_coef[ii - 1],x = x_coef[ii],x_nextlayer = x_coef[ii + 1],frzD = True,layer=ii))

        if self.noL > 1:
            if layer == self.noL - 1:
                z_coef.append(self.updateZ_lastlayer(x_prevlayer = x_coef[layer - 1],x = x_coef[layer]))
            else:
                z_coef.append(self.updateZ(x_prevlayer=x_coef[layer - 2],x = x_coef[layer - 1],x_nextlayer = x_coef[layer],frzD = frzD,layer =layer - 1))

        return z_sig,z_coef

    def updateZ(self,x_prevlayer,x,x_nextlayer,frzD,layer):
        if frzD:
            return x - 1/self.lpstz*self.gradfLayers[layer].freezeD((x_prevlayer,x,x_nextlayer))
        else:
            return x - 1/self.lpstz*self.gradfLayers[layer]((x_prevlayer,x,x_nextlayer))

    def updateZ_lastlayer(self,x_prevlayer,x):
        return x - 1/self.lpstz*self.gradfLastLayer((x_prevlayer,x))

    def updateW(self,y,x0):
        return y - 1/self.lpstz*self.gradfY((y,x0))
        
    def proxstep(self,z,negC):
        return self.proxstep_trunc(z,negC,layer=self.noL)

    def proxstep_trunc(self,z,negC,layer):
        z_sig,z_coef = z
        zprox = []
        for ii in range(0,layer):
            zprox.append(self.proxZ(z = z_coef[ii],layer=ii))
        return (self.prox_sigz(z_sig,negC),zprox)

    def prox_sigz(self,z_sig,negC):
        return self.FFT[0](self.prox_sigz_sp(self.IFFT[0](z_sig),negC))

    def prox_sigz_sp(self,z_sig,negC):
        z_sig_s= tf.squeeze(z_sig,axis = -1)
        cropped_z_sig = self.cropAndMerge.crop(z_sig_s)
        sig_prox = self.proxZ_sig(cropped_z_sig + negC[0],negC[1])
        z_sig_prox = self.cropAndMerge.merge((sig_prox - negC[0],z_sig_s))
        return util.addDim(z_sig_prox)

    def proxZ(self,z,layer):
        return self.shrinkLayer[layer](z)

    def proxZ_sig(self,z_sig,negC):
        return self.proxy((z_sig,negC))

    def momstep(self,r,x,z):
        return self.momstep_trunc(r,x=x,z=z,layer=self.noL)

    def momstep_trunc(self,r,x,z,layer):
        x_sig,x_coef = x
        z_sig,z_coef = z
        xnew_coef = []
        rprev = r
        r = self.update_r(r)
        for ii in range(layer):
            xnew_coef.append(self.updateX(r,rprev,x_coef[ii],z_coef[ii]))
        xnew_sig = self.updateX(r,rprev,x_sig,z_sig)
        return r,(xnew_sig,xnew_coef)

    def updateX(self,r,rprev,x,z):
        #print(z.dtype)
        #print(rprev.dtype)
        #print(r.dtype)
        #print(x.dtype)
        return z + ((rprev - 1)/r)*(z - x)

    def update_r(self,r):
        return 1./2.*(1. + tf.math.sqrt(1. + 4.*tf.math.square(r)))

    #class ITSTAT:
    #     def __init__(self):
    #         x = []
    #         z = []
    #     def insertx(x_coef):
    #         self.x.insert(0,x_coef[-1])
    #     def insertz(z_coef):
    #         self.z.insert(0,z_coef[-1])
        
    def itstats_record(self,r,x,z,itstats):
        return itstats

    def init_itrstats(self,r,s,x,z):
        itstats = {}
        itstats['x'] = []
        itstats['z'] = []
        return itstats

    def itstats_record_final(self,r,x,z,itstats):
        x_sig,x_coef = x
        z_sig,z_coef = z
        itstats['x'].insert(0,x_coef[-1])
        itstats['z'].insert(0,z_coef[-1])
        return itstats

    # Before and After:
    def preprocess(self,s):
        return s
    def get_output(self,r,s,x,z,negC,itstats):
        s_LF,QWs = negC
        for layer in range(self.noL - 1,0,-1):
            z = self.gradstep_trunc(x,frzD=False,layer=layer)
            z = self.proxstep_trunc(z,negC,layer)
            r,x = self.momstep_trunc(r,x,z,layer)
            itstats = self.itstats_record_final(r,x,z,itstats)

        Dx = self.IFFT[0](self.dmul[0](z[1][0]))
        z = self.gradstep_trunc(x,frzD=False,layer=-1)
        return (self.cropAndMerge.crop(tf.squeeze(Dx,axis=-1)) + s_LF,self.cropAndMerge.crop(tf.squeeze(self.IFFT[0](z[0]),axis=-1)) + s_LF,itstats)

    def IFFT_all(self,z_coef):
        #z_sig,z_coef = z
        z_sp = []
        for ll in range(self.noL):
            z_sp.append(self.IFFT[ll](z_coef[ll]))
        return z_sp

    #def get_obj(self,z):
    #    z_sp = self.IFFT_all(z)
    #    return self.data_fid(z_sp) + self.l1_penalty(z_sp)

    def get_zsp_Dz0_v(self,z,negC):
        z_sig,z_coef = z
        Dz0 = self.IFFT[0](self.dmul[0](z_coef[0]))
        z_sp = self.IFFT_all(z_coef)
        v = self.prox_sigz_sp(Dz0,negC)
        return z_sig,v,Dz0,z_coef,z_sp

    def get_obj(self,z,negC):
        z_sig,v,Dz0,z_coef,z_sp = self.get_zsp_Dz0_v(z,negC)
        return self.sig_recon_err(v,Dz0) + self.data_fid(z_coef,z_sp) + self.l1_penalty(z_sp)

    def sig_recon_err(self,v,Dz0):
        return self.gradfLayers[0].mu/2*tf.math.reduce_sum(tf.math.square(v - Dz0))

    def get_obj_and_cnstr(self,z,negC):
        z_sig,v,Dz0,z_coef,z_sp
        return self.sig_recon_err(v,Dz0) + self.data_fid(z_coef,z_sp) + l1_penalty(z_sp),self.quantization_constraint(z_sig,negC)

    def data_fid(self,z_coef,z_sp): 
        data_fid_sum = tf.cast(0,self.cmplxdtype.real_dtype)
        for ll in range(1,self.noL - 1):
            data_fid_sum += self.data_fid_layer(self.gradfLayers[ll - 1].mu,z_sp[ll - 1],z_coef[ll],ll)
        data_fid_sum += self.data_fid_layer(self.gradfLastLayer.mu,z_sp[self.noL - 2],z_coef[self.noL - 1],self.noL - 1)
        return data_fid_sum

    def l1_penalty(self,z_coef):
        l1_sum = tf.cast(0,self.gradfLayers[0].mu.dtype)
        for ll in range(0,self.noL - 1):
            #tf.print('lambda[' + str(ll) + ']: ',self.shrinkLayer[ll].thrsh*tf.cast(self.gradfLayers[ll].mu,self.gradfLayers[ll].mu.dtype.real_dtype)/self.stepSz)
            l1_sum += self.l1_penalty_layer(self.shrinkLayer[ll].thrsh*self.gradfLayers[ll].mu/self.stepSz,z_coef[ll])
        l1_sum += self.l1_penalty_nonneglayer(self.shrinkLayer[self.noL - 1].thrsh*tf.cast(self.gradfLastLayer.mu,self.gradfLastLayer.mu.dtype.real_dtype)/self.stepSz,z_coef[self.noL - 1])
        #tf.print('lambda[' + str(self.noL - 1) + ']: ',self.shrinkLayer[self.noL - 1].thrsh*tf.cast(self.gradfLastLayer.mu,self.gradfLastLayer.mu.dtype.real_dtype)/self.stepSz)
        return l1_sum       

    def data_fid_layer(self,mu,z_prevlayer_sp,z_currlayer,layer):
        return mu/2*tf.math.reduce_sum(tf.math.square(z_prevlayer_sp - self.IFFT[layer](self.dmul[layer](z_currlayer))))

    def l1_penalty_layer(self,lmbda,z):
        return tf.math.reduce_sum(tf.math.abs(lmbda*z))

    def l1_penalty_nonneglayer(self,lmbda,z):
        return tf.math.reduce_sum(self.relu(lmbda*z))


    def nonnegativity_constraint_layer(self,z):
        return tf.math.reduce_sum(z - tf.math.abs(z),keepdims = True)/(-2)

    def quantization_contraint(self,z_sig,negC):
        z_sig_s= tf.squeeze(z_sig,axis = -1)
        cropped_z_sig = self.cropAndMerge.crop(z_sig_s)
        sig_prox = self.proxZ_sig(cropped_z_sig + negC[0],negC[1])
        
        return 1/2*(tf.math.reduce_sum(tf.math.square(sig_prox - cropped_z_sig_s),keepdims = True))

    def get_dict(self):
        D = []
        for ii in range(self.noL):
            Dextend = self.IFFT[ii](tf.complex(self.dhmul[ii].Dfreal,self.dhmul[ii].Dfimag)) #fltrSz is actually in dhmul
            fltrSz = self.dhmul[ii].updtD.fltrSz
            Dtrunc = Dextend[slice(None),slice(0,fltrSz[0],1),slice(0,fltrSz[1],1),slice(None),slice(None)]
            D.append(tf.reshape(Dtrunc,Dtrunc.shape[1:]))
        return D
    def get_lamda(self):
        lmbda = []
        for ii in range(self.noL - 1):
            lmbda.append(self.shrinkLayer[ii].thrsh*self.gradfLayers[ii].mu/self.stepSz)
        lmbda.append(self.shrinkLayer[self.noL - 1].thrsh*self.gradfLastLayer.mu/self.stepSz)
        return lmbda
    def get_mu(self):
        mu = []
        for ii in range(self.noL - 1):
            mu.append(self.gradfLayers[ii].mu)
        mu.append(self.gradfLastLayer.mu)
        return mu

class Wrap_ML_FISTA(tf.keras.layers.Layer):
    def __init__(self,lpstz,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,D,noi,noL,cmplxdtype,longitstat=False,*args,**kwargs):
        self.fista = ML_FISTA(lpstz,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,D,noi,noL,cmplxdtype,longitstat,*args,**kwargs)
        super().__init__(dtype = self.fista.cmplxdtype.real_dtype,*args,**kwargs)
    def call(self,inputs):
        return self.fista.solve_coef(inputs)

class Get_Obj(tf.keras.layers.Layer):
    def __init__(self,ml_csc,*args,**kwargs):
        self.ml_csc = ml_csc
        super().__init__(dtype = self.ml_csc.fista.cmplxdtype,*args,**kwargs)
    def call(self,inputs):
        x,negC = inputs
        return self.ml_csc.fista.get_obj(x,negC)

class Gradf_y(tf.keras.layers.Layer):
    def __init__(self,mu0,*args,**kwargs):
        self.mu = mu0
        super().__init__(*args,**kwargs)
    def call(self,x):
        y,x0 = x
        return tf.cast(self.mu,self.dtype)*(y - x0)

class Gradf(tf.keras.layers.Layer):
    def __init__(self,mucurr,munext,dmul,dhmul,dnextmul,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with tf.name_scope(self.name):
            self.mu = tf.Variable(initial_value=mucurr,trainable=True,dtype=tf.as_dtype(self.dtype).real_dtype,name='mu')
        self.munext =  munext
        self.dmul = dmul
        self.dhmul = dhmul
        self.dnextmul = dnextmul
        ppg.PostProcess.add_update(self.mu.name,self._update_mu)
        
    def call(self,x):
        xprev,xcurr,xnext = x
        return -tf.cast(self.mu,self.dtype)*self.dhmul.freezeD(xprev) + tf.cast(self.mu,self.dtype)*self.dhmul.freezeD(self.dmul.freezeD(xcurr)) + tf.cast(self.munext,self.dtype)*xcurr - tf.cast(self.munext,self.dtype)*self.dnextmul(xnext)

    def freezeD(self,x):
        xprev,xcurr,xnext = x
        return -tf.cast(self.mu,self.dtype)*self.dhmul.freezeD(xprev) + tf.cast(self.mu,self.dtype)*self.dhmul.freezeD(self.dmul.freezeD(xcurr)) + tf.cast(self.munext,self.dtype)*xcurr - tf.cast(self.munext,self.dtype)*self.dnextmul.freezeD(xnext)
    def _update_mu(self):
        return [self.mu.assign(tf.where(self.mu < 1e-3,tf.cast(1e-3,dtype=tf.as_dtype(self.dtype).real_dtype),self.mu))]

class Gradf_Last(tf.keras.layers.Layer):
    def __init__(self,mu_init,dmul,dhmul,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with tf.name_scope(self.name):
            self.mu = tf.Variable(initial_value=mu_init,trainable=True,dtype=tf.as_dtype(self.dtype).real_dtype,name = 'mu')
        self.dhmul = dhmul
        self.dmul = dmul
        ppg.PostProcess.add_update(self.mu.name,self._update_mu)

    def call(self,x):
        xprev,xcurr = x
        return -tf.cast(self.mu,self.dtype)*self.dhmul(xprev) + tf.cast(self.mu,self.dtype)*self.dhmul(self.dmul(xcurr))
    def freezeD(self,x):
        xprev,xcurr = x
        return -tf.cast(self.mu,self.dtype)*self.dhmul.freezeD(xprev) + tf.cast(self.mu,self.dtype)*self.dhmul.freezeD(self.dmul.freezeD(xcurr))
    def _update_mu(self):
        return [self.mu.assign(tf.where(self.mu < 1e-3,tf.cast(1e-3,dtype=tf.as_dtype(self.dtype).real_dtype),self.mu))]

class Shrink(tf.keras.layers.Layer):
    def __init__(self,thrsh,IFFT,FFT,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with tf.name_scope(self.name):
            self.thrsh = tf.Variable(thrsh,trainable=True,dtype=tf.as_dtype(self.dtype).real_dtype,name='b')
        ppg.PostProcess.add_update(self.thrsh.name,self._update_b)
        self.IFFT = IFFT
        self.FFT = FFT
        self.RELU = util.Shrinkage(dtype = tf.as_dtype(self.dtype).real_dtype)
    def call(self,x):
        return self.FFT(self.RELU((self.IFFT(x),self.thrsh)))
    def _update_b(self):
        return [self.thrsh.assign(tf.where(self.thrsh < 0.,tf.cast(0,dtype=tf.as_dtype(self.dtype).real_dtype),self.thrsh)),]

class BiasedReLU(tf.keras.layers.Layer):
    def __init__(self,thrsh,IFFT,FFT,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with tf.name_scope(self.name):
            self.thrsh = tf.Variable(thrsh,trainable=True,dtype=tf.as_dtype(self.dtype).real_dtype,name='b')
        ppg.PostProcess.add_update(self.thrsh.name,self._update_b)
        self.IFFT = IFFT
        self.FFT = FFT

        self.RELU = util.BiasedReLU(dtype = tf.as_dtype(self.dtype).real_dtype)
    def call(self,x):
        return self.FFT(self.RELU((self.IFFT(x),self.thrsh)))
    def _update_b(self):
        return [self.thrsh.assign(tf.where(self.thrsh < 0.,tf.cast(0,dtype=tf.as_dtype(self.dtype).real_dtype),self.thrsh)),]

class DhMul(tf.keras.layers.Layer,ppg.PostProcess):
    def __init__(self,Df,fltrSz,IFFT,FFT,noc,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with tf.name_scope(self.name):
            self.Dfreal = tf.Variable(initial_value=tf.math.real(Df),trainable=True,name='Dfreq_real')
            self.Dfimag = tf.Variable(initial_value=tf.math.imag(Df),trainable=True,name='Dfreq_imag')
            #self.Dfprev = tf.Variable(initial_value=Df,trainable=False,name='Dfprev')
        self.varname = self.Dfreal.name
        self.FFT = FFT
        self.updtD = IFFT_Trunc_Normalize(fltrSz,IFFT,noc)
        ppg.PostProcess.add_update(self.Dfreal.name,self._updateD)
    def get_config(self):
        return {'varname': self.varname}

    def call(self, inputs):
        Df = tf.complex(self.Dfreal,self.Dfimag)
        return tf.matmul(a=Df,b=inputs,adjoint_a=True)

    def freezeD(self, inputs):
        Df = tf.complex(self.Dfreal,self.Dfimag)
        return tf.matmul(a=tf.stop_gradient(Df),b=inputs,adjoint_a=True)

    def _updateD(self):
        Df = self.FFT(self.updtD(tf.complex(self.Dfreal,self.Dfimag)))
        self.Dfreal.assign(tf.math.real(Df))
        self.Dfimag.assign(tf.math.imag(Df))
        

class DMul(tf.keras.layers.Layer):
    def __init__(self,dhmul,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dhmul = dhmul
        
    def call(self, inputs):
        Df = tf.complex(self.dhmul.Dfreal,self.dhmul.Dfimag)
        return tf.matmul(a=Df,b=inputs)

    def freezeD(self, inputs):
        Df = tf.complex(self.dhmul.Dfreal,self.dhmul.Dfimag)
        return tf.matmul(a=tf.stop_gradient(Df),b=inputs)



class IFFT_Trunc_Normalize(tf.keras.layers.Layer):
    def __init__(self,fltrSz,IFFT,noc,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ifft = IFFT
        self.fltrSz = fltrSz
        self.noc = noc
        super().__init__(*args,**kwargs)
    def get_config(self):
        return {'fltrSz': self.fltrSz, 'noc': self.noc}
    def call(self,inputs):
        D = self.ifft(inputs)
        Dtrunc = D[slice(None),slice(0,self.fltrSz[0],1),slice(0,self.fltrSz[1],1),slice(None),slice(None)]
        return Dtrunc/tf.math.sqrt(tf.reduce_sum(input_tensor=Dtrunc**2,axis=(1,2,3),keepdims=True))
    def get_config(self):
        config_dict = {'filter_size': self.fltrSz,
                       'num_of_channels': self.noc}
        return config_dict
