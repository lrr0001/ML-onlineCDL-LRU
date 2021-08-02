import optmz
import tensorflow as tf
import post_process_grad as ppg
import jpeg_related_functions as jrf
class ML_FISTA(optmz.FISTA):
    def __init__(self,lpstz,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,D,noi,noL,cmplxdtype,longitstat=False,*args,**kwargs):
        self.lpstz = lpstz
        self.strides = strides
        self.noi = noi
        stepSz = 1/lpstz
        self.cmplxdtype = cmplxdtype
        dtype = cmplxdtype.real_dtype
        qY = tf.reshape(qY,(1,1,1,64))
        qUV = tf.reshape(qUV,(1,1,1,64))
        self.cropAndMerge = cropAndMerge
        self.FFT,self.IFFT,self.FFT_factor = self.build_fft_layers(fftSz,noL)
        self.dmul = []
        self.dhmul = []
        for ll in range(noL):
            Df = self.FFT[ll](D[ll]/tf.math.sqrt(tf.reduce_sum(input_tensor=D[ll]**2,axis=(1,2,3),keepdims=True)))
            self.dhmul.append(DhMul(Df,name='Dh_mul_' + str(ll))) # Don't forget to add names
            self.dmul.append(DMul[ll](self.dhmul,name='D_mul_' + str(ll))) # Don't forget to add names
        self.gradfLastLayer,mu = build_grad_last_layer(mu_init,dmul,dhmul)
        reversedGradLayers = []
        for ll in range(noL - 2,-1,-1):
            grad_layer,mu = build_grad_layer(mu_init,mu,self.dmul,self.dhmul,ll)
            reversedGradLayers.append(grad_layer)
        self.gradfY = Gradf_y(mu)

        self.gradfLayers = []
        for ll in range(noL - 1):
            gradfLayers.append(reversedGradLayers(noL - ll - 1))
        

        self.shrinkLayer = []
        for ll in range(noL):
            self.shrinkLayer.append(Shrink(lmbd[ll]*stepSz,self.FFT[ll],self.IFFT[ll]))
        self.proxy = jrf.ZUpdate_JPEG_Implicit(qY,qUV,jrf.RGB2JPEG_Coef(dtype=cmplxdtype.real_dtype),jrf.JPEG_Coef2RGB(dtype=cmplxdtype.real_dtype))
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

    def build_fft_layers(self,fftSz,noL):
        FFT = []
        IFFT = []
        FFT_factor = []
        for ii in range(noL):
            FFT.append(transf.fft2d_inner(fftSz[ii]))
            IFFT.append(transf.ifft2d_inner(fftSz[ii]))
            FFT_factor.append(np.prod(fftSz[ii]))
        return FFT,IFFT,FFT_factor

    def build_grad_layer(mu_init,munext,dmul,dhmul,layer):
        grad_layer = Gradf(mu_init,munext,dmul[layer],dhmul[layer],dmul[layer + 1]) # add names
        return grad_layer,grad_layer.mu

    def build_grad_last_layer(mu_init,dmul,dhmul): 
        grad_layer = Gradf_Last(mu_init,dmul,dhmul) # add names
        return grad_layer,grad_layer.mu


    def get_negative_C(self,s): # creates a cropped version of the low-frequency image computes the corresponding JPEG coef
        s_HF,s_LF,compressed = s
        Ws = self.W(compressed)
        Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,self.dtype),tf.cast(0.,self.dtype))
        QWs = jrf.threeChannelQuantize(Ws,self.qY,self.qUV,Yoffset)
        return (self.cropAndMerge.crop(s_LF),QWs)
    def init_z(self,s,negC):
        s_HF,temp1,temp2 = s
        s_LF,QWs = negC
        z = []
        z.append(self.zinit(self.FFT[0](util.addDim(s_HF)),layer = 0))
        for ii in range(1,self.noL):
            if self.strides[ii - 1] == 2:
                z.append(self.zinit(util.freq_downsample(x[ii - 1]),layer=ii))
            else:
                z.append(self.zinit(z[ii - 1],layer = ii))
        return s_HF,z
    def init_x(self,s,z,negC):
        z_sig,z_coef = z
        x = []
        for ii in range(self.noL):
            x.append(self.xinit(z=z_coef[ii],layer=ii))
        return z_sig,x
    def init_itstats(self,s,x,z):
        return []

    def zinit(self,xprev,layer):
        return self.shrinkLayer[layer](self.dhmul[layer].freezeD(xprev))

    def xinit(self,z,layer):
        return z

    # iterative steps:
    #def gradstep(self,x):
    def gradstep(self,x):
        return self.gradstep_trunc(x,frzD = True,layer=self.noL)
    def gradstep_trunc(self,x,frzD,layer = self.noL - 1):
        x_sig,x_coef = x

        z_coef = []
        if layer < 0
            if frzD:
                z_sig = self.updateW(x_sig,self.dmul.freezeD(x_coef[0]))
            else:
                z_sig = self.updateW(x_sig,self.dmul(x_coef[0]))
            return z_sig,z_coef

        z_sig = self.updateW(x_sig,self.dmul(x_coef[0]))

        
        if self.noL == 1:
            z_coef.append(self.updateZ_lastlayer(x_prevlayer=x_sig,x = x_coef))
        else:
            z_coef.append(self.updateZ(x_prevlayer=x_sig,x=x_coef[0],x_nextlayer = x_coef[1],frzD = True,layer=0))
        for ii in range(1,layer - 1):
            z_coef.append(self.updateZ(x_prevlayer=x_coef[ii - 1],x = x_coef[ii],x_nextlayer = x_coef[ii + 1],frzD = True,layer=ii))

        if self.noL > 1:
            if layer == self.noL - 1:
                z_coef.append(self.updateZ_lastlayer(x_prevlayer = x_coef[ii],x = x_coef[ii + 1])
            else:
                z_coef.append(self.updateZ(x_prevlayer=x_coef[ii],x = x_coef[ii + 1],x_nextlayer = x_coef[ii + 1],frzD = frzD,layer =ii + 1))

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

    def proxstep_trunc(self,z,negC,layer=self.noL):
        z_sig,z_coef
        zprox = []
        for ii in range(0,layer):
            zprox.append(self.proxZ(z = z_coef[ii],layer=ii))
        z_sig_s= self.IFFT[0](z_sig)
        cropped_z_sig = self.cropAndMerge.crop(tf.squeeze(z_sig_s,axis=-1))
        sig_prox = self.proxZ_sig(cropped_z_sig + negC[0],negC[1])
        z_sig_prox = self.cropAndMerge((sig_prox - negC[0],z_sig_s))
        return self.FFT[0](util.addDim(z_sig_prox)),zprox

    def proxZ(self,z,layer):
        return self.shrinkLayer(layer)(z)

    def proxZ_sig(self,z_sig,negC):
        return self.proxy((z_sig,negC))

    def momstep(self,r,x,z):
        return momstep_trunc(self,r,x=x,z=z,layer=self.noL)

    def momstep_trunc(self,r,x,z,layer = self.noL):
        x_sig,x_coef = x
        z_sig,z_coef = z
        xnew_coef = []
        rprev = r
        r = update_r(r)
        for ii in range(layer):
            xnew_coef.append(self.updateX(r,rprev,x_coef[ii],z_coef[ii]))
        xnew_sig = self.updateX(r,rprev,x_sig,z_sig)
        return r,(xnew_sig,xnew_coef)

    def updateX(self,r,rprev,x,z):
        return z + ((rprev - 1)/r)*(z - x)

    def update_r(self,r):
        return 1./2.*(1. + math.sqrt(1. + 4.*r**2))
        
    def itstats_record(self,r,x,z,itstats):
        return itstats

    def init_itstats(self,s):
        itstats.x = []
        itstats.z = []

    def itstats_record_final(self,r,x,z,itstats):
        x_sig,x_coef = x
        z_sig,z_coef = z
        itstats.x.insert(0,x_coef[end])
        itstats.z.insert(0,z_coef[end])
        return itstats

    # Before and After:
    def preprocess(self,s):
        return s
    def get_output(self,r,s,x,z,itstats):
        s_LF,QWs = negC
        for layer in range(self.noL - 1,0,-1):
            z = self.gradstep_trunc(x,frzD=False,layer=layer)
            z = self.proxstep_trunc(z,negC,layer)
            r,x = self.momstep_trunc(r,x,z,layer)
            itstats = self.itstats_record_final(r,x,z,itstats)

        z = self.gradstep_trunc(x,frzD=False,layer=-1)
        return (self.cropAndMerge.crop(tf.squeeze(Dx,axis=-1)) + s_LF,self.cropAndMerge.crop(tf.squeeze(vreal,axis=-1)) + s_LF,itstats)


class Gradf_y(tf.keras.layers.Layer):
    def __init__(self,mu0,*args,**kwargs):
        self.mu = mu0
    def call(self,x):
        y,x0 = x
        return self.mu*(y - x0)

class Gradf(tf.keras.layers.Layer):
    def __init__(self,mucurr,munext,dmul,dhmul,dnextmul,*args,**kwargs):
        self.mu = tf.Variable(initial_value=mucurr,trainable=True)
        self.munext =  munext
        self.dmul = dmul
        self.dhmul = dhmul
        self.dnextmul = dnextmul
        
    def call(self,x):
        xprev,xcurr,xnext = x
        return -self.mucurr*self.dhmul.freezeD(xprev) + self.mucurr*self.dhmul.freezeD(self.dmul.freezeD(xcurr)) + self.munext*xcurr - self.munext*self.dnextmul(xnext)

    def freezeD(self,x):
        xprev,xcurr,xnext = x
        return -self.mucurr*self.dhmul.freezeD(xprev) + self.mucurr*self.dhmul.freezeD(self.dmul.freezeD(xcurr)) + self.munext*xcurr - self.munext*self.dnextmul.freezeD(xnext)


class Gradf_Last(tf.keras.layers.Layer):
    def __init__(self,mu_init,dmul,dhmul,*args,**kwargs):
        self.mu = tf.Variable(initial_value=mu_init,trainable=True)
        self.dhmul = dhmul
        self.dmul = dmul
    def call(self,x):
        xprev,xcurr = x
        return -self.mu*self.dhmul(xprev) + self.mu*self.dhmul(self.dmul(xcurr))
    def freezeD(self,x):
        xprev,xcurr = x
        return -self.mu*self.dhmul.freezeD(xprev) + self.mu*self.dhmul.freezeD(self.dmul.freezeD(xcurr))


class Shrink(tf.keras.layers.Layer):
    def __init__(self,thrsh,IFFT,FFT,*args,**kwargs):
        self.thrsh=thrsh
        self.IFFT = IFFT
        self.FFT = FFT
        super().__init__(*args,**kwargs)
    def call(self,x):
        return self.FFT(self.RELU(self.IFFT(x) - thrsh))
    def get_config(self):
        config_dict = {'threshold': self.thrsh}
        return config_dict

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
        Df = self.FFT(self.upDtD(tf.complex(self.Dfreal,self.Dfimag)))
        self.Dfreal.assign(Df.real)
        self.Dfimag.assign(Df.imag)
        

class DMul(tf.kaers.layers.Layer):
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
