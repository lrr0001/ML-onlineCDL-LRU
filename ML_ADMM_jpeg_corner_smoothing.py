import tensorflow as tf
import matrix_decompositions_tf as fctr
import optmz
import jpeg_related_functions as jrf
import transforms as transf
import util
import numpy as np
import post_process_grad as ppg
import ML_ADMM_2d as mladmm


class MultiLayerCSC_JPEGY(mladmm.MultiLayerCSC):
    def __init__(self,rho,alpha_init,mu_init,b_init,blkSmthCoef,qY,cropAndMerge,fftSz,strides,D,n_components,noi,noL,cmplxdtype,longitstat=False,*args,**kwargs):
        rho,mu_init = self.init_param(rho,alpha_init,mu_init,cropAndMerge,noi,noL,cmplxdtype,longitstat,*args,**kwargs)
        self.qY = qY
        self.blkSmthCoef = blkSmthCoef
        self.noc = D[0].shape[-2]
        self.initializeLayers(rho,mu_init,alpha_init,util.makelist(b_init,noL),noL,fftSz,strides,D,n_components,cmplxdtype)
        self.initializeInputHandlingLayers(fftSz)

    def initializeInputHandlingLayers(self,fftSz):
        if self.noL > 1:
            mu = self.updateZ_layer[0].mu
        else:
            mu = self.updateZ_lastlayer.mu
        noc = self.dictObj[0].divide_by_R.D.shape[-2]
        self.updateeta = UpdateEta(alpha = self.alpha,dtype = self.cmplxdtype.real_dtype)
        xshapePair = tuple(self.cropAndMerge.blkPaddingTuple[ii][0] + self.cropAndMerge.blkPaddingTuple[ii][1] + fftSz[0][ii] for ii in range(2)) 
        xshapeSubsequent = (self.noc,)
        xshapeinit = xshapePair + xshapeSubsequent
        croppedshapePair = tuple(fftSz[0][ii] - self.cropAndMerge.paddingTuple[ii][0] - self.cropAndMerge.paddingTuple[ii][1]  for ii in range(2)) 
        self.smoothblksinit = jrf.Smooth_Blk_Edges(xshape = xshapeinit,a = self.blkSmthCoef/mu,blkSz = (8,8),dtype = self.cmplxdtype.real_dtype)
        self.smoothblks = jrf.Smooth_Blk_Edges(xshape = croppedshapePair + (self.noc,), a = self.blkSmthCoef/(self.rho + mu),blkSz = (8,8),dtype = self.cmplxdtype.real_dtype)
        self.measureblkedges = jrf.Measure_Blk_Edges(xshape = croppedshapePair + (self.noc,), blkSz = (8,8),dtype = self.cmplxdtype.real_dtype)
        self.W = jrf.Y2JPEG_Coef(dtype = self.cmplxdtype.real_dtype)
        self.Wt = jrf.JPEG_Coef2Y(dtype = self.cmplxdtype.real_dtype)
        self.jpeg_update = jrf.ZUpdate_JPEGY_Implicit(self.qY,self.W,self.Wt,dtype = self.cmplxdtype.real_dtype)

    def get_config(self):
        config_dict = {'complex_dtype': self.cmplxdtype,
                       'num_of_Layers': self.noL,
                       'strides': self.strides,
                       'rho': self.rho,
                       'num_of_iterations': self.noi,
                       'record_iteration_stats': self.longitstat,
                       'qY': self.qY,
                       'block_smooth_coef': self.blkSmthCoef}
        return config_dict

    def preprocess(self,s):
        s_HP,s_LP,compressed = s
        v = self.cropAndMerge.blkcrop(self.smoothblksinit(self.cropAndMerge.blkpad(s_HP)))
        return (v,s_LP,compressed)

    def get_output(self,s,y,u,By,negC,itstats):
        s_HP,s_LP,s_cmprsed = s
        s_LP_cropped, QWs = negC
        for layer in range(self.noL - 1,0,-1):
            x,Ax = self.xstep_trunc(y,u,By,negC,layer)
            u = self.relax_trunc(u,Ax,By,negC,layer)
            y,By = self.ystep_trunc(x,u,Ax,negC,layer,frozen=False)
            u = self.ustep_trunc(u,Ax,By,negC,layer)

        x,Ax = self.xstep_trunc(y,u,By,negC,layer=0)
        Dx = self.dictObj[0].dmul_sp(x[0])
        output1_hp = self.cropAndMerge.crop(tf.squeeze(Dx,axis=-1))
        output1 = output1_hp + s_LP_cropped
        output2 = self.jpeg_update((self.cropAndMerge.crop(tf.squeeze(Dx,axis=-1)) + s_LP_cropped,QWs))
        return ( output1,output2,itstats)
    def get_negative_C(self,s):
        s_HF,s_LF,compressed = s
        return (self.cropAndMerge.crop(s_LF),self.W(compressed))
    def get_obj(self,y,negC):
        v,z = y
        s_LP,QWs = negC
        return self.data_fid_z(y,self.Wt(QWs) - s_LP) + self.coef_penalty(z) + self.measureblkedges(self.cropAndMerge.crop(v))

    def init_x(self,s,negC):
        '''Initializes x for all layers. Ax includes an empty slot up front, anticipating its use later. While it would make more sense to add the variable to x as well, naming gets messy, so I'll leave that be. My plan for the added variable has Axzero = -xzero, so storing it with the rest of x is unnecessary, since its value can be accessed from By in the x and u updates. The second element of tuple Ax would more accurately be the negative of what it is in this code, but there's no reason to store the negative version of the same variable. Instead, the equations flip the sign whenever Ax[1] is accessed.'''
        s_HP,temp,temp = s
        s_LP,QWs = negC
        x = []
        x.append(self.xinit(tf.expand_dims(s_HP,axis=-1),layer = 0))
        for ii in range(1,self.noL):
            if self.strides[ii - 1] == 2:
                x.append(self.xinit(util.freq_downsample(x[ii - 1]),layer=ii))
            else:
                x.append(self.xinit(x[ii - 1],layer = ii))
        Azero = self.cropAndMerge.crop(s_HP)#what goes here?
        Ax = (Azero,x)
        return x,Ax
    def init_y(self,s,x,Ax,negC):
        Azero,Ax_layers = Ax
        s_LP,QWs = negC
        v,Bv = self.vinit(s,x[0])
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
        s_LP_cropped,QWs = negC
        gamma = []
        for ii in range(self.noL):
            gamma.append(self.gammainit(Ax_layers[ii],Bz[ii],ii))
        eta = self.etainit(Azero,Bzero,s_LP_cropped)
        return (eta,gamma)
    def ystep(self,x,uhalf,Ax,negC):
        ''' This code computes the updates for z, v, and Bzero. The other step functions use a truncated step function. However, while it is good practice to avoid code redundancy, here the last layer is distinct from ystep_trunc. ystep_trunc also does not freeze the dictionary weights and does not use updateZlast. If-else logic could resolve these distinctions, but I find it clearer to leave these as two separate functions.'''
        eta,gamma = uhalf
        Azero,Ax_layers = Ax
        v_initial,QWs = negC
        z = []
        for ii in range(self.noL - 1):
            z.append(self.updateZ(x_nextlayer=x[ii + 1],Ax=Ax_layers[ii],gamma_scaled=gamma[ii],layer=ii))
        z.append(self.updateZ_last(Ax=Ax_layers[self.noL - 1],gamma_scaled=gamma[self.noL - 1]))
        v,Bzero = self.updateV(x[0],eta,Azero)
        y = (v,z)
        By = (Bzero,z)
        return y,By
    def xstep_trunc(self,y,u,By,negC,layer): # Need to add new constraint.
        s_LP_cropped,QWs = negC
        v,z = y
        eta,gamma = u
        Bzero,z_again = By
        x = []
        x.append(self.updateX(z_prevlayer=v,z=z[0],gamma_scaled = gamma[0],layer=0))
        for ii in range(1,layer + 1):
            if self.strides[ii - 1] == 2:
                x.append(self.updateX(z_prevlayer=util.freq_downsample(z[ii - 1]),z = z[ii],gamma_scaled = gamma[ii],layer=ii))
            else:
                x.append(self.updateX(z_prevlayer=z[ii - 1],z = z[ii],gamma_scaled = gamma[ii],layer=ii))
        Azero = self.jpeg_update((Bzero + eta + s_LP_cropped,QWs)) - s_LP_cropped
        return x,(Azero,x)
    def ystep_trunc(self,x,u,Ax,negC,layer,frozen=True):
        eta,gamma = u
        Azero,Ax_layers = Ax
        v_initial,QWs = negC
        z = []
        for ii in range(layer - 1):
            z.append(self.updateZ(x_nextlayer=x[ii + 1],Ax=Ax_layers[ii],gamma_scaled=gamma[ii],layer=ii))
        z.append(self.updateZ(x_nextlayer=x[layer],Ax=Ax_layers[layer - 1],gamma_scaled=gamma[layer - 1],layer=layer - 1,frozen=frozen))
        v,Bzero = self.updateV(x[0],eta,Azero)
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
        eta = self.updateEta(eta,Azero,Bv,relax_bool)
        return (eta,gamma)
    def coef_penalty(self,y):
        v, z = y
        penalty_sum = (self.blkSmthCoef/2)*self.measureblkedges(self.cropAndMerge.crop(v))
        if self.noL > 1:
            penalty_sum += self.penaltyTerm(z[0],0)
        for ll in range(1,self.noL - 1):
            penalty_sum += self.penaltyTerm(z[ll],ll)
        R = tf.reshape(self.dictObj[-1].divide_by_R.R,shape=(1,1,1,self.dictObj[-1].divide_by_R.R.shape[4],1))
        penalty_sum += self.penaltyTerm(z[self.noL - 1]*R,self.noL - 1)
        return penalty_sum
    def cnstrPenalty(self,u,Ax,By,negC):
        s_lp_cropped,QWs = negC
        eta,gamma = u
        Azero,Ax_layers = Ax
        Bzero,Bz = By
        cnstr_penalty_sum = (self.rho/2)*self.reconstructionTerm_sp(Bzero + eta,Azero)
        for ll in range(self.noL - 1):
            R = tf.reshape(self.dictObj[ll].divide_by_R.R,shape=(1,1,1,self.dictObj[ll].divide_by_R.R.shape[4],1))
            mu = self.updateZ_layer[ll].mu
            cnstr_penalty_sum += (mu*self.rho/2)*self.reconstructionTerm_sp(Bz[ll]/R + gamma[ll],Ax_layers[ll])
        mu = self.updateZ_lastlayer.mu
        cnstr_penalty_sum += (mu*self.rho/2)*self.reconstructionTerm_sp(Bz[self.noL - 1] + gamma[self.noL - 1],Ax_layers[self.noL - 1])
        return cnstr_penalty_sum
    def vinit(self,s,x_0):
        s_HP,s_LP,s_compressed = s
        return (tf.expand_dims(s_HP,axis = -1),self.cropAndMerge.crop(s_HP))
    def updateV(self,x_0,eta,Azero,frozen=True):
        if frozen:
            Dx = self.dictObj[0].dmul_sp.freezeD(x_0)
        else:
            Dx = self.dictObj[0].dmul_sp(x_0)
        Dx = tf.squeeze(Dx,axis = -1)
        if self.noL > 1:
            mu = self.updateZ_layer[0].mu
        else:
            mu = self.updateZlast.mu
        Bzero = self.smoothblks((mu*self.cropAndMerge.crop(Dx) + self.rho*(Azero - eta))/(mu + self.rho))
        return (tf.expand_dims(self.cropAndMerge.merge((Bzero,Dx)),axis=-1),Bzero)
    def updateEta(self,eta_over_rho,Azero,Bv,relax_bool): # Need to fix this, adding new constraint
        if relax_bool:
            return self.updateeta.halfstep((eta_over_rho,Bv,Azero))
        else:
            return self.updateeta((eta_over_rho,Bv,Azero))




class GetNextIterX(mladmm.GetNextIterX):
    pass

class GetNextIterZ(mladmm.GetNextIterZ):
    pass 

class GetNextIterZ_lastlayer(mladmm.GetNextIterZ_lastlayer):
    pass



class GetNextIterGamma(mladmm.GetNextIterGamma):
    pass

class UpdateEta(mladmm.UpdateEta):
    pass

class CropPadObject(mladmm.CropPadObject):
    pass
