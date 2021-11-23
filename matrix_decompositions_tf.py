import tensorflow as tf
import transforms as transf
import numpy as np
import post_process_grad as ppg
import tf_rewrites as tfr
import util

class dictionary_object2D(tf.keras.layers.Layer,ppg.PostProcess,ppg.CondPostProcess):
    def __init__(self,fltrSz,fftSz,noc,nof,rho,objname,n_components,epsilon=1e-15,cmplxdtype=tf.complex128,*args,**kwargs):
        tf.keras.layers.Layer.__init__(self,name=objname,dtype=cmplxdtype,*args,**kwargs)
        self.fftSz = fftSz
        self.noc = noc
        self.nof = nof
        self.fltrSz = fltrSz
        self.epsilon = epsilon
        self.rho = rho
        self.n_components = n_components
        self.FFT = transf.fft2d_inner(fftSz)
        Df = self.init_dict(fltrSz=fltrSz,fftSz=fftSz,noc=noc,nof=nof,name=self.name)
        self.dhmul = DhMul(Df,*args,dtype=self.dtype,name=self.name + '/dhmul',**kwargs)
        self.dmul = DMul(self.dhmul,*args,dtype=self.dtype,name=self.name + 'dmul',**kwargs)
        self.qinv = QInv(self.dmul,self.dhmul,noc,nof,rho,*args,dtype=self.dtype,name=self.name + 'qinv',**kwargs)
        self.get_constrained_D = ifft_trunc_normalize(fltrSz,fftSz,noc,dtype=self.dtype)

        ppg.PostProcess.add_update(self.dhmul.varname,self._dict_update)
        self.build_shift_test(5)
        #self.state_save()
        

    def get_config(self):
        return {'fftSz': self.fftSz,'noc': self.noc,'nof':self.nof,'fltrSz': self.fltrSz,'epsilon': self.epsilon,'rho': self.rho,'n_components': self.n_components}

    def init_dict(self,fltrSz,fftSz,noc,nof,name):
        assert(tf.dtypes.as_dtype(self.dtype).is_complex)
        Drand = tf.random.normal(shape=(1,) + fltrSz + (noc,nof),dtype=tf.dtypes.as_dtype(self.dtype).real_dtype)
        #Dmeaned = Drand - tf.math.reduce_mean(input_tensor=Drand,axis = (1,2),keepdims=True)
        #Dnormalized = noc*Dmeaned/tf.math.sqrt(tf.reduce_sum(input_tensor=Dmeaned**2,axis=(1,2,3),keepdims=True))
        Dnormalized = Drand/tf.math.sqrt(tf.reduce_sum(input_tensor=Drand**2,axis=(1,2,3),keepdims=True))
        self.divide_by_R = Coef_Divide_By_R(Dnormalized,noc,name=name + 'div_by_R',dtype=self.dtype)
        with tf.name_scope(self.name):
            self.DbeforeApprox = tf.Variable(initial_value=self.divide_by_R.D,trainable=False,name='D_before_approx')
        #self.D = tf.Variable(initial_value=Dnormalized,trainable=False)
        #self.R = tf.Variable(initial_value = self.computeR(self.D),trainable=False) # keras may not like this
        return self.FFT(self.divide_by_R.D)    

    def Dmul(self,inputs):
        return self.dmul(inputs)

    def Dhmul(self,inputs):
        return self.dhmul(inputs)

    def call(self,inputs):
        return self.qinv(inputs)

    def freezeD(self,inputs):
        return self.qinv.freezeD(inputs)

#    def computeR(self,D):
#        return tf.math.sqrt(tf.math.reduce_sum(input_tensor=D**2,axis=(1,2,3),keepdims=True))/self.noc

    def _dict_update(self):
        return self._dict_update_LR()

    def _dict_update_LR(self):
        Dnew = self.get_constrained_D(tf.complex(self.dhmul.Dfreal,self.dhmul.Dfimag))
        DbeforeApprox = self.DbeforeApprox.assign(Dnew)
        # compute low rank approximation of the update
        theUpdate = Dnew - self.divide_by_R.D
        U,V,approx = stack_svd(theUpdate,5,n_components = self.n_components)
 
        # Update Spatial-Domain Dictionary and Normalization Factor
        D = self.divide_by_R.D.assign_add(approx)
        R = self.divide_by_R.R.assign(computeR(D,D.shape[4]))
        
        # Compute DFT (The conjugate is necessary because F{A} = F{U}F{V}^T
        Uf = tf.cast(U,tf.complex128)#util.complexNum(U)
        Vf = tf.math.conj(self.FFT(tf.cast(V,tf.float64)))

        # Update Decomposition and Frequency-Domain Dictionary
        Df,L = self._update_decomposition_LR(Uf,Vf,self.dhmul.Dfprev,self.qinv.L)
        return [D,R,DbeforeApprox,self.dhmul.Dfreal.assign(tf.math.real(Df)),self.dhmul.Dfimag.assign(tf.math.imag(Df)),L]

    def _update_decomposition_LR(self,U,V,Dfprev,L):
        L = self._rank1_updates(U,V,L)
        L,asVec = self._rank2_updates(U,V,Dfprev,L)
        # Update dictionary 
        with tf.control_dependencies([asVec]):
            Dfprev = self.dhmul.Dfprev.assign(Dfprev + tf.cast(U @ util.conj_tp(V),self.dtype))
        return Dfprev,L

    def _rank1_updates(self,U,V,L):
         # rank-1 Hermitian updates        
        if self.qinv.wdbry:
            eigvals,eigvecs = tf.linalg.eigh(tf.linalg.matmul(V,V,adjoint_a=True))
            for val,vec in zip(tf.unstack(eigvals,axis=-1),tf.unstack(tf.linalg.matmul(U,eigvecs),axis=-1)):
                L = tfr.cholesky_update(L,vec,val)#self.qinv.L.assign(tfr.cholesky_update(L,vec,val))
        else:
            # Redundant Computation: This is also computed in the rank-2 eigendecomposition
            UhU = tf.math.reduce_sum(tf.math.conj(U)*U,axis=3,keepdims=False) # conjugate unnecessary: U is real.
            for v,uhu in zip(tf.unstack(V,axis=-1),tf.unstack(UhU,axis=-1)):
                #L = self.qinv.L.assign(tfr.cholesky_update(L,v,uhu))
                L = tfr.cholesky_update(L,v,uhu)
        return self.qinv.L.assign(L)

    def _rank2_updates(self,U,V,Dfprev,L):
        eigvals,eigvecs,asVec = self._get_eigen_decomp(U,V,tf.cast(Dfprev,tf.complex128))
        L = self._eig_chol_update(eigvals,eigvecs,L)
        return self.qinv.L.assign(L),asVec

    def _get_eigen_decomp(self,U,V,Dfprev):
        if self.qinv.wdbry:
            asVec = tf.linalg.matmul(Dfprev,V)
            asVec = util.rotate_dims_right(asVec)
            Ushifted = util.rotate_dims_right(U)
            eigvals,eigvecs = rank2eigen(Ushifted,asVec,self.epsilon)
        else:
            asVec = tf.linalg.matmul(Dfprev,U,adjoint_a = True)
            asVec = util.rotate_dims_right(asVec)
            Vshifted = util.rotate_dims_right(V)
            eigvals,eigvecs = rank2eigen(Vshifted,asVec,self.epsilon)
        return eigvals,eigvecs,asVec

    def _eig_chol_update(self,eigvals,eigvecs,L):
        for vals,vecs in zip(eigvals,eigvecs):
            for val,vec in zip(tf.unstack(vals,axis=0),tf.unstack(vecs,axis=0)):
                #L = self.qinv.L.assign(tfr.cholesky_update(L,vec,val))
                L = tfr.cholesky_update(L,vec,val)
        return L

    def _dict_update_full(self):
        Df = tf.complex(self.dhmul.Dfreal,self.dhmul.Dfimag)
        Dnew = self.get_constrained_D(Df)
        return self._dict_update_full_from_D(Dnew)

    def _dict_update_reset_drift(self):
        return self._dict_update_full_from_D(self.DbeforeApprox)


    def _dict_update_full_from_D(self,Dnew):
        D = self.divide_by_R.D.assign(Dnew)
        R = self.divide_by_R.R.assign(computeR(Dnew,Dnew.shape[4]))
        Dfprev = self.dhmul.Dfprev.assign(self.FFT(D))
        with tf.control_dependencies([Dfprev]):
            Df,L = self._update_decomposition_full()
        Dfr = self.dhmul.Dfreal.assign(tf.math.real(Df))
        Dfi = self.dhmul.Dfimag.assign(tf.math.imag(Df))
        return [D,R,Dfr,Dfi,L]
        #return [D,R]

    def _update_decomposition_full(self,U=None,V=None):
        Dfprev = tf.cast(self.dhmul.Dfprev,tf.complex128)
        if self.qinv.wdbry:
            idMat = tf.linalg.eye(num_rows = self.noc,batch_shape = (1,1,1),dtype=tf.complex128)
            L = tf.linalg.cholesky(tf.cast(self.rho,tf.complex128)*idMat + tf.linalg.matmul(a = Dfprev,b = Dfprev,adjoint_b = True))
        else:
            idMat = tf.linalg.eye(num_rows = self.nof,batch_shape = (1,1,1),dtype=tf.complex128)
            L = tf.linalg.cholesky(tf.cast(self.rho,tf.complex128)*idMat + tf.linalg.matmul(a = Dfprev,b = Dfprev,adjoint_a = True))
        L = self.qinv.L.assign(L)
        return self.dhmul.Dfprev,L

    def build_shift_test(self,number_of_samples):
        self.constx = tf.constant(self.FFT(tf.random.normal(shape=(number_of_samples,) + self.fftSz + (self.nof,1,),dtype=tf.as_dtype(self.dtype).real_dtype)),dtype=self.dtype)
        ppg.CondPostProcess.add_cupdate(self.dhmul.varname,tf.function(self.shift_test),self._dict_update_reset_drift)
        return None

    def shift_test(self):
        y = self.rho*self.constx + self.dhmul(self.dmul(self.constx))
        z = self.qinv(y)
        return tf.math.reduce_max(tf.math.abs(self.constx - z))

    def state_save(self):
        ppg.StateSaveProcess.add_save(self.dhmul.varname,self._return_states)


    def _return_states(self):
        return {'dictionary_before': tf.keras.backend.get_value(self.dhmul.Dfprev), 'dictionary_current':tf.keras.backend.get_value(self.dhmul.Dfreal) + 1j * tf.keras.backend.get_value(self.dhmul.Dfimag),'L':tf.keras.backend.get_value(self.qinv.L),'R': tf.keras.backend.get_value(self.divide_by_R.R)}


class dictionary_object2D_init(dictionary_object2D):
    def __init__(self,fftSz,D,rho,objname,n_components=3,cmplxdtype=tf.complex128,epsilon=1e-15,*args,**kwargs):
        cmplxdtype = util.complexify_dtype(D.dtype)
        tf.keras.layers.Layer.__init__(self,dtype=cmplxdtype,name=objname,*args,**kwargs)
        self.fftSz = fftSz
        self.noc = D.shape[-2]
        self.nof = D.shape[-1]
        self.fltrSz = D.shape[1:3]
        self.epsilon = epsilon
        self.rho = rho
        self.n_components = n_components
        self.FFT = transf.fft2d_inner(self.fftSz)
        Df = self.init_dict(fftSz=fftSz,D=D,name=self.name)

        self.dhmul = DhMul(Df,*args,dtype=self.dtype,name=self.name + '/dhmul',**kwargs)
        self.dmul = DMul(self.dhmul,*args,dtype=self.dtype,name=self.name + '/dmul',**kwargs)
        self.qinv = QInv(self.dmul,self.dhmul,self.noc,self.nof,rho,*args,dtype=self.dtype,name=self.name + '/qinv',**kwargs)
        self.get_constrained_D = ifft_trunc_normalize(self.fltrSz,self.fftSz,self.noc,dtype=self.dtype)

        ppg.PostProcess.add_update(self.dhmul.varname,self._dict_update)
        self.build_shift_test(5)
        #self.state_save()


    def get_config(self):
        return {'fftSz': self.fftSz,'noc': self.noc,'nof':self.nof,'fltrSz': self.fltrSz,'epsilon': self.epsilon,'rho': self.rho,'n_components': self.n_components}
        
    def init_dict(self,fftSz,D,name):
        assert(tf.dtypes.as_dtype(self.dtype).is_complex)
        #Dmeaned = D - tf.math.reduce_mean(input_tensor=D,axis = (1,2),keepdims=True)
        #Dnormalized = D.shape[-2]*Dmeaned/tf.math.sqrt(tf.reduce_sum(input_tensor=Dmeaned**2,axis=(1,2,3),keepdims=True))
        Dnormalized = D/tf.math.sqrt(tf.reduce_sum(input_tensor=D**2,axis=(1,2,3),keepdims=True))
        #self.D = tf.Variable(initial_value=Dnormalized,trainable=False)
        #self.R = tf.Variable(initial_value = self.computeR(self.D),trainable=False)
        #return transf.fft2d_inner(fftSz)(self.D)
        noc = D.shape[-2]
        self.divide_by_R = self.get_divide_by_R(Dnormalized,noc,name=name + 'div_by_R',dtype = self.dtype)
        with tf.name_scope(self.name):
            self.DbeforeApprox = tf.Variable(initial_value=self.divide_by_R.D,trainable=False,name='D_before_approx')
        return self.FFT(self.divide_by_R.D)
    def get_divide_by_R(self,Dnormalized,noc,name,dtype)
        return Coef_Divide_By_R(Dnormalized,noc,name=name,dtype=dtype)


class dictionary_object2D_full(dictionary_object2D):
    def _dict_update(self):
        return self._dict_update_full()
    def build_shift_test(self,num_of_samples):
        return None

class dictionary_object2D_init_full(dictionary_object2D_init):
    def _dict_update(self):
        return self._dict_update_full()
    def build_shift_test(self,num_of_samples):
        return None



# ***************** TIGHT-FRAME CLASSES *****************
class dictionary_object2D_tight_frame_init(tf.keras.layers.Layer):
    def __init__(self,D,rho,objname,*args,**kwargs):
        super().__init__(name=objname,dtype = D.dtype,*args,**kwargs)
        self.rho = rho
        self.D = tf.Variable(initial_value = D,trainable=True)
        self.dmul = Conv2D(D = self.D,dtype = self.dtype)
        self.dhmul = Conv2D_Transpose(D=self.D,dtype = self.dtype)
        self.qinv = Tight_Frame_Inverse(self.rho,self.D,self.dmul,self.dhmul,dtype=self.dtype)
        self.qinvDT = Tight_Frame_Inverse_DT(self.rho,self.D,self.dhmul,dtype=self.dtype)
    def call(self,inputs):
        zcurr,zprev = inputs
        return self.qinv(zcurr) + self.qinvDT(zprev)

class Conv2D(tf.keras.layers.Layer):
    def __init__(self,D,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.D = D
    def call(self,inputs):
        return tf.nn.conv2d(input = inputs,filters = self.D,padding='SAME')

class Conv2D_Transpose(Conv2D):
    def call(self,inputs):
        return tf.nn.conv2d_transpose(input = inputs,filters = self.D,output_shape = inputs.shape, padding= 'SAME')

class Tight_Frame_Inverse(tf.keras.layers.Layer):
    def __init__(self,rho,D,dmul,dhmul,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rho = rho
        self.dmul = dmul
        self.dhmul = dhmul
    def call(self,inputs):
        return inputs - 1./(self.rho + 1.)*dhmul(dmul(inputs))

class Tight_Frame_Inverse_DT(tf.keras.layers.Layer):
    def __init__(self,rho,D,dhmul,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rho = rho
        self.dhmul = dhmul
    def call(self,inputs):
        return 1./(self.rho + 1.)*self.dhmul(inputs)


# *******************BACK TO OUR MAIN PROGRAMMING**************************************


class Solve_Inverse(tf.keras.layers.Layer):
    # I want auto-differentiation of the input, but not of the weights.
    # The solution? Create a pass-through "gradient layer" that computes the gradient for the weights.
    def __init__(self,dmul,dhmul,L,wdbry,*args,**kwargs):
        self.dmul = dmul
        self.dhmul = dhmul
        self.L = L
        self.wdbry = wdbry
        super().__init__(*args,**kwargs)
    @tf.custom_gradient
    def gradient_trick(self,y,Df):
        def grad(dg):
            halfway = tf.linalg.triangular_solve(matrix=self.L,rhs=tf.cast(dg,tf.complex128),lower=True)
            ainvdg = tf.cast(tf.linalg.triangular_solve(matrix=self.L,rhs=halfway,lower=True,adjoint=True),self.dtype)
            if self.wdbry:
                Dhy = self.dhmul(y)
                DhyH = util.conj_tp(Dhy)
                Dhainvdg = self.dhmul(ainvdg)
                DhainvdgH = util.conj_tp(Dhainvdg)
                gradD = -ainvdg*DhyH - y*DhainvdgH
            else:
                Dy = self.dmul(y)
                Dainvdg = self.dmul(ainvdg)
                yH = util.conj_tp(y)
                ainvdgH = util.conj_tp(ainvdg)
                gradD = -Dy*ainvdgH - Dainvdg*yH
            return (tf.identity(dg),tf.math.reduce_sum(input_tensor=gradD,axis=0,keepdims=True))
        return tf.identity(y),grad
    def get_config(self):
        return {'wdbry': self.wdbry}
    def call(self, x):
        Df = tf.complex(self.dhmul.Dfreal,self.dhmul.Dfimag)
        output = self.freezeD(x)
        return self.gradient_trick(output,Df)
    def freezeD(self,x):
        x = tf.cast(x,tf.complex128)
        halfway = tf.linalg.triangular_solve(matrix=self.L,rhs=x,lower=True)
        return tf.cast(tf.linalg.triangular_solve(matrix=self.L,rhs=halfway,lower=True,adjoint=True),self.dtype)

class QInv(tf.keras.layers.Layer):
    def __init__(self,dmul,dhmul,noc,nof,rho,*args,**kwargs):
        # layers are included in the inputs for the purposes of sharing weights.
        super().__init__(*args,**kwargs)
        self.dmul = dmul
        self.dhmul = dhmul
        self.rho = rho
        self.init_chol(noc,nof)

        self.solve_inverse = Solve_Inverse(dmul = dmul,dhmul = dhmul,L = self.L,wdbry=self.wdbry,*args,**kwargs)

    def get_config(self):
        return {'rho': self.rho}

    def call(self, inputs):
        if self.wdbry:
            y = self.dmul(inputs)
            z = self.solve_inverse(y)
            return (inputs - self.dhmul(z))/self.rho
        else:
            return self.solve_inverse(inputs)

    def freezeD(self, inputs):
        if self.wdbry:
            y = self.dmul.freezeD(inputs)
            z = self.solve_inverse.freezeD(y)
            return (inputs - self.dhmul.freezeD(z))/self.rho
        else:
            return self.solve_inverse.freezeD(inputs)

    def init_chol(self,noc,nof):
        Df = tf.complex(tf.cast(self.dhmul.Dfreal,tf.float64),tf.cast(self.dhmul.Dfimag,tf.float64))
        if noc <= nof:
            idMat = tf.linalg.eye(num_rows = noc,batch_shape = (1,1,1),dtype=tf.complex128)
            L = tf.linalg.cholesky(tf.cast(self.rho,tf.complex128)*idMat + tf.linalg.matmul(a = Df,b = Df,adjoint_b = True))
            self.wdbry = True
        else:
            idMat = tf.linalg.eye(num_rows = nof,batch_shape = (1,1,1),dtype=tf.complex128)
            L = tf.linalg.cholesky(tf.cast(self.rho,tf.complex128)*idMat + tf.linalg.matmul(a = Df,b= Df,adjoint_a = True))
            self.wdbry = False
        self.L = tf.Variable(initial_value = L,trainable=False,name='L')

class QInv_Tight_Frame(tf.keras.layers.Layer):
    def __init__(self,dmul,dhmul,rho,*args,**kwargs):
        # layers are included in the inputs for the purposes of sharing weights.
        super().__init__(*args,**kwargs)
        self.dmul = dmul
        self.dhmul = dhmul
        self.rho = rho

    def get_config(self):
        return {'rho': self.rho}

    def call(self, inputs):
        return (1./self.rho)*(inputs - self.dhmul(self.dmul(inputs))/(self.rho + 1.)) 



class QInv_auto(tf.keras.layers.Layer):
    def __init__(self,dhmul,rho,wdbry=False,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.Df = tf.Variable(initial_value=dhmul.Df,trainable=True,name='Dfreq') #What on earth is this???
        self.rho = rho
        raise NotImplementedError
    def get_config(self):
        return {'rho': self.rho}
    def call(self,inputs):
        if self.wdbry:
            Dx = tf.linalg.matmul(self.Df,inputs)
            idmat = tf.eye(num_rows = self.Df.shape[-2],batch_shape = (1,1,1),dtype = self.Df.dtype)
            ainv = tf.linalg.inv(self.rho*idmat + tf.linalg.matmul(self.Df,self.Df,adjoint_b=True))
            return 1/self.rho*(inputs - tf.linalg.matmul(self.Df,tf.linalg.matmul(ainv,Dx),adjoint_a = True))
        else:
            idmat = tf.eye(num_rows = self.Df.shape[-1],batch_shape= (1,1,1),dtype = self.Df.dtype)
            a = self.rho*idmat + tf.linalg.matmul(self.Df,self.Df,adjoint_a = True)
            inputs = tf.transpose(inputs,perm=(4,1,2,3,0))
            x = tf.linalg.solve(a,inputs)
            return tf.transpose(x,perm = (4,1,2,3,0))

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


class DhMul(tf.keras.layers.Layer):
    def __init__(self,Df,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with tf.name_scope(self.name):
            self.Dfprev = tf.Variable(initial_value=Df,trainable=False,dtype=self.dtype,name='Dfreq_previous')
            self.Dfreal = tf.Variable(initial_value=tf.math.real(Df),trainable=True,name='Dfreq_real')
            self.Dfimag = tf.Variable(initial_value=tf.math.imag(Df),trainable=True,name='Dfreq_imag')
        self.varname = self.Dfreal.name
    def get_config(self):
        return {'varname': self.varname}

    def call(self, inputs):
        Df = tf.complex(self.Dfreal,self.Dfimag)
        return tf.matmul(a=Df,b=inputs,adjoint_a=True)

    def freezeD(self, inputs):
        Df = tf.complex(self.Dfreal,self.Dfimag)
        return tf.matmul(a=tf.stop_gradient(Df),b=inputs,adjoint_a=True)

def get_lowrank_approx(A,*args,**kwargs):
    U,s,V = randomized_svd(A,*args,**kwargs)
    if A.shape[1] > A.shape[0]:
        U = U*tf.cast(tf.reshape(s,(1,-1)),U.dtype)
    else:
        V = V*tf.cast(tf.reshape(s,(1,-1)),V.dtype)
    return (U,V, tf.linalg.matmul(U,tf.transpose(V,conjugate=True)))


def randomized_svd(A, n_components=3, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto'):
    """ This code was adapted from the randomized_svd function from scikit-learn library.
    (sklearn.utils.extmath.randomized_svd)

    Computes a truncated randomized SVD

    Parameters
    ----------
    A : ndarray or sparse matrix
        Matrix to decompose

    n_components : int
        Number of singular values and vectors to extract.

    n_oversamples : int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.

    n_iter : int or 'auto' (default is 'auto')
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.

    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).

    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 https://arxiv.org/abs/0909.4061

    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    Nrand = n_components + n_oversamples
    M, N = A.shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < .1 * min(A.shape) else 4

    if transpose == 'auto':
        transpose = N < M
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        A = tf.transpose(A)
    if Nrand < M and Nrand < N:
        Q = randomized_range_finder(
            A, rangeSize=Nrand, n_iter=n_iter,
            power_iteration_normalizer=power_iteration_normalizer)
        B = tf.matmul(Q, A,adjoint_a=True)
        s, Uhat, V = tf.linalg.svd(tf.debugging.check_numerics(tensor=B,message='projected svd inf or NaN error'), full_matrices = False)

        if Uhat.dtype.is_complex and not Q.dtype.is_complex:
            U = tf.cast(Q,Uhat.dtype) @ Uhat[:, :n_components]
        else:
            U = Q @ Uhat[:, :n_components]
    else:
       s, U, V = tf.linalg.svd(tf.debugging.check_numerics(tensor=A,message='full svd inf or NaN error'), full_matrices = False)

    if transpose:
        # transpose back the results according to the input convention
        return tf.math.conj(V[:,:n_components]), s[:n_components], tf.math.conj(U[:,:n_components])
    else:
        return U[:,:n_components], s[:n_components], V[:,:n_components]

def randomized_range_finder(A, rangeSize, n_iter,
                            power_iteration_normalizer='auto'):
    """This code was adapted from the randomized_range_finder function in the scikit-learn library.
    (sklearn.utils.extmath.randomized_range_finder)

    Computes an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    A : 2D array of size (M,N)
        The input data matrix

    rangeSize : integer
        Number of columns of the return array

    n_iter : integer
        Number of power iterations used to stabilize the result

    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

    Returns
    -------
    Q : 2D array
        A (M x rangeSize) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----

    Follows Algorithm 4.3 of
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
    """
    dtype = A.dtype
    if dtype.is_complex:
        Q2 = tf.complex(tf.random.normal(shape=(A.shape[-1],rangeSize),dtype=dtype.real_dtype),tf.random.normal(shape=(A.shape[-1],rangeSize),dtype=dtype.real_dtype))
    else:
        Q2 = tf.random.normal(shape = (A.shape[-1],rangeSize),dtype=dtype)

    # Deal with "auto" mode
    if power_iteration_normalizer == 'auto':
        if n_iter <= 2:
            power_iteration_normalizer = 'none'
        else:
            power_iteration_normalizer = 'QR'

    # Perform power iterations of A with Q to increase eigenvalue separation
    for i in range(n_iter):
        if power_iteration_normalizer == 'none':
            Q2 = tf.linalg.matmul(A, (A @ Q2),adjoint_a=True)
        # LU factorization requires square matrix
        elif power_iteration_normalizer == 'LU':
            AQ = A @ Q2
            Q, _ = tf.linalg.lu(tf.linalg.matmul(AQ,AQ,adjoint_a = True))
            Q2 = Q2 @ Q
        elif power_iteration_normalizer == 'QR':
            Q, _ = tf.linalg.qr(A @ Q2)
            Q2, _ = tf.linalg.qr(tf.linalg.matmul(A, Q,adjoint_a=True))

    # Compute an orthonormal basis for a linear projection of A
    Q, _ = tf.linalg.qr(A @ Q2)
    return Q



class ifft_trunc_normalize(tf.keras.layers.Layer):
    def __init__(self,fltrSz,fftSz,noc,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ifft = transf.ifft2d_inner(fftSz)
        self.fltrSz = fltrSz
        self.noc = noc
    def get_config(self):
        return {'fltrSz': self.fltrSz, 'noc': self.noc}
    def call(self,inputs):
        D = self.ifft(inputs)
        Dtrunc = D[slice(None),slice(0,self.fltrSz[0],1),slice(0,self.fltrSz[1],1),slice(None),slice(None)]
        return Dtrunc/tf.math.sqrt(tf.reduce_sum(input_tensor=Dtrunc**2,axis=(1,2,3),keepdims=True))

def rank2eigen(U,V,epsilon=1e-15):
    vhv = tf.math.reduce_sum(tf.math.conj(V)*V,axis=-1,keepdims=True)
    uhu = tf.math.reduce_sum(tf.math.conj(U)*U,axis=-1,keepdims=True)
    uhv = tf.math.reduce_sum(tf.math.conj(U)*V,axis=-1,keepdims=True)
    rootRadicand = tf.math.sqrt(vhv*uhu - util.complexNum(tf.math.imag(uhv)**2))
        
    valPlus = util.complexNum(tf.math.real(uhv)) + rootRadicand
    valMinus = util.complexNum(tf.math.real(uhv)) - rootRadicand
    vecPlus = vhv*U + (1j*util.complexNum(tf.math.imag(uhv)) + rootRadicand)*V
    vecMinus = vhv*U + (1j*util.complexNum(tf.math.imag(uhv))  - rootRadicand)*V
    vecPlus = tf.where(tf.abs(rootRadicand) > epsilon,vecPlus,U)
    vecMinus = tf.where(tf.abs(rootRadicand) > epsilon,vecMinus,-tf.math.divide_no_nan(uhv,uhu)*U + V)
        
    vecPlus = tf.math.l2_normalize(vecPlus,axis=-1,epsilon=epsilon)
    vecMinus = tf.math.l2_normalize(vecMinus,axis=-1,epsilon=epsilon)
    valPlus = tf.reshape(valPlus,valPlus.shape[:-1])
    valMinus = tf.reshape(valMinus,valMinus.shape[:-1])
    return ((valPlus,valMinus),(vecPlus,vecMinus))

def stack_svd(x,r,**kwargs):
    s = r - 2
    forwardPerm = (s,) + tuple(range(s)) + (s + 1,)
    backwardPerm = tuple(range(1,s + 1)) + (0,s + 1,)
    y = tf.transpose(x,perm = forwardPerm)
    z = tf.reshape(y,(x.shape[s],-1,))
    U,V,approx = get_lowrank_approx(z,**kwargs)
    approx = tf.reshape(approx,[x.shape[ii] for ii in forwardPerm])
    approx = tf.transpose(approx,backwardPerm)
    U = tf.reshape(U,(1,)*s + tuple(U.shape))
    V = tf.reshape(V,tuple([x.shape[ii] for ii in forwardPerm[1:]]) + (V.shape[-1],))
    return U,V,approx

def computeR(D,noc):
    return tf.math.sqrt(tf.math.reduce_sum(input_tensor=D**2,axis=(1,2,3),keepdims=True))


class Coef_Divide_By_R(tf.keras.layers.Layer):
    def __init__(self,D,noc,trainableD=False,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with tf.name_scope(self.name):
            self.D = tf.Variable(initial_value = D,trainable=trainableD,name='D')
            self.R = tf.Variable(initial_value = computeR(D,noc),trainable=False,name='R')
            if trainableD:
                self.Dprev = tf.Variable(initial_value = D,trainable=False,name='Dprev')
    def call(self,inputs):
        R = tf.cast(tf.reshape(self.R,self.R.shape[:3] + (self.R.shape[4],self.R.shape[3],) + self.R.shape[5:]),dtype=self.dtype)
        return inputs/R
    def get_dict(self):
        return tf.reshape(self.D,self.D.shape[1:])



class MulD_Sp(tf.keras.layers.Layer):
    def __init__(self,D,*args,**kwargs):
        self.D = D
        self.noc = D.shape[2]
        self.nof = D.shape[3]
        super().__init__(*args,**kwargs)
    def call(self,inputs):
        x = tf.reshape(inputs,inputs.shape[:-2] + (inputs.shape[-1],))
        return tf.nn.conv2d_transpose(input = x,filters = self.D,output_shape = x.shape[:-1] + (self.noc,self.nof),strides = 1,padding='SAME',dilations=1)
    def freeze(self,inputs):
        x = tf.reshape(inputs,inputs.shape[:-2] + (inputs.shape[-1],))
        return tf.nn.conv2d_transpose(input = x,filters = tf.stop_gradient(self.D),output_shape = x.shape[:-1] + (self.noc,self.nof),strides = 1,padding='SAME',dilations=1)
    def get_config(self):
        config_dict = {'noc': self.noc,
                       'nof': self.nof}
        return config_dict

class MulDT_Sp(tf.keras.layers.Layer):
    def __init__(self,D,*args,**kwargs):
        self.D = D
        super().__init__(*args,**kwargs)
    def call(self,inputs):
        x = tf.reshape(inputs,inputs.shape[:-2] + (inputs.shape[-1],))
        return tf.nn.conv2d(input = x,filters = self.D,strides = 1,padding='SAME',dilations=1)
    def freeze(self,inputs):
        x = tf.reshape(inputs,inputs.shape[:-2] + (inputs.shape[-1],))
        return tf.nn.conv2d(input = x,filters = tf.stop_gradients(self.D),strides = 1,padding='SAME',dilations=1)


class dictionary_object_init_sp(dictionary_object2D_init)
    def __init__(self,fftSz,D,rho,objname,n_components=3,cmplxdtype=tf.complex128,epsilon=1e-15,*args,**kwargs):
        super().__init__(fftSz,D,rho,objname,n_components=3,cmplxdtype,epsilon,*args,**kwargs)
        self.dtmul_sp = MulDT_Sp(D,dtype=cmplxdtype.real_dtype)
        self.dmul_sp = MulD_Sp(D,dtype=cmplxdtype.real_dtype)
    def get_divide_by_R(self,Dnormalized,noc,name,dtype)
        return Coef_Divide_By_R(Dnormalized,noc,trainableD = True,name=name,dtype=dtype)
    def _dict_update_LR(self):
        Dnew = self.get_constrained_D(tf.complex(self.dhmul.Dfreal,self.dhmul.Dfimag)) + self.divide_by_R.D - self.divide_by_R.Dprev
        DbeforeApprox = self.DbeforeApprox.assign(Dnew)
        # compute low rank approximation of the update
        theUpdate = Dnew - self.divide_by_R.Dprev
        U,V,approx = stack_svd(theUpdate,5,n_components = self.n_components)
 
        # Update Spatial-Domain Dictionary and Normalization Factor
        Dprev = self.divide_by_R.Dprev.assign_add(approx)
        D = self.divide_by_R.D.assign(Dprev)
        R = self.divide_by_R.R.assign(computeR(Dprev,Dprev.shape[4]))
        
        # Compute DFT (The conjugate is necessary because F{A} = F{U}F{V}^T
        Uf = tf.cast(U,tf.complex128)#util.complexNum(U)
        Vf = tf.math.conj(self.FFT(tf.cast(V,tf.float64)))

        # Update Decomposition and Frequency-Domain Dictionary
        Df,L = self._update_decomposition_LR(Uf,Vf,self.dhmul.Dfprev,self.qinv.L)
        return [Dprev,D,R,DbeforeApprox,self.dhmul.Dfreal.assign(tf.math.real(Df)),self.dhmul.Dfimag.assign(tf.math.imag(Df)),L]
class dictionary_object_init_full_sp(dictionary_object2D_init_full):
    def __init__(self,fftSz,D,rho,objname,n_components=3,cmplxdtype=tf.complex128,epsilon=1e-15,*args,**kwargs):
        super().__init__(fftSz,D,rho,objname,n_components=3,cmplxdtype,epsilon,*args,**kwargs)
        self.dtmul_sp = MulDT_Sp(D,dtype=cmplxdtype.real_dtype)
        self.dmul_sp = MulD_Sp(D,dtype=cmplxdtype.real_dtype)
    def get_divide_by_R(self,Dnormalized,noc,name,dtype)
        return Coef_Divide_By_R(Dnormalized,noc,trainableD = True,name=name,dtype=dtype)
    def _dict_update_full(self):
        Df = tf.complex(self.dhmul.Dfreal,self.dhmul.Dfimag)
        Dnew = self.get_constrained_D(Df) + self.divide_by_R.D - self.divide_by_R.Dprev
        return self._dict_update_full_from_D(Dnew)
    def _dict_update_full_from_D(self,Dnew):
        output = super()._dict_update_full_from_D(Dnew)
        Dprev = self.divide_by_R.Dprev.assign(Dnew)
        return [Dprev,] + output
