import tensorflow as tf
import transforms as transf
import numpy as np
import post_process_grad as ppg
import tf_rewrites as tfr
import util

class dictionary_object2D(ppg.PostProcess):
    def __init__(self,fltrSz,fftSz,noc,nof,rho,name,lraParam = {},epsilon=1e-6,*args,dtype=tf.complex64,**kwargs):
        self.dtype = dtype
        self.fftSz = fftSz
        self.noc = noc
        self.nof = nof
        self.fltrSz = fltrSz
        self.epsilon = epsilon
        self.rho = rho
        self.lraParam = lraParam
        self.FFT = transf.fft2d_inner(fftSz)
        Df = self.init_dict(fltrSz=fltrSz,fftSz=fftSz,noc=noc,nof=nof,name=name)
        self.dhmul = DhMul(Df,*args,dtype=self.dtype,name=name + '/dhmul',**kwargs)
        self.dmul = DMul(self.dhmul,*args,dtype=self.dtype,name=name + 'dmul',**kwargs)
        self.qinv = QInv(self.dmul,self.dhmul,noc,nof,rho,*args,dtype=self.dtype,name=name + 'qinv',**kwargs)
        self.get_constrained_D = ifft_trunc_normalize(fltrSz,fftSz,noc,dtype=dtype)

        ppg.PostProcess.add_update(self.dhmul.varname,self._dict_update)
        
    def init_dict(self,fltrSz,fftSz,noc,nof,name):
        assert(tf.dtypes.as_dtype(self.dtype).is_complex)
        Drand = tf.random.normal(shape=(1,) + fltrSz + (noc,nof),dtype=tf.dtypes.as_dtype(self.dtype).real_dtype)
        #Dmeaned = Drand - tf.math.reduce_mean(input_tensor=Drand,axis = (1,2),keepdims=True)
        #Dnormalized = noc*Dmeaned/tf.math.sqrt(tf.reduce_sum(input_tensor=Dmeaned**2,axis=(1,2,3),keepdims=True))
        Dnormalized = Drand/tf.math.sqrt(tf.reduce_sum(input_tensor=Drand**2,axis=(1,2,3),keepdims=True))
        self.divide_by_R = Coef_Divide_By_R(Dnormalized,noc,name=name + 'div_by_R',dtype=self.dtype)
        #self.D = tf.Variable(initial_value=Dnormalized,trainable=False)
        #self.R = tf.Variable(initial_value = self.computeR(self.D),trainable=False) # keras may not like this
        return self.FFT(self.divide_by_R.D)    

    def Dmul(self,inputs):
        return self.dmul(inputs)

    def Dhmul(self,inputs):
        return self.dhmul(inputs)

    def Qinv(self,inputs):
        return self.qinv(inputs)

#    def computeR(self,D):
#        return tf.math.sqrt(tf.math.reduce_sum(input_tensor=D**2,axis=(1,2,3),keepdims=True))/self.noc

    def _dict_update(self):
        Dnew = self.get_constrained_D(self.dhmul.Df)

        # compute low rank approximation of the update
        theUpdate = Dnew - self.divide_by_R.D
        U,V,approx = stack_svd(theUpdate,5,**self.lraParam)
 
        # Update Spatial-Domain Dictionary and Normalization Factor
        D = self.divide_by_R.D.assign_add(approx)
        R = self.divide_by_R.R.assign(computeR(D,D.shape[4]))
        
        # Compute DFT (The conjugate is necessary because F{A} = F{U}F{V}^T
        Uf = util.complexNum(U)
        Vf = tf.math.conj(self.FFT(V))

        # Update Decomposition and Frequency-Domain Dictionary
        Df,L = self._update_decomposition(Uf,Vf,self.dhmul.Dfprev,self.qinv.L)
        return [D,R,self.dhmul.Df.assign(Df),self.qinv.L.assign(L)]

    def _update_decomposition(self,U,V,Dfprev,L):
        L = self._rank1_updates(U,V,L)
        L,asVec = self._rank2_updates(U,V,Dfprev,L)
        # Update dictionary
        with tf.control_dependencies([asVec]):
            Dfprev = self.dhmul.Dfprev.assign_add(U @ util.conj_tp(V))
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
        return L

    def _rank2_updates(self,U,V,Dfprev,L):
        eigvals,eigvecs,asVec = self._get_eigen_decomp(U,V,Dfprev)
        L = self._eig_chol_update(eigvals,eigvecs,L)
        return L,asVec

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

class dictionary_object2D_init(dictionary_object2D):
    def __init__(self,fftSz,D,rho,name,lraParam = {},epsilon=1e-6,*args,**kwargs):
        self.dtype = util.complexify_dtype(D.dtype)
        self.fftSz = fftSz
        self.noc = D.shape[-2]
        self.nof = D.shape[-1]
        self.fltrSz = D.shape[1:3]
        self.epsilon = epsilon
        self.rho = rho
        self.lraParam = lraParam
        self.name = name
        self.FFT = transf.fft2d_inner(self.fftSz)
        Df = self.init_dict(fftSz=fftSz,D=D,name=name)

        self.dhmul = DhMul(Df,*args,dtype=self.dtype,name=name + '/dhmul',**kwargs)
        self.dmul = DMul(self.dhmul,*args,dtype=self.dtype,name=name + '/dmul',**kwargs)
        self.qinv = QInv(self.dmul,self.dhmul,self.noc,self.nof,rho,*args,dtype=self.dtype,name=name + '/qinv',**kwargs)
        self.get_constrained_D = ifft_trunc_normalize(self.fltrSz,fftSz,self.noc,dtype=self.dtype)

        ppg.PostProcess.add_update(self.dhmul.varname,self._dict_update)
        
    def init_dict(self,fftSz,D,name):
        assert(tf.dtypes.as_dtype(self.dtype).is_complex)
        #Dmeaned = D - tf.math.reduce_mean(input_tensor=D,axis = (1,2),keepdims=True)
        #Dnormalized = D.shape[-2]*Dmeaned/tf.math.sqrt(tf.reduce_sum(input_tensor=Dmeaned**2,axis=(1,2,3),keepdims=True))
        Dnormalized = D/tf.math.sqrt(tf.reduce_sum(input_tensor=D**2,axis=(1,2,3),keepdims=True))
        #self.D = tf.Variable(initial_value=Dnormalized,trainable=False)
        #self.R = tf.Variable(initial_value = self.computeR(self.D),trainable=False)
        #return transf.fft2d_inner(fftSz)(self.D)
        noc = D.shape[-2]
        self.divide_by_R = Coef_Divide_By_R(Dnormalized,noc,name=name + 'div_by_R',dtype=self.dtype)
        return self.FFT(self.divide_by_R.D)

class dictionary_object2D_full(dictionary_object2D):
    def _dict_update(self):
        D = self.get_constrained_D(self.dhmul.Df)
        self.dhmul.Dfprev = self.dhmul.Dfprev.assign(self.FFT(D))
        Df,self.qinv.L = self._update_decomposition()
        return [self.divide_by_R.D.assign(D),self.divide_by_R.R.assign(computeR(D,D.shape[4])),self.dhmul.Df.assign(Df),self.qinv.L]

    def _update_decomposition(self,U=None,V=None):
        if self.qinv.wdbry:
            idMat = tf.linalg.eye(num_rows = self.noc,batch_shape = (1,1,1),dtype=tf.dtypes.as_dtype(self.dtype))
            L = tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b = self.dhmul.Df,adjoint_b = True))
        else:
            idMat = tf.linalg.eye(num_rows = self.nof,batch_shape = (1,1,1),dtype=tf.dtypes.as_dtype(self.dtype))
            L = tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b = self.dhmul.Df,adjoint_a = True))
        L = self.qinv.L.assign(L)
        return self.dhmul.Dfprev,L

class dictionary_object2D_init_full(dictionary_object2D_init):
    def _dict_update(self):
        D = self.get_constrained_D(self.dhmul.Df)  
        self.divide_by_R.D = self.divide_by_R.D.assign(D)
        self.divide_by_R.R = self.divide_by_R.R.assign(computeR(D,D.shape[4]))
        self.dhmul.Dfprev = self.dhmul.Dfprev.assign(self.FFT(D))
        Df,self.qinv.L = self._update_decomposition()
        return [self.divide_by_R.D,self.divide_by_R.R,self.dhmul.Df.assign(Df),self.qinv.L]

    def _update_decomposition(self,U=None,V=None):
        if self.qinv.wdbry:
            idMat = tf.linalg.eye(num_rows = self.noc,batch_shape = (1,1,1),dtype=tf.dtypes.as_dtype(self.dtype))
            L = tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Dfprev,b = self.dhmul.Dfprev,adjoint_b = True))
        else:
            idMat = tf.linalg.eye(num_rows = self.nof,batch_shape = (1,1,1),dtype=tf.dtypes.as_dtype(self.dtype))
            L = tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Dfprev,b = self.dhmul.Dfprev,adjoint_a = True))
        L = self.qinv.L.assign(L)
        return self.dhmul.Dfprev,L



class QInv(tf.keras.layers.Layer):
    def __init__(self,dmul,dhmul,noc,nof,rho,*args,**kwargs):
        # layers are included in the inputs for the purposes of sharing weights.
        super().__init__(*args,**kwargs)
        self.dmul = dmul
        self.dhmul = dhmul
        self.rho = rho
        self.init_chol(noc,nof)


        def solve_inverse(x):
            # I want auto-differentiation of the input, but not of the weights.
            # The solution? Create a pass-through "gradient layer" that computes the gradient for the weights.
 
            halfway = tf.linalg.triangular_solve(matrix=self.L,rhs=x,lower=True)
            output = tf.linalg.triangular_solve(matrix=self.L,rhs=halfway,lower=True,adjoint=True)

            @tf.custom_gradient
            def gradient_trick(y,Df):
                def grad(dg):
                    halfway = tf.linalg.triangular_solve(matrix=self.L,rhs=dg,lower=True)
                    ainvdg = tf.linalg.triangular_solve(matrix=self.L,rhs=halfway,lower=True,adjoint=True)
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
            return gradient_trick(output,self.dhmul.Df)

        self.solve_inverse = lambda x: solve_inverse(x)

    def get_config(self):
        return {'rho': self.rho}

    def call(self, inputs):
        if self.wdbry:
            y = self.dmul(inputs)
            z = self.solve_inverse(y)
            return (inputs - self.dhmul(z))/self.rho
        else:
            return self.solve_inverse(inputs)

    def init_chol(self,noc,nof):
        if noc <= nof:
            idMat = tf.linalg.eye(num_rows = noc,batch_shape = (1,1,1),dtype=tf.dtypes.as_dtype(self.dtype))
            L = tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b = self.dhmul.Df,adjoint_b = True))
            self.wdbry = True
        else:
            idMat = tf.linalg.eye(num_rows = nof,batch_shape = (1,1,1),dtype=tf.dtypes.as_dtype(self.dtype))
            L = tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b= self.dhmul.Df,adjoint_a = True))
            self.wdbry = False
        self.L = tf.Variable(initial_value = L,trainable=False)

class QInv_auto(tf.keras.layers.Layer):
    def __init__(self,dhmul,rho,wdbry=False,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.Df = tf.Variable(initial_value=dhmul.Df,trainable=True)
        self.rho = rho
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
        return tf.matmul(a=self.dhmul.Df,b=inputs)


class DhMul(tf.keras.layers.Layer):
    def __init__(self,Df,*args,dtype=tf.complex64,**kwargs):
        super().__init__(*args,**kwargs)
        with tf.name_scope(self.name):
            self.Dfprev = tf.Variable(initial_value=tf.identity(Df),trainable=False,dtype=dtype)
            self.Df = tf.Variable(initial_value=tf.identity(self.Dfprev),trainable=True)
        self.varname = self.Df.name
    def get_config(self):
        return {'varname': self.varname}

    def call(self, inputs):
        return tf.matmul(a=self.Df,b=inputs,adjoint_a=True)


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
        s, Uhat, V = tf.linalg.svd(B, full_matrices = False)

        if Uhat.dtype.is_complex and not Q.dtype.is_complex:
            U = tf.cast(Q,Uhat.dtype) @ Uhat[:, :n_components]
        else:
            U = Q @ Uhat[:, :n_components]
    else:
       s, U, V = tf.linalg.svd(A, full_matrices = False)

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
        return self.noc*Dtrunc/tf.math.sqrt(tf.reduce_sum(input_tensor=Dtrunc**2,axis=(1,2,3),keepdims=True))

def rank2eigen(U,V,epsilon=1e-5):
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
    def __init__(self,D,noc,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with tf.name_scope(self.name):
            self.D = tf.Variable(initial_value = D,trainable=False)
            self.R = tf.Variable(initial_value = computeR(D,noc),trainable=False)
    def call(self,inputs):
        R = tf.cast(tf.reshape(self.R,self.R.shape[:3] + (self.R.shape[4],self.R.shape[3],) + self.R.shape[5:]),dtype=self.dtype)
        return inputs/R
