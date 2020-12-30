import tensorflow as tf
import tensorflow_probability as tfp
import transforms as transf
import numpy as np
import post_process_grad as ppg

class dictionary_object2D(ppg.PostProcess):
    def __init__(self,fltrSz,fftSz,noc,nof,rho,epsilon=1e-6,*args,dtype=tf.complex64,**kwargs):
        self.dtype = dtype
        self.fftSz = fftSz
        self.noc = noc
        self.nof = nof
        self.fltrSz = fltrSz
        self.epsilon = epsilon
        self.rho = rho


        self.n_components=3

        Df = self.init_dict(fltrSz=fltrSz,fftSz=fftSz,noc=noc,nof=nof)

        self.dhmul = DhMul(Df,*args,dtype=self.dtype,**kwargs)
        self.dmul = DMul(self.dhmul,*args,dtype=self.dtype,**kwargs)
        self.qinv = QInv(self.dmul,self.dhmul,noc,nof,rho,*args,dtype=self.dtype,**kwargs)

        ppg.PostProcess.update[self.dhmul.varname] = self._dict_update
        
    def init_dict(self,fltrSz,fftSz,noc,nof):
        assert(tf.dtypes.as_dtype(self.dtype).is_complex)
        Drand = tf.random.normal(shape=(1,) + fltrSz + (noc,nof),dtype=tf.dtypes.as_dtype(self.dtype).real_dtype)
        Dmeaned = Drand - tf.math.reduce_mean(input_tensor=Drand,axis = (1,2),keepdims=True)
        Dnormalized = noc*Dmeaned/tf.math.sqrt(tf.reduce_sum(input_tensor=Dmeaned**2,axis=(1,2,3),keepdims=True))
        self.D = tf.Variable(initial_value=Dnormalized,trainable=False)
        return transf.fft2d_inner(fftSz)(self.D)       

    def Dmul(self,inputs):
        return self.dmul(inputs)

    def Dhmul(self,inputs):
        return self.dhmul(inputs)

    def Qinv(self,inputs):
        return self.qinv(inputs)

    def _dict_update(self):
        Dnew = transf.ifft2d_inner(self.fftSz)(self.dhmul.Df)
        # Truncate Dnew to match filter size.
        theUpdate = Dnew[slice(None),slice(0,self.fltrSz[0],1),slice(0,self.fltrSz[1],1),slice(None),slice(None)] - self.D

        U,V,approx = stack_svd(theUpdate,5)
        self.D.assign(self.D + approx)
        Uf = complexNum(U)
        Vf = transf.fft2d_inner(self.fftSz)(V)

        Df = self._update_decomposition(Uf,Vf)
        self.dhmul.Df = self.dhmul.Df.assign(Df)
        
    def _update_decomposition(self,U,V):

        # rank-1 Hermitian updates        
        if self.qinv.wdbry:
            # Redundant Computation: This is also computed in the rank-2 eigendecomposition.
            VhV = tf.math.reduce_sum(tf.math.conj(V)*V,axis=3,keepdims=False)
            for u,vhv in zip(tf.unstack(U,axis=-1),tf.unstack(VhV,axis=-1)):
                self.qinv.L = self.qinv.L.assign(tfp.math.cholesky_update(self.qinv.L,u,vhv))
        else:
            # Redundant Computation: This is also computed in the rank-2 eigendecomposition
            UhU = tf.math.reduce_sum(tf.math.conj(U)*U,axis=3,keepdims=False) # conjugate unnecessary: U is real.
            for v,uhu in zip(tf.unstack(V,axis=-1),tf.unstack(UhU,axis=-1)):
                self.qinv.L = self.qinv.L.assign(tfp.math.cholesky_update(self.qinv.L,v,uhu))

        # rank-2 Hermitian updates
        for u,v in zip(tf.unstack(U,axis=-1),tf.unstack(V,axis=-1)):
            if self.qinv.wdbry:
                asvec = tf.linalg.matvec(self.dhmul.Dfprev,v) #assymmetic vector
                eigvals,eigvecs = rank2eigen(u,asvec)
            else:
                asvec = tf.linalg.matvec(self.dhmul.Dfprev,u,adjoint_a=True)
                eigvals,eigvecs = rank2eigen(v,asvec) # assymmetric vector
            for ii in range(2):
                self.qinv.L = self.qinv.L.assign(tfp.math.cholesky_update(self.qinv.L,eigvecs[ii],eigvals[ii]))
            with tf.control_dependencies([asvec]):
                self.dhmul.Dfprev = self.dhmul.Dfprev.assign(self.dhmul.Dfprev + tf.reshape(u,u.shape + (1,))*tf.transpose(tf.reshape(v,v.shape + (1,)),perm=(0,1,2,4,3),conjugate=True))
        return self.dhmul.Dfprev

class dictionary_object2D_full(dictionary_object2D):
    def _dict_update(self):
        #D = transf.ifft2d_inner(self.fftSz)(self.dhmul.Df)
        #Dtrunc = D[slice(None),slice(0,self.fltrSz[0],1),slice(0,self.fltrSz[1],1),slice(None),slice(None)]
        #self.dhmul.Df = self.dhmul.Df.assign(transf.fft2d_inner(self.fftSz)(Dtrunc))
        if self.qinv.wdbry:
            idMat = tf.linalg.eye(num_rows = self.noc,batch_shape = (1,1,1),dtype=tf.dtypes.as_dtype(self.dtype))
            self.qinv.L = self.qinv.L.assign(tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b = self.dhmul.Df,adjoint_b = True)))
        else:
            idMat = tf.linalg.eye(num_rows = self.nof,batch_shape = (1,1,1),dtype=tf.dtypes.as_dtype(self.dtype))
            self.qinv.L = self.qinv.L.assign(tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b= self.dhmul.Df,adjoint_a = True)))

class symplified_dict_obj2D(dictionary_object2D):
    #def __init__(self,noc,nof,rho,*args,dtype=tf.float32,**kwargs):
    #    self.dtype = dtype
    #    D = self.init_dict(noc,nof)
    #    self.dhmul = DhMul(D,*args,dtype=self.dtype,**kwargs)
    #    self.dmul = DMul(self.dhmul,*args,dtype=self.dtype,**kwargs)
    #    self.qinv = QInv(self.dmul,self.dhmul,noc,nof,rho,*args,dtype=self.dtype,**kwargs)
     
    def init_dict(self,fltrSz=None,fftSz=None,noc=3,nof=16):
        self.D = tf.random.normal(shape=(1,1,1,noc,nof,),dtype=tf.dtypes.as_dtype(self.dtype))
        return self.D

    #def Dmul(self,inputs):
    #    return self.dmul(inputs)

    #def Dhmul(self,inputs):
    #    return self.dhmul(inputs)

    #def Qinv(self,inputs):
    #    return self.qinv(inputs)

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
            # I will need to make sure that TensorFlow does not attempt to differentiate self.L.  
            halfway = tf.linalg.triangular_solve(matrix=self.L,rhs=x,lower=True)
            output = tf.linalg.triangular_solve(matrix=self.L,rhs=halfway,lower=True,adjoint=True)

            @tf.custom_gradient
            def gradient_trick(y,Df):
                def grad(dg):
                    halfway = tf.linalg.triangular_solve(matrix=self.L,rhs=tf.math.conj(dg),lower=True)
                    ainvdg = tf.linalg.triangular_solve(matrix=self.L,rhs=halfway,lower=True,adjoint=True)
                    if self.wdbry:
                        Dhy = self.dhmul(y)
                        Dhy_T = tf.transpose(a=Dhy,perm=(0,1,2,4,3),conjugate=False)
                        Dhainvdg = self.dhmul(ainvdg)
                        dgainvD = tf.transpose(a=Dhainvdg,perm=(0,1,2,4,3),conjugate=True)
                        return (tf.identity(dg),tf.math.reduce_sum(input_tensor=tf.math.conj(ainvdg*Dhy_T) + y*dgainvD,axis=0,keepdims=True))
                    else:
                        y_T = tf.transpose(a=y,perm=(0,1,2,4,3),conjugate=False)
                        Dy = self.dmul(y)
                        Dainvdg = self.dmul(ainvdg)
                        dgainv = tf.transpose(a=ainvdg,perm=(0,1,2,4,3),conjugate=True)
                        return (tf.identity(dg),tf.math.reduce_sum(input_tensor=Dy*dgainv + tf.math.conj(y_T)*Dainvdg,axis=0,keepdims=True))
                return tf.identity(y),grad
            return gradient_trick(output,self.dhmul.Df)

        self.solve_inverse = lambda x: solve_inverse(x)

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
            self.L = tf.Variable(initial_value=tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b = self.dhmul.Df,adjoint_b = True)),trainable=False)
            #self.L = tf.stop_gradient(input=tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b = self.dhmul.Df,adjoint_b = True))
            self.wdbry = True
        else:
            idMat = tf.linalg.eye(num_rows = nof,batch_shape = (1,1,1),dtype=tf.dtypes.as_dtype(self.dtype))
            self.L = tf.Variable(initial_value = tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b= self.dhmul.Df,adjoint_a = True)),trainable=False)
            #self.L = tf.stop_gradient(input = tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b= self.dhmul.Df,adjoint_a = True))
            self.wdbry = False

class QInv_auto(tf.keras.layers.Layer):
    def __init__(self,dhmul,rho,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.Df = tf.Variable(initial_value=dhmul.Df,trainable=True)
        self.rho = rho
        #self.wdbry = self.Df.shape[-2] <= self.Df.shape[-1]
    def call(self,inputs):
        #if self.wdbry:
        #    Dx = tf.linalg.matmul(self.Df,inputs)
        #    idmat = tf.eye(num_rows = self.Df.shape[-2],batch_shape = (1,1,1),dtype = self.Df.dtype)
        #    ainv = tf.linalg.inv(self.rho*idmat + tf.linalg.matmul(self.Df,self.Df,adjoint_b=True))
        #    return 1/self.rho*(inputs - tf.linalg.matmul(self.Df,tf.linalg.matmul(ainv,Dx),adjoint_a = True))
        #else:
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
        self.Dfprev = tf.Variable(initial_value=tf.identity(Df),trainable=False,dtype=dtype)
        self.Df = tf.Variable(initial_value=tf.identity(self.Dfprev),trainable=True)
        self.varname = self.Df.name

    def call(self, inputs):
        return tf.matmul(a=self.Df,b=inputs,adjoint_a=True)

def get_lowrank_approx(A,n_components=3,n_oversamples=10,n_iter='auto',powerIterationNormalizer='auto',transpose='auto'):
    U,s,V = randomized_svd(A,n_components,n_oversamples,n_iter,powerIterationNormalizer,transpose)
    if A.shape[1] > A.shape[0]:
        U = U*tf.cast(tf.reshape(s,(1,-1)),U.dtype)
    else:
        V = V*tf.cast(tf.reshape(s,(1,-1)),V.dtype)
    return (U,V, tf.linalg.matmul(U,tf.transpose(V,conjugate=True)))



# This function has not been tested.
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
            Q, _ = tf.linalg.lu(tf.matmul(AQ,AQ,adjoint_a = True))
            Q2 = Q2 @ Q
        elif power_iteration_normalizer == 'QR':
            Q, _ = tf.linalg.qr(A @ Q2)
            Q2, _ = tf.linalg.qr(tf.linalg.matmul(A, Q,adjoint_a=True))

    # Compute an orthonormal basis for a linear projection of A
    Q, _ = tf.linalg.qr(A @ Q2)
    return Q

def complexNum(x):
    return tf.complex(x,tf.cast(0.0,dtype = x.dtype))

def rank2eigen(U,V,epsilon=1e-5):
    vhv = tf.reduce_sum(tf.math.conj(V)*V,axis=-1,keepdims=True)
    uhu = tf.reduce_sum(tf.math.conj(U)*U,axis=-1,keepdims=True)
    uhv = tf.reduce_sum(tf.math.conj(U)*V,axis=-1,keepdims=True)
    rootRadicand = tf.math.sqrt(vhv*uhu - complexNum(tf.math.imag(uhv)**2))
        
    valPlus = complexNum(tf.math.real(uhv)) + rootRadicand
    valMinus = complexNum(tf.math.real(uhv)) - rootRadicand
    vecPlus = vhv*U + (1j*complexNum(tf.math.imag(uhv)) + rootRadicand)*V
    vecMinus = vhv*U + (1j*complexNum(tf.math.imag(uhv))  - rootRadicand)*V
    vecPlus = tf.where(tf.abs(rootRadicand) > epsilon,vecPlus,U)
    vecMinus = tf.where(tf.abs(rootRadicand) > epsilon,vecMinus,-tf.math.divide_no_nan(uhv,uhu)*U + V)
    #tf.debugging.Assert(condition=tf.abs(rootRadicands) > 1e-5,data=rootRadicands)
        
    vecPlus = tf.math.l2_normalize(vecPlus,epsilon=epsilon)
    vecMinus = tf.math.l2_normalize(vecMinus,epsilon=epsilon)
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

