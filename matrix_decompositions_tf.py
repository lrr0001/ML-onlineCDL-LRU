import tensorflow as tf
import transforms as transf

class dictionary_object2D:
    def __init__(self,fltrSz,fftSz,noc,nof,rho,*args,dtype=tf.complex64,**kwargs):
        self.dtype = dtype
        self.fftSz = fftSz
        self.noc = noc
        self.nof = nof
        self.fltrSz = fltrSz
        Df = self.init_dict(fltrSz,fftSz,noc,nof)
        self.dhmul = DhMul(Df,*args,dtype=self.dtype,**kwargs)
        self.dmul = DMul(self.dhmul,*args,dtype=self.dtype,**kwargs)
        self.qinv = QInv(self.dmul,self.dhmul,noc,nof,rho,*args,dtype=self.dtype,**kwargs)
        
    def init_dict(self,fltrSz,fftSz,noc,nof):
        assert(tf.dtypes.as_dtype(self.dtype).is_complex)
        Drand = tf.random.normal(shape=(1,) + fltrSz + (noc,nof),dtype=tf.dtypes.as_dtype(self.dtype).real_dtype)
        Dmeaned = Drand - tf.math.reduce_mean(input_tensor=Drand,axis = (1,2),keepdims=True)
        self.D = fltrSz*Dmeaned/tf.math.sqrt(tf.reduce_sum(input_tensor=Dmeaned**2,axis=(1,2,3),keepdims=True))
        return transf.fft2d_inner(fftSz)(self.D)       

    def Dmul(self,inputs):
        return self.dmul(inputs)

    def Dhmul(self,inputs):
        return self.dhmul(inputs)

    def Qinv(self,inputs):
        return self.qinv(inputs)

    def _update_dict(self):
        Dnew = transf.ifft2d_inner(self.fftSz)(self.dhmul.Df)
        theUpdate = Dnew - self.D
        theUpdate = tf.transpose(theUpdate)
        # these are flipped on purpose
        V,U,approx = get_lowrank_approx(theUpdate.reshape(shape=(prod(self.fltrSz)*nof,self.noc)):
        approx = tf.transpose(approx.reshape(shape=((1,) + fltrSz + (nof,noc,))))
        self.D.assign(self.D + approx)
        U = tf.transpose(U.reshape((1,1,1,self.n_components,self.noc))
        V = tf.math.conj(tf.transpose(transf.fft2d_inner(self.fftSz)(V.reshape((1,) + fltrSz + (nof,n_components,)))))
        UhU = tf.math.reduce_sum(tf.math.conj(U)*U,axes=3,keepdims=True) 
        VhV = tf.math.reduce_sum(tf.math.conj(V)*V,axes=3,keepdims=True)
        if self.qinv.wdbry:
            for u,vhv in zip(tf.unstack(U),tf.unstack(VhV)):
                symmUp = tfp.math.cholesky_update(self.qinv.L,u,vhv)
            Df = self._sym_rank2_updates_wdbry(U,V,UhU,symmUp)
        else:
            for v,uhu in zip(unstack(V),unstack(UhU)):
                symmUp = tfp.math.cholesky_update(self.qinv.L,v,uhu)
            Df = self._symm_rank2_updates(uf,Vf,VhV,symmUp)

        self.dhmul.Df.assign(Df)

    def _symm_rank2_updates(self,Uf,Vf,VhV,symmUp):
        dfprev = tf.identity(self.dhmul.Dfprev)
        for u,v,vhv in zip(tf.unstack(Uf),tf.unstack(Vf),tf.unstack(VhV)):
            dhu = tf.linalg.vecmul(dfprev,u,adjoint_a=True)
            vhdhu = tf.math.reduce_sum(tf.math.conj(v),dhu,axis=3,keepdims=True)
            uhddhu = tf.math.reduce_sum(tf.math.conj(dhu),dhu,axis=3,keepdims=True)

            # Compute eigenvalues
            rootRadicand = tf.math.sqrt(vhv*uhddhu - tf.complex(tf.math.imag(vhdhu)**2,0))
            eigvalPlus = tf.complex(tf.math.real(vhdhu),0) + rootRadicand
            eigvalMinus = tf.complex(tf.math.real(vhdhu),0) - rootRadicand

            # Compute eigenvectors
            eigvecPlus = uhddhu*v + (j1*tf.complex(tf.math.imag(vhdhu),0) + rootRadicand)*dhu
            eigvecMinus = uhddhu*v + (j1*tf.complex(tf.math.imag(vhdhu),0) - rootRadicand)*dhu

            # If the two eigenvalues are not distinct, eigenvectors must be chosen to be orthogonal
            eigvecPlus = tf.where(tf.math.abs(rootRadicand) > tol,eigvecPlus,v)
            eigvecMinus = tf.where(tf.math.abs(rootRadicand) > tol,eigvecMinus,-tf.math.divide_no_nan(vhdhu,vhv)*v + dhu)

            # Normalize eigenvectors
            eigvecPlus = tf.l2_normalize(eigvecPlus,axis=3,epsilon=tol)
            eigvecMinus = tf.l2_normalize(eigvecMinus,axis=3,epsilon=tol)

            # Update Cholesky decomposition
            with tf.control_dependencies([symmUp]):
                tfp.math.cholesky_update(self.qinv.L,eigvecPlus,eigvalPlus)
                tfp.math.cholesky_update(self.qinv.L,eigvecMinus,eigvalMinus)

            # Update dictionary (after dhu is computed)
            with tf.control_dependencies([dhu]):
                dfprev = self.dhmul.Dfprev.assign(self.dhmul.Dfprev + u*tf.transpose(v.reshape(v.shape + (1,),conjugate=True))
        return dfprev

    def _symm_rank2_updates_wdbry(self,Uf,Vf,UhU):
        dfprev = tf.identity(self.dhmul.Dfprev)
        for u,v,uhu in zip(tf.unstack(Uf),tf.unstack(Vf),tf.unstack(UhU)):
            dv = tf.linalg.vecmul(dfprev,v)
            uhdv = tf.math.reduce_sum(tf.math.conj(u),dv,axis=3,keepdims=True)
            vhdhdv = tf.math.reduce_sum(tf.math.conj(dv),dv,axis=3,keepdims=True)

            # Compute eigenvalues
            rootRadicand = tf.math.sqrt(uhu*vhdhdv - tf.complex(tf.math.imag(uhdv)**2,0))
            eigvalPlus = tf.complex(tf.math.real(uhdv),0) + rootRadicand
            eigvalMinus = tf.complex(tf.math.real(uhdv),0) - rootRadicand

            # Compute eigenvectors
            eigvecPlus = vhdhdv*u + (j1*tf.complex(tf.math.imag(uhdv),0) + rootRadicand)*dv
            eigvecMinus = uhddhu*v + (j1*tf.complex(tf.math.imag(uhdv),0) - rootRadicand)*dv

            # If the two eigenvalues are not distinct, eigenvectors should be chosen to be orthogonal
            eigvecPlus = tf.where(tf.math.abs(rootRadicand) > tol,eigvecPlus,u)
            eigvecMinus = tf.where(tf.math.abs(rootRadicand) > tol,eigvecMinus,-tf.math.divide_no_nan(uhdv,uhu)*u + dv)

            # Normalize eigenvectors
            eigvecPlus = tf.l2_normalize(eigvecPlus,axis=3,epsilon=tol)
            eigvecMinus = tf.l2_normalize(eigvecMinus,axis=3,epsilon=tol)

            # Update Cholesky decomposition
            tfp.math.cholesky_update(self.qinv.L,eigvecPlus,eigvalPlus)
            tfp.math.cholesky_update(self.qinv.L,eigvecMinus,eigvalMinus)

            # Update dictionary (after dhu is computed)
            with tf.control_dependencies([dhu]):
                dfprev = self.dhmul.Dfprev.assign(self.dhmul.Dfprev + u*tf.transpose(v.reshape(v.shape + (1,),conjugate=True))
        return dfprev



    def _DupdateFun(Dupdate,M,n_components,n_oversamples,n_iter,power_iteration_normalizer,transpose):
        Dutransposed = tf.transpose(Dupdate,perm=(0,1,2,4,3))
        returnShape = Dutransposed.shape
        Dureshaped = tf.reshape(Dutransposed,shape = (-1,C))
        # Not sure what module this will end up in. Seems like a decomposition
        # Might switch order of outputs to match Tensorflow's SVD
        Us,s,Vs = randomized_svd(Dureshaped,n_components,n_oversamples,n_iter,power_iteration_normalizer,transpose)
        # Need to assess which is more efficient
        if thisIsMoreEfficient:
            Us = tf.math.mathvec(Us,s)
        else:
            Vs = tf.transpose(tf.math.mathvec(Vs,s,adjoint_a=True),conjugate=True)
        # Reshape U and V
        V = tf.transpose(Us.reshape(shape=returnShape[:-1] + (n_components,)),adjoint = True)
        U = tf.transpose(Vs.reshape(shape=(1,1,1,n_components,C)),adjoint = False)
        # Convert both U and V to frequency domain
        Uf = U # (1,1,1,C,n_components)
        Vf = transf.switch_spatial_and_channel_fltr(transf.fft(transf.switch_spatial_and_channelfltr(V))) # (1,fft0,fft1,M,n_components)
        UhU = tf.math.reduce_sum(tf.math.conj(Uf)*Uf,axes=3, keepdims= True)
        VhV = tf.math.reduce_sum(tf.math.conj(Vf)*Vf,axes=3,keepdims=True)
        DhU = tf.linalg.matmul(Dfprev,Uf,adjoint_a=True) # (1,1,1,M,n_components)
        UhDV = tf.math.reduce_sum(tf.math.conj(DhU)*Vf,axes=3,keepdims=True)
        DV = tf.linalg.matmul(Dfprev,V) # (1,fft0,fft1,C,n_components)
        VhDhU = tf.math.reduce_sum(tf.math.conj(DV)*Uf,axes=3,keepdims=True)
        radicand = tf.math.sqrt(VhV*UhDDhU - tf.math.imag(UhDV)**2)
        eigvalsPlus = tf.math.real(UhDV) + radicand
        eigvalsMinus = tf.math.real(UhDV) - radicand
        eigvecsPlus = tf.where(tf.abs(radicand) > tol,UhDdhu*Vf + (-1j*tf.math.imag(UhDV)+radicand)*DhU,Vf)
        eigvecsMinus = tf.where(tf.abs(radicand) > tol,UhDdhu*Vf + (-1j*tf.math.imag(UhDV)-radicand)*DhU,-tf.math.divide_no_nan(VhDhU,VhV)*Vf + DhU)
        eigvecsPlus = tf.l2_normalize(eigvecsPlus,axis=3,epsilon=tol)
        eigvecsMinus = tf.l2_normalize(eigvecsMinus,axis=3,epsilon=tol)
        
        # need to use split to break up updates
        # need to use eigs to update cholesky decomposition.


class symplified_dict_obj2D(dictionary_object2D):
    #def __init__(self,noc,nof,rho,*args,dtype=tf.float32,**kwargs):
    #    self.dtype = dtype
    #    D = self.init_dict(noc,nof)
    #    self.dhmul = DhMul(D,*args,dtype=self.dtype,**kwargs)
    #    self.dmul = DMul(self.dhmul,*args,dtype=self.dtype,**kwargs)
    #    self.qinv = QInv(self.dmul,self.dhmul,noc,nof,rho,*args,dtype=self.dtype,**kwargs)
     
    def init_dict(fltrSz=None,fftSz=None,noc,nof):
        self.D = tf.random.normal(shape=(1,1,1,noc,nof,),dtype=tf.dtypes,as_dtype(self.dtype))
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
                    y_T = tf.transpose(a=y,perm=(0,1,2,4,3),conjugate=False)
                    Dy = self.dmul(y)
                    halfway = tf.linalg.triangular_solve(matrix=self.L,rhs=tf.math.conj(dg),lower=True)
                    ainvdg = tf.linalg.triangular_solve(matrix=self.L,rhs=halfway,lower=True,adjoint=True)
                    Dainvdg = self.dmul(ainvdg)
                    ainvdg_T = tf.transpose(a=ainvdg,perm=(0,1,2,4,3),conjugate=True)
                    return (tf.identity(dg),Dy*ainvdg_T + y_t*tf.math.conj(Dainvdg))
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
            self.L = tf.variable(initial_value=tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b = self.dhmul.Df,adjoint_b = True),trainable=False)
            #self.L = tf.stop_gradient(input=tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b = self.dhmul.Df,adjoint_b = True))
            self.wdbry = True
        else:
            idMat = tf.linalg.eye(num_rows = nof,batch_shape = (1,1,1),dtype=tf.dtypes.as_dtype(self.dtype))
            self.L = tf.variable(initial_value = tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b= self.dhmul.Df,adjoint_a = True),trainable=False)
            #self.L = tf.stop_gradient(input = tf.linalg.cholesky(self.rho*idMat + tf.linalg.matmul(a = self.dhmul.Df,b= self.dhmul.Df,adjoint_a = True))
            self.wdbry = False


    
        

class DMul(tf.keras.layers.Layer):
    def __init__(self,dhmul,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dhmul = dhmul

    def call(self, inputs):
        return tf.matmul(a=self.dhmul.Df,b=inputs)


class DhMul(tf.keras.layers.Layer):
    def __init__(self,Df,*args,dtype=tf.complex64,**kwargs):
        super().__init__(*args,**kwargs)
        self.Dfprev = tf.Variable(initial_value=Df,trainable=False,dtype=dtype)
        self.Df = tf.Variable(initial_value=tf.identity(self.Dfprev),trainable=True)
        self.varname = self.Df.name

    def call(self, inputs):
        return tf.matmul(a=self.Df,b=inputs,adjoint_a=True)

def eig2x2(a,b,c,d):
    apd = (a + d)/2
    amd = (a - d)/2
    bc = b*c
    radicand = tf.math.sqrt(amd + bc)
    eig1 = apd + radicand
    eig2 = apd - radicand

def eig2x2_conja(a,b,c,axis=0:
    # preliminary computations
    apd = tf.math.real(a)
    amd = tf.math.imag(a)
    bc = b*c
    radicand = tf.math.sqrt(tf.complex(bc - amd**2,.0))

    # compute eigenvalues
    val1 = tf.complex(apd,0.) + radicand
    val2 = tf.complex(apd,0.) - radicand

    # compute eigenvectors
    temp1 =radicand - 1j*amd

    # tf.concat does not broadcast

    # Can't normalize zeros, so this check is neccessary. 
    if tf.math.abs(temp1) < tol and tf.math.abs(b) < tol:
        vec1 = tf.concat((tf.ones(b.shape),tf.zeros(temp1.shape),axis=axis)
        vec2 = tf.concat((tf.zeros(b.shape),tf.ones(temp1.shape),axis=axis)
    else:
        vec1 = tf.concat((b,temp1),axis=axis)
        temp1 = -tf.math.conj(temp1)
        vec2 = tf.concat((b,temp1),axis=axis)
        # normalize the vectors
        vec1 = tf.math.l2_normalize(vec1,axis)
        vec2 = tf.math.l2_normalize(vec2,axis)
    return ((val1,val2),(vec1,vec2))

def get_lowrank_approx(A,n_components,n_oversamples,n_iter,powerIterationNormalizer,transpose):
    U,s,V = randomized_svd(A,n_components,n_oversamples,n_iter,powerIterationsNormalizer,transpose)
    if transpose:
        U = tf.math.mathvec(U,s)
    else:
        V = tf.trasnpose(tf.linalg.mathvec(V,s,adjoint_a=True),conjugate=True)
    return (U,V, tf.linalg.matmul(U,V))



# This function has not been tested.
def randomized_svd(A, n_components, n_oversamples=10, n_iter='auto',
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
        n_iter = 7 if n_components < .1 * min(M.shape) else 4

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
        U = Q @ Uhat[:, :n_components]
    else:
       s, U, V = tf.linalg.svd(A, full_matrices = False)

    if transpose:
        # transpose back the results according to the input convention
        return tf.transpose(V[:n_components,:]), s[:n_components], tf.transpose(U[:,:n_components])
    else:
        return U[:,:n_components], s[:n_components], V[:n_components,:]

def randomized_range_finder(A, rangeSize, n_iter,
                            power_iteration_normalizer='auto',dtype= tf.float32):
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

