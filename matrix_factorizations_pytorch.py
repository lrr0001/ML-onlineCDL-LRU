import sporco.linalg
import numpy
import numpy.linalg
import scipy.linalg
import sklearn_modified_complex_svd
import math



class factoredMatrix:
    """ Currently, I don't see what kind of functions would be necessary for all matrix decompositions, so I'll leave this empty for now.
    """
    pass

class factoredMatrix_aIpBhB(factoredMatrix):
    """
    This is a class of decompositions for matrices of the form aI + B^H B. Rank-1 B updates can be converted to 3 symmetric rank-1 updates for the decomposition.

    """
    def sym_update(self,x,sign):
        raise NotImplementedError

    def inv_mat(self,b,D):
        raise NotImplementedError

    def update(self,u,v,D):

        lamb1,lamb2,lamb3,x1,x2,x3 = self.get_update_vectors(u,v,D)
        # eigenvalues can be negative, so the order here matters. Third eigenvalue is guarenteed to be positive.
        self.sym_update(numpy.sqrt(lamb3)*x3,sign=1)
        self.sym_update(numpy.sqrt(lamb1)*x1,sign=numpy.sign(lamb1))
        self.sym_update(numpy.sqrt(lamb2)*x2,sign=numpy.sign(lamb2))

    def get_update_vectors(self,u,v,D): # requires that self.flipped is defined.
        """
        get_update_vectors(self,u,v,D)
          Converts update to symmetric rank-1 updates.
          Dnew = D + addDim(u)*np.swapaxes(np.conj(addDim(v)),-1,-2)
          ***Inputs***
            u shape:  (...,M)
            v shape:  (...,N)
            D shape:  (...,M,N)

          ***Outputs***
          (eigval1,eigval2,eigval3,eigvec1,eigvec2,eigvec3)
            eigval3 is positive semidefinite
            eigval1, eigval2, eigval3 shape: (...,1)
            eigvec1, eigvec2, eigvec3 shape: either (...,M) if flipped or (..., N) if not flipped
        """
        baseShape = D.shape[0:-2]
        flatShape = (1,)*len(baseShape)
        uhu = vec_sqrnorm(u)
        vhv = vec_sqrnorm(v)
        e1 = numpy.concatenate((numpy.ones(flatShape + (1,)),numpy.zeros(flatShape + (1,))),-1)
        e1 = addDim(e1)
        e2 = numpy.concatenate((numpy.zeros(flatShape + (1,)),numpy.ones(flatShape + (1,))),-1)
        e2 = addDim(e2)

        if self.flipped is True:
            # I think there is a numpy function that does this already
            u_broadcast = numpy.empty(baseShape + (u.shape[-1],) + (1,),dtype=u.dtype) # allows broadcasting for u
            u_broadcast[:] = addDim(u)
            dv = numpy.matmul(D,addDim(v))
            B = numpy.concatenate((dv,u_broadcast),axis=-1)
            A = conj_tp(numpy.concatenate((u_broadcast,dv),axis=-1))
            eigval3 = vhv # technically not an eigenvalue unless u is unit length.
            x3 = u
        else:
            v_broadcast = numpy.empty(baseShape + (v.shape[-1],) + (1,),dtype=v.dtype) # allows broadcasting for v
            v_broadcast[:] = addDim(v)
            dhu = numpy.matmul(conj_tp(D),addDim(u))
            B = numpy.concatenate((dhu,v_broadcast),axis=-1)
            A = conj_tp(numpy.concatenate((v_broadcast,dhu),axis=-1))
            eigval3 = uhu # technically not an eigenvalue unless v is unit length.
            x3 = v

        # Need eigenvalues of BA, identical to eigenvalues of AB:
        AB = numpy.matmul(A,B)
        a = numpy.matmul(conj_tp(e1),numpy.matmul(AB,e1))
        a = minusDim(a)
        b = numpy.matmul(conj_tp(e1),numpy.matmul(AB,e2))
        b = minusDim(b)
        c = numpy.matmul(conj_tp(e2),numpy.matmul(AB,e1))
        c = minusDim(c)
        d = numpy.matmul(conj_tp(e2),numpy.matmul(AB,e2))
        d = minusDim(d)
        eigval1, eigval2,seigvec1,seigvec2 = eig2by2(a,b,c,d)


        # If the eigenvalues are equal, any basis will serve for eigenvectors. Euclidean is good default choice.
        seigvec1[numpy.concatenate((numpy.abs(eigval1 - eigval2) < 1e-10,numpy.zeros(eigval1.shape,dtype=bool)),-1)] = 1
        seigvec2[numpy.concatenate((numpy.abs(eigval1 - eigval2) < 1e-10,numpy.zeros(eigval1.shape,dtype=bool)),-1)] = 0
        seigvec1[numpy.concatenate((numpy.zeros(eigval1.shape,dtype=bool),numpy.abs(eigval1 - eigval2) < 1e-10),-1)] = 0
        seigvec2[numpy.concatenate((numpy.zeros(eigval1.shape,dtype=bool),numpy.abs(eigval1 - eigval2) < 1e-10),-1)] = 1



        # Convert eigenvectors of AB to eigenvectors of BA:
        eigvec1 = numpy.matmul(B,addDim(seigvec1))
        eigvec1 = minusDim(eigvec1)
        mag1 = numpy.sqrt(vec_sqrnorm(eigvec1))
        mag1[numpy.abs(mag1)< 1e-10] = 1e10        
        eigvec2 = numpy.matmul(B,addDim(seigvec2))
        eigvec2 = minusDim(eigvec2)
        mag2 = numpy.sqrt(vec_sqrnorm(eigvec2))
        mag2[numpy.abs(mag2) < 1e-10] = 1e10
        x1 = eigvec1/mag1
        x2 = eigvec2/mag2
        return(eigval1,eigval2,eigval3,x1,x2,x3)

    def inv_check_ls(self,D):
        """ Checks the left-side inverse property of object's mat_inv function.

        """
        M = D.shape[-2]
        N = D.shape[-1]
        matId = numpy.zeros(D.shape[0:-2] + (N,N,))
        for ii in range(N):
            s = [slice(0,D.shape[jj]) for jj in range(len(D.shape[0:-2]))] + [slice(ii,ii + 1),slice(ii,ii + 1)]
            matId[tuple(s)] = 1
        aIpDhD = self.rho*matId + numpy.matmul(conj_tp(D),D)
        return numpy.amax(numpy.abs(matId - self.inv_mat(aIpDhD,D)))



class factoredMatrix_chol(factoredMatrix_aIpBhB):
    def __init__(self,D=None,dtype=None,rho=1):
        if D is None:
            raise NotImplementedError('Cholesky factorization currently requires input dictionary.')
        
        self.dtype=dtype
        self.rho = rho
        if isinstance(D,list):
            self.list_based_init(D,dtype,rho)
            return None

        m = D.shape[-2]
        n = D.shape[-1]
        if dtype is None:
            dtype= D.dtype


        if m <= n: # (flipped) Woodbury formulation may be more efficient.
            idMat = numpy.identity(m,dtype=self.dtype)
            idMat = numpy.reshape(idMat,(1,)*len(D.shape[:-2]) + (m,m,))
            self.L = numpy.linalg.cholesky(rho*idMat + numpy.matmul(D,conj_tp(D)))
            self.flipped=True
        else:
            idMat = numpy.identity(n,dtype= self.dtype)
            idMat = numpy.reshape(idMat,(1,)*len(D.shape[:-2]) + (n,n,))
            self.L = numpy.linalg.cholesky(rho*idMat + numpy.matmul(conj_tp(D),D))
            self.flipped=False

    def list_based_init(self,D,dtype,rho):
        """Here, D is a list of rank one updates to scaled identity.

        The shape is used to communicate whether or not the factorization is flipped:

        D[0].shape = (N_1,N_2,...,1,N) factors rho*I + DD^H, flipped = False
        D[0].shape = (N_1,N_2,...,M,1) factors rho*I + D^HD, flipped = True
          """
        assert(len(D) > 0)
        if dtype is None:
            self.dtype = D[0].dtype
        if D[0].shape[-1] == 1:
            self.flipped = True
            m = D[0].shape[-2]
            idMat = numpy.identity(m,self.dtype)
            self.L = numpy.sqrt(rho)*numpy.ones(D[0].shape[:-2] + (m,m,))*idMat.reshape(len(D[0].shape[:-2])*(1,) + (m,m,))
        else:
            self.flipped = False
            n = D[0].shape[-1]
            idMat = numpy.identity(n,self.dtype)
            self.L = numpy.sqrt(rho)*numpy.ones(D[0].shape[:-2] + (n,n,))*idMat.reshape(len(D[0].shape[:-2])*(1,) + (n,n,))

        for ii in range(len(D)):
            if self.flipped:
                self.sym_update(x = minusDim(D[ii]),sign=1)
            else:
                self.sym_update(x = minusDim(conj_tp(D[ii])),sign=1)
        

    def sym_update(self,x,sign=1):
        self.L = cholesky_rank1_update(self.L,x,sign)

    def inv_vec(self, b, D=None):
        if self.flipped:
            assert D is not None
            y = numpy.matmul(D,addDim(b))
            y = minusDim(y)
            z = numpy.zeros(y.shape,dtype=self.dtype)


            w = solve_triangular(self.L,y,lower=True)
            z = solve_triangular(self.L,w,lower=True,trans=True)
            z = minusDim(z)
            #for inds in loop_magic(self.L.shape[0:-2]): # Loop slows things down, but I don't have vectorized triangular solver
            #    w = scipy.linalg.solve_triangular(self.L[inds],y[inds],lower=True,check_finite=False)
            #    z[inds] = scipy.linalg.solve_triangular(self.L[inds],w,lower=True,trans=2,overwrite_b=True,check_finite=False)
            return (b - (numpy.matmul(conj_tp(D),z.reshape(z.shape + (1,)))).reshape(b.shape))/self.rho


        else:
            y = solve_triangular(self.L,b,lower=True)
            z = solve_triangular(self.L,y,lower=True,trans=True)
            #z = numpy.zeros(b.shape,dtype=self.dtype)
            #for inds in loop_magic(self.L.shape[0:-2]): # Loop slows things down, but I don't have vectorized triangular solver
            #    y = scipy.linalg.solve_triangular(self.L[inds],b[inds],lower=True,check_finite=False)
            #    z[inds] = scipy.linalg.solve_triangular(self.L[inds],y,lower=True,trans=2,overwrite_b=True,check_finite=False)
            return minusDim(z)

    def inv_mat(self,b,D=None):
        if self.flipped:
            assert D is not None
            y = numpy.matmul(D,b)
            w = solve_triangular(self.L,y,lower=True)
            z = solve_triangular(self.L,w,lower=True,trans=True)
            #z = numpy.zeros(y.shape,dtype=self.dtype)
            #for inds in loop_magic(self.L.shape[0:-2]):
            #    w = scipy.linalg.solve_triangular(self.L[inds],y[inds],lower=True,check_finite=False)
            #    z[inds] = scipy.linalg.solve_triangular(self.L[inds],w,lower=True,trans=2,overwrite_b=True,check_finite=False)
            return (b - numpy.matmul(conj_tp(D),z))/self.rho
        else:
            y = solve_triangular(self.L,b,lower=True)
            z = solve_triangular(self.L,y,lower=True,trans=True)
            #z = numpy.zeros(b.shape,dtype=self.dtype)
            #for inds in loop_magic(self.L.shape[0:-2]):
            #    y = scipy.linalg.solve_triangular(self.L[inds],b[inds],lower=True,check_finite=False)
            #    z[inds] = scipy.linalg.solve_triangular(self.L[inds],y,lower=True,trans=2,overwrite_b=True,check_finite=False)
            return z
