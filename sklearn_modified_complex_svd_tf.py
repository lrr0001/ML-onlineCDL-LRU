import tensorflow as tf

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

