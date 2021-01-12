import tensorflow as tf
import util

def broadcast_shape(x_shape, y_shape):
  """Computes the shape of a broadcast.
  When both arguments are statically-known, the broadcasted shape will be
  computed statically and returned as a `TensorShape`.  Otherwise, a rank-1
  `Tensor` will be returned.
  Args:
    x_shape: A `TensorShape` or rank-1 integer `Tensor`.  The input `Tensor` is
      broadcast against this shape.
    y_shape: A `TensorShape` or rank-1 integer `Tensor`.  The input `Tensor` is
      broadcast against this shape.
  Returns:
    shape: A `TensorShape` or rank-1 integer `Tensor` representing the
      broadcasted shape.
  """
  x_shape_static = tf.get_static_value(x_shape)
  y_shape_static = tf.get_static_value(y_shape)
  if (x_shape_static is None) or (y_shape_static is None):
    return tf.broadcast_dynamic_shape(x_shape, y_shape)

  return tf.broadcast_static_shape(
      tf.TensorShape(x_shape_static), tf.TensorShape(y_shape_static))



def cholesky_update(chol, update_vector, multiplier=1., name=None):
  """Returns cholesky of chol @ chol.T + multiplier * u @ u.T.
  Given a (batch of) lower triangular cholesky factor(s) `chol`, along with a
  (batch of) vector(s) `update_vector`, compute the lower triangular cholesky
  factor of the rank-1 update `chol @ chol.T + multiplier * u @ u.T`, where
  `multiplier` is a (batch of) scalar(s).
  If `chol` has shape `[L, L]`, this has complexity `O(L^2)` compared to the
  naive algorithm which has complexity `O(L^3)`.
  Args:
    chol: Floating-point `Tensor` with shape `[B1, ..., Bn, L, L]`.
      Cholesky decomposition of `mat = chol @ chol.T`. Batch dimensions
      must be broadcastable with `update_vector` and `multiplier`.
    update_vector: Floating-point `Tensor` with shape `[B1, ... Bn, L]`. Vector
      defining rank-one update. Batch dimensions must be broadcastable with
      `chol` and `multiplier`.
    multiplier: Floating-point `Tensor` with shape `[B1, ..., Bn]. Scalar
      multiplier to rank-one update. Batch dimensions must be broadcastable
      with `chol` and `update_vector`. Note that updates where `multiplier` is
      positive are numerically stable, while when `multiplier` is negative
      (downdating), the update will only work if the new resulting matrix is
      still positive definite.
    name: Optional name for this op.
  #### References
  [1] Oswin Krause. Christian Igel. A More Efficient Rank-one Covariance
      Matrix Update for Evolution Strategies. 2015 ACM Conference.
      https://www.researchgate.net/publication/300581419_A_More_Efficient_Rank-one_Covariance_Matrix_Update_for_Evolution_Strategies
  """
  # TODO(b/154638092): Move this functionality in to TensorFlow.
  with tf.name_scope(name or 'cholesky_update'):
    dtype = chol.dtype
    chol = tf.convert_to_tensor(chol, name='chol', dtype=dtype)
    update_vector = tf.convert_to_tensor(
        update_vector, name='update_vector', dtype=dtype)
    multiplier = tf.convert_to_tensor(
        multiplier, name='multiplier', dtype=dtype)

    batch_shape = broadcast_shape(broadcast_shape(tf.shape(chol)[:-2],
            tf.shape(update_vector)[:-1]), tf.shape(multiplier))
    chol = tf.broadcast_to(
        chol, tf.concat(
            [batch_shape, tf.shape(chol)[-2:]], axis=0))
    update_vector = tf.broadcast_to(
        update_vector, tf.concat(
            [batch_shape, tf.shape(update_vector)[-1:]], axis=0))
    multiplier = tf.broadcast_to(multiplier, batch_shape)

    chol_diag = tf.linalg.diag_part(chol)

    # The algorithm in [1] is implemented as a double for loop. We can treat
    # the inner loop in Algorithm 3.1 as a vector operation, and thus the
    # whole algorithm as a single for loop, and hence can use a `tf.scan`
    # on it.

    # We use for accumulation omega and b as defined in Algorithm 3.1, since
    # these are updated per iteration.

    def compute_new_column(accumulated_quantities, state):
      """Computes the next column of the updated cholesky."""
      _, _, omega, b = accumulated_quantities
      index, diagonal_member, col = state
      omega_at_index = tf.gather(omega, index, axis=-1)

      # Line 4
      new_diagonal_member = tf.math.sqrt(
          tf.math.square(diagonal_member) + multiplier / b * util.cmplx_sqr(
              omega_at_index))
      # `scaling_factor` is the same as `gamma` on Line 5.
      scaling_factor = (tf.math.square(diagonal_member) * b +
                        multiplier * util.cmplx_sqr(omega_at_index))

      # The following updates are the same as the for loop in lines 6-8.
      omega = omega - (omega_at_index / diagonal_member)[..., tf.newaxis] * col
      new_col = new_diagonal_member[..., tf.newaxis]  * (
          col / diagonal_member[..., tf.newaxis] +
          (multiplier * tf.math.conj(omega_at_index) / scaling_factor)[
              ..., tf.newaxis] * omega)
      b = b + multiplier * util.cmplx_sqr(omega_at_index / diagonal_member)
      return new_diagonal_member, new_col, omega, b

    # We will scan over the columns.
    chol = util.rotate_dims_right(chol)
    chol_diag = util.rotate_dims_right(chol_diag)
    new_diag, new_chol, _, _ = tf.scan(
        fn=compute_new_column,
        elems=(tf.range(0, tf.shape(chol)[0]), chol_diag, chol),
        initializer=(
            tf.zeros_like(multiplier),
            tf.zeros_like(chol[0, ...]),
            update_vector,
            tf.ones_like(multiplier)))
    new_chol = util.rotate_dims_left(new_chol)
    new_diag = util.rotate_dims_left(new_diag)
    new_chol = tf.linalg.set_diag(new_chol, new_diag)
    return new_chol
