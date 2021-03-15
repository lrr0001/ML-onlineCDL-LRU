from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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



from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export
import complex_split_operators as cplx_split_ops


class Adam(tf.keras.optimizers.Adam):
  r"""Optimizer that implements the Adam algorithm.

  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.

  According to
  [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
  the method is "*computationally
  efficient, has little memory requirement, invariant to diagonal rescaling of
  gradients, and is well suited for problems that are large in terms of
  data/parameters*".

  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use, The
      learning rate. Defaults to 0.001.
    beta_1: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use. The
      exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use, The
      exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon: A small constant for numerical stability. This epsilon is
      "epsilon hat" in the Kingma and Ba paper (in the formula just before
      Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
      1e-7.
    amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
      the paper "On the Convergence of Adam and beyond". Defaults to `False`.
    name: Optional name for the operations created when applying gradients.
      Defaults to `"Adam"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Usage:

  >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> # The first step is `-learning_rate*sign(grad)`
  >>> var1.numpy()
  9.9

  Reference:
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    - [Reddi et al., 2018](
        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

  Notes:

  The default value of 1e-7 for epsilon might not be a good default in
  general. For example, when training an Inception network on ImageNet a
  current good choice is 1.0 or 0.1. Note that since Adam uses the
  formulation just before Section 2.1 of the Kingma and Ba paper rather than
  the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
  hat" in the paper.

  The sparse implementation of this algorithm (used when the gradient is an
  IndexedSlices object, typically because of `tf.gather` or an embedding
  lookup in the forward pass) does apply momentum to variable slices even if
  they were not used in the forward pass (meaning they have a gradient equal
  to zero). Momentum decay (beta1) is also applied to the entire momentum
  accumulator. This means that the sparse behavior is equivalent to the dense
  behavior (in contrast to some momentum implementations which ignore momentum
  unless a variable slice was actually used).
  """

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    if var.dtype.is_complex:
      v_scaled_g_values = cmplx_split_ops.sqrare(grad) * coefficients['one_minus_beta_2_t']
    else:
      v_scaled_g_values = (grad*grad) * coefficients['one_minus_beta_2_t']
    v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                           use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

    if not self.amsgrad:
      if var.dtype.is_complex:
        v_sqrt = cmplx_split_ops.sqrt(v_t)
        var_update = state_ops.assign_sub(var, cmplx_split_ops.safe_divide(coefficients['lr'] * m_t,v_sqrt,coefficients['epsilon']), use_locking=self._use_locking)
      else:
        v_sqrt = math_ops.sqrt(v_t)
        var_update = state_ops.assign_sub(var, coefficients['lr'] * m_t/ (v_sqrt + coefficients['epsilon']), use_locking=self._use_locking) 
      return control_flow_ops.group(*[var_update, m_t, v_t])
    else:
      v_hat = self.get_slot(var, 'vhat')
      if var.dtype.is_complex:
        v_hat_t = cmplx_split_ops.realmaximum(v_hat, v_t)
      else:
        v_hat_t = math_ops.maximum(v_hat, v_t)
      with ops.control_dependencies([v_hat_t]):
        v_hat_t = state_ops.assign(
            v_hat, v_hat_t, use_locking=self._use_locking)
      v_hat_sqrt = math_ops.sqrt(v_hat_t)
      var_update = state_ops.assign_sub(
          var,
          coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])


class NonFusedAdam(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the Adam algorithm without fused kernels.

  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.
  According to the paper
  [Adam: A Method for Stochastic Optimization. Kingma et al.,
  2014](http://arxiv.org/abs/1412.6980), the method is "*computationally
  efficient, has little memory requirement, invariant to diagonal rescaling of
  gradients, and is well suited for problems that are large in terms of
  data/parameters*".

  For AMSGrad see [On The Convergence Of Adam And Beyond.
  Reddi et al., 5-8](https://openreview.net/pdf?id=ryQu7f-RZ).

  **If amsgrad = False**:

  initialize $m_0$ as 1st moment vector
  initialize $v_0$ as 2nd moment vector

  The update rule for $\theta$ with gradient $g$ uses an optimization
  described at the end of section 2 of the paper:

  $$lr_t = \mathrm{learning\_rate} *
    \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
  $$m_t = \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
  $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * g^2$$
  $$\theta_t = \theta_{t-1} - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$

  **If amsgrad = True**:

  initialize $m_0$ as 1st moment vector
  initialize $v_0$ as 2nd moment vector
  initialize $\hat{v}_0$ as 2nd moment vector

  The update rule for $\theta$ with gradient $g$ uses an optimization
  described at the end of section 2 of the paper:

  $$lr_t = \mathrm{learning\_rate} *
    \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$

  $$m_t = \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
  $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * g^2$$
  $$\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$$
  $$\theta_t = \theta_{t-1} - lr_t * m_t / (\sqrt{\hat{v}_t} + \epsilon)$$

  The default value of 1e-7 for epsilon might not be a good default in
  general. For example, when training an Inception network on ImageNet a
  current good choice is 1.0 or 0.1. Note that since Adam uses the
  formulation just before Section 2.1 of the Kingma and Ba paper rather than
  the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
  hat" in the paper.

  The sparse implementation of this algorithm (used when the gradient is an
  IndexedSlices object, typically because of `tf.gather` or an embedding
  lookup in the forward pass) does apply momentum to variable slices even if
  they were not used in the forward pass (meaning they have a gradient equal
  to zero). Momentum decay (beta1) is also applied to the entire momentum
  accumulator. This means that the sparse behavior is equivalent to the dense
  behavior (in contrast to some momentum implementations which ignore momentum
  unless a variable slice was actually used).

  Usage:

  >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> # The first step is `-learning_rate*sign(grad)`
  >>> var1.numpy()
  9.9
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               name='Adam',
               **kwargs):
    """Construct a new Adam optimizer.

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable that
        takes no arguments and returns the actual value to use, The learning
        rate. Defaults to 0.001.
      beta_1: A float value or a constant float tensor, or a callable that takes
        no arguments and returns the actual value to use. The exponential decay
        rate for the 1st moment estimates. Defaults to 0.9.
      beta_2: A float value or a constant float tensor, or a callable that takes
        no arguments and returns the actual value to use, The exponential decay
        rate for the 2nd moment estimates. Defaults to 0.999.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
      amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond". Defaults to `False`.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """

    super(NonFusedAdam, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self.amsgrad = amsgrad

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')
    for var in var_list:
      self.add_slot(var, 'v')
    if self.amsgrad:
      for var in var_list:
        self.add_slot(var, 'vhat')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(NonFusedAdam, self)._prepare_local(var_device, var_dtype, apply_state)

    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    lr = (
        apply_state[(var_device, var_dtype)]['lr_t'] *
        (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
    apply_state[(var_device, var_dtype)].update(
        dict(
            lr=lr,
            epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t))

  def set_weights(self, weights):
    params = self.weights
    # If the weights are generated by Keras V1 optimizer, it includes vhats
    # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
    # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
    num_vars = int((len(params) - 1) / 2)
    if len(weights) == 3 * num_vars + 1:
      weights = weights[:len(params)]
    super(NonFusedAdam, self).set_weights(weights)

  @def_function.function(experimental_compile=True)
  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    alpha = (
        coefficients['lr_t'] * math_ops.sqrt(1 - coefficients['beta_2_power']) /
        (1 - coefficients['beta_1_power']))
    m.assign_add((grad - m) * (1 - coefficients['beta_1_t']))
    v.assign_add((math_ops.square(grad) - v) * (1 - coefficients['beta_2_t']))
    if self.amsgrad:
      vhat = self.get_slot(var, 'vhat')
      vhat.assign(math_ops.maximum(vhat, v))
      v = vhat
    var.assign_sub(
        (m * alpha) / (math_ops.sqrt(v) - coefficients['epsilon']))

  @def_function.function(experimental_compile=True)
  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m.assign(m * coefficients['beta_1_t'])
    m.scatter_add(ops.IndexedSlices(m_scaled_g_values, indices))

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
    v.assign(v * coefficients['beta_2_t'])
    v.scatter_add(ops.IndexedSlices(v_scaled_g_values, indices))

    if not self.amsgrad:
      var.assign_sub(coefficients['lr'] * m /
                     (math_ops.sqrt(v) + coefficients['epsilon']))
    else:
      v_hat = self.get_slot(var, 'vhat')
      v_hat.assign(math_ops.maximum(v_hat, v))
      var.assign_sub(coefficients['lr'] * m /
                     (math_ops.sqrt(v_hat) + coefficients['epsilon']))

  def get_config(self):
    config = super(NonFusedAdam, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
        'amsgrad': self.amsgrad,
    })
    return config
