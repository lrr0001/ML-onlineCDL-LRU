import tensorflow as tf

def complexify_dtype(dtype):
    if dtype.is_floating:
        if dtype == tf.float16:
            return tf.complex64
        if dtype == tf.float32:
            return tf.complex64
        if dtype == tf.float64:
            return tf.complex128
    elif dtype.is_complex:
        return tf.dtype
    else:
        ValueError('complexify_dtype only accepts floating point or complex floating point tensorflow data-type objects.')

def cmplx_sqr(x):
  return x*tf.math.conj(x)

def rotate_dims_right(x,r = None):
  if r is None:
    r = tf.rank(x)
  return tf.transpose(x,perm=tf.concat((tf.range(r-1,r),tf.range(0,r - 1)),axis=0))

def rotate_dims_left(x,r = None):
  if r is None:
    r = tf.rank(x)
  return tf.transpose(x,perm=tf.concat((tf.range(1,r),tf.range(0,1)),axis=0))

def conj_tp(x):
    return tf.transpose(x,perm=(0,1,2,4,3),conjugate=True)

def addDim(x):
    return tf.reshape(x,shape=x.shape + (1,))

def complexNum(x):
    return tf.complex(x,tf.cast(0.0,dtype = x.dtype))
