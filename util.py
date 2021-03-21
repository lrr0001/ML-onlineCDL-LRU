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
    return tf.expand_dims(input=x,axis=-1)
    #return tf.reshape(x,shape=x.shape + (1,))

def complexNum(x):
    return tf.complex(x,tf.cast(0.0,dtype = x.dtype))

def alternate_concat(values,axis): 
    concat_tensor = tf.concat(list(map(alternate_reshape,zip(values,len(values)*(axis,)))),axis)
    return tf.reshape(concat_tensor,concat_tensor.shape[:axis] + (concat_tensor.shape[axis]*concat_tensor.shape[axis + 1],) + concat_tensor.shape[axis + 2:])

class AlternateConcat(tf.keras.layers.Layer):
    def __init__(self,shape,num_of_items,axis,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert(axis != 0)
        self.axis = axis
        self.num_of_items = num_of_items
        target_shape = shape[:axis - 1] + (shape[axis - 1]*num_of_items,) + shape[axis:]
        self.reshape = tf.keras.layers.Reshape(target_shape = target_shape, dtype = self.dtype)
    def call(self,inputs):
        concat_tensor = tf.concat(list(map(alternate_reshape,zip(inputs,self.num_of_items*(self.axis,)))),self.axis)
        return self.reshape(concat_tensor)
    def get_config(self):
        return {'axis': self.axis, 'num_of_items': self.num_of_items}

def alternate_reshape(inputs):
    input_tensor,axis = inputs
    return tf.expand_dims(input=input_tensor,axis=axis)
    #return tf.reshape(input_tensor,input_tensor.shape[:axis] + (1,) + input_shape[axis:])

def downsample(input_tensor):
    return tf.strided_slice(input_tensor,begin= [0,0,0],end = [0,0,0],strides=[1,2,2],end_mask=7)

def freq_downsample(input_tensor):
    # Assumes dimensions in spatial domain are even and that rfft is taken. Still needs to be tested.
    M = input_tensor.shape[1]
    M_over_2 = int(M/2)
    N_over_2 = input_tensor.shape[2] - 1
    if N_over_2 % 2 == 1:
        N_over_4 = int(input_tensor.shape[2]/2)
    else:
        N_over_4 = int(N_over_2/2) + 1
    bottomright = tf.strided_slice(input_tensor,begin=[0,1,1],end=[0,M_over_2,N_over_4],end_mask=1)
    bottomright += tf.strided_slice(input_tensor,begin=[0,M_over_2 + 1,1],end=[0,M,N_over_4],end_mask=1)
    bottomright += tf.math.conj(tf.strided_slice(input_tensor,begin=[0,M_over_2 - 1,N_over_2 - 1],end=[0,0,N_over_2 - N_over_4],strides=[1,-1,-1],end_mask=1))
    bottomright += tf.math.conj(tf.strided_slice(input_tensor,begin=[0,M - 1,N_over_2 - 1],end=[0,M_over_2,N_over_2 - N_over_4],strides=[1,-1,-1],end_mask=1))
    topleft = tf.strided_slice(input_tensor,begin=[0,0,0],end=[0,1,1],end_mask=1)
    topleft += tf.strided_slice(input_tensor,begin=[0,M_over_2,0],end=[0,M_over_2 + 1,1],end_mask=1)
    topleft += tf.strided_slice(input_tensor,begin=[0,0,N_over_2],end=[0,1,N_over_2 + 1],end_mask=1)
    topleft += tf.strided_slice(input_tensor,begin=[0,M_over_2,N_over_2],end=[0,M_over_2 + 1,N_over_2 + 1],end_mask=1)
    topright = tf.strided_slice(input_tensor,begin=[0,0,1],end=[0,1,N_over_4],end_mask=1)
    topright += tf.strided_slice(input_tensor,begin=[0,M_over_2,1],end=[0,M_over_2 + 1,N_over_4],end_mask=1)
    topright += tf.math.conj(tf.strided_slice(input_tensor,begin=[0,0,N_over_2 - 1],end=[0,1,N_over_2 - N_over_4],strides=[1,1,-1],end_mask=1))
    topright += tf.math.conj(tf.strided_slice(input_tensor,begin=[0,M_over_2,N_over_2 - 1],end=[0,M_over_2 + 1,N_over_2 - N_over_4],strides=[1,1,-1],end_mask=1))
    bottomleft = tf.strided_slice(input_tensor,begin=[0,1,0],end=[0,M_over_2,1],end_mask=1)
    bottomleft += tf.strided_slice(input_tensor,begin=[0,M_over_2 + 1,0],end=[0,M,1],end_mask=1)
    bottomleft += tf.math.conj(tf.strided_slice(input_tensor,begin=[0,M - 1,N_over_2],end=[0,M_over_2,N_over_2 + 1],strides=[1,-1,1],end_mask=1))
    bottomleft += tf.math.conj(tf.strided_slice(input_tensor,begin=[0,M_over_2 - 1,N_over_2],end=[0,0,N_over_2 + 1],strides=[1,-1,1],end_mask=1))
    return tf.concat((tf.concat((topleft,bottomleft),axis=1),tf.concat((topright,bottomright),axis=1)),axis=2)/4.

def shift_downsample(input_tensor):
    return tf.strided_slice(input_tensor,begin= [0,1,0],end = [0,0,0],strides=[1,2,2],end_mask=7)

def col_downsample(input_tensor):
    return tf.strided_slice(input_tensor,begin=[0,0,1],end=[0,0,0],strides=[1,1,2],end_mask=7)
