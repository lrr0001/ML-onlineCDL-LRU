import tensorflow as tf

class fft2d(tf.keras.layers.Layer):
    def __init__(self,fftSz,*args,**kwargs):
        self.fftSz = fftSz
        super().__init__(*args,autocast=False,**kwargs)
    def call(self,inputs):
        return tf.signal.rfft2d(input_tensor=inputs,fft_length=self.fftSz)


class switch_spatial_and_channelfltr(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,autocast=False,**kwargs)
    def call(self,inputs):
        return tf.transpose(a=inputs,perm=(0,3,4,1,2),conjugate=False)

class ifft2d(fft2d):
    def call(self,inputs):
        return tf.signal.irfft2d(input_tensor=inputs,fft_length=self.fftSz)

class fft2d_inner(fft2d):
    def call(self,inputs):
        return switch_spatial_and_channelfltr()(fft2d(self.fftSz)(switch_spatial_and_channelfltr()(inputs)))

class ifft2d_inner(fft2d):
    def call(self,inputs):
        return switch_spatial_and_channelfltr()(ifft2d(self.fftSz)(switch_spatial_and_channelfltr()(inputs)))

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
