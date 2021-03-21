import tensorflow as tf
import util

class fft2d(tf.keras.layers.Layer):
    def __init__(self,fftSz,*args,**kwargs):
        self.fftSz = fftSz
        super().__init__(*args,autocast=False,**kwargs)
    def call(self,inputs):
        return tf.signal.rfft2d(input_tensor=inputs,fft_length=self.fftSz)
    def get_config(self):
        return {'fftSz': fftSz}


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

class fft2d_multichannel(fft2d):
    def call(self,inputs):
        return util.rotate_dims_left(fft2d(self.fftSz)(util.rotate_dims_right(inputs,4)),4)

class ifft2d_multichannel(fft2d):
    def call(self,inputs):
        return util.rotate_dims_left(ifft2d(self.fftSz)(util.rotate_dims_right(inputs,4)),4)
