import tensorflow as tf
import util

class fft2d(tf.keras.layers.Layer):
    def __init__(self,fftSz,*args,**kwargs):
        self.fftSz = fftSz
        super().__init__(*args,autocast=False,**kwargs)
    def fft(self,inputs):
        return tf.signal.rfft2d(input_tensor=inputs,fft_length=self.fftSz)
    def call(self,inputs):
        return self.fft(inputs)
    def get_config(self):
        return {'fftSz': fftSz}

def _sscf(inputs):
    return tf.transpose(a=inputs,perm=(0,3,4,1,2),conjugate=False)

class switch_spatial_and_channelfltr(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,autocast=False,**kwargs)
    def call(self,inputs):
        return _sscf(inputs)

class ifft2d(fft2d):
    def fft(self,inputs):
        return tf.signal.irfft2d(input_tensor=inputs,fft_length=self.fftSz)

class fft2d_inner(fft2d):
    def call(self,inputs):
        return _sscf(self.fft(_sscf(inputs)))

class ifft2d_inner(ifft2d):
    def call(self,inputs):
        return _sscf(self.fft(_sscf(inputs)))

class fft2d_multichannel(fft2d):
    def call(self,inputs):
        return util.rotate_dims_left(self.fft(util.rotate_dims_right(inputs,4)),4)

class ifft2d_multichannel(ifft2d):
    def call(self,inputs):
        return util.rotate_dims_left(self.fft(util.rotate_dims_right(inputs,4)),4)
