import tensorflow as tf
import math

def generate_dct2D_filters():
    x = tf.reshape(2*tf.range(8.) + 1,(8,1,1,1))
    y = tf.reshape(2*tf.range(8.) + 1,(1,8,1,1))
    u = tf.reshape(math.pi*tf.range(8.)/16.,(1,1,8,1))
    v = tf.reshape(math.pi*tf.range(8.)/16.,(1,1,1,8))
    alpha_u = tf.reshape([1./math.sqrt(2),1.,1.,1.,1.,1.,1.,1.],(1,1,8,1))
    alpha_v = tf.reshape([1./math.sqrt(2),1.,1.,1.,1.,1.,1.,1.],(1,1,1,8))
    cosxu = alpha_u*tf.math.cos(x*u)
    cosyv = alpha_v*tf.math.cos(y*v)
    return tf.reshape(1./4.*cosxu*cosyv,(8,8,1,64))
    

def generate_idct2D_filters():
    return tf.reshape(tf.transpose(generate_dct2D_filters(),perm=(3,0,1,2)),(8,8,1,64))

class RGB2JPEG_Coef(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dct_filters = generate_dct2D_filters()
        self.downsample = tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='valid')
    def call(self,inputs):
        yuv = tf.image.rgb_to_yuv(inputs)
        y,u,v = tf.split(yuv,axis=3,num_or_size_splits=3)
        y_calibrated = y - 1./2.
        u_ds = self.downsample(u)
        v_ds = self.downsample(v)
        ydcc_blks = tf.nn.conv2d(y_calibrated,self.dct_filters,strides=8,padding='VALID')
        udcc_blks = tf.nn.conv2d(u_ds,self.dct_filters,strides=8,padding='VALID')
        vdcc_blks = tf.nn.conv2d(v_ds,self.dct_filters,strides=8,padding='VALID')
        return (ydcc_blks,udcc_blks,vdcc_blks) # discrete cosine coefficients

class JPEG_Coef2RGB(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.idct_filters = generate_idct2D_filters()
        self.Upsample = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')
    def call(self,inputs):
        ydcc_blks,udcc_blks,vdcc_blks = inputs
        ydcc = tf.nn.depth_to_space(ydcc_blks,block_size=8)
        udcc = tf.nn.depth_to_space(udcc_blks,block_size=8)
        vdcc = tf.nn.depth_to_space(vdcc_blks,block_size=8)
        y_blks = tf.nn.conv2d(ydcc,self.idct_filters,strides=8,padding='VALID')
        u_ds_blks = tf.nn.conv2d(udcc,self.idct_filters,strides=8,padding='VALID')
        v_ds_blks = tf.nn.conv2d(vdcc,self.idct_filters,strides=8,padding='VALID')
        y = tf.clip_by_value(tf.nn.depth_to_space(y_blks,block_size=8) + 1./2.,0.,1.)
        u_ds = tf.nn.depth_to_space(u_ds_blks,block_size = 8)
        v_ds = tf.nn.depth_to_space(v_ds_blks,block_size = 8)      
        u = tf.clip_by_value(self.Upsample(u_ds),-0.5,0.5)
        v = tf.clip_by_value(self.Upsample(v_ds),-0.5,0.5)
        yuv = tf.concat((y,u,v),axis=3)
        return tf.clip_by_value(tf.image.yuv_to_rgb(yuv),0.,1.)
        
def get_JPEG_coef_mask(jpegImages,rgb2jpeg_coef_layer,epsilon):
    ydcc_blks,udcc_blks,vdcc_blks = rgb2jpeg_coef_layer(jpegImages)
    nonzero = lambda x: tf.math.greater(tf.math.abs(x),epsilon)
    return (nonzero(ydcc_blks),nonzero(udcc_blks),nonzero(vdcc_blks))

class Linear_JPEG_Compression_Approx(tf.keras.layers.Layer):
    def __init__(self,jpegImages,epsilon=1e-4,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.jpeg_coef2rgb = JPEG_Coef2RGB(*args,**kwargs)
        self.rgb2jpeg_coef = RGB2JPEG_Coef(*args,**kwargs)
        self.masky,self.masku,self.maskv = get_JPEG_coef_mask(jpegImages,self.rgb2jpeg_coef,epsilon=1e-4)
    def call(self,inputs):
        y_jpeg_coef,u_jpeg_coef,v_jpeg_coef = self.rgb2jpeg_coef(inputs)
        y_jpeg_coef_kept = tf.where(self.masky,y_jpeg_coef,0.)
        u_jpeg_coef_kept = tf.where(self.masku,u_jpeg_coef,0.)
        v_jpeg_coef_kept = tf.where(self.maskv,v_jpeg_coef,0.)
        return self.jpeg_coef2rgb((y_jpeg_coef_kept,u_jpeg_coef_kept,v_jpeg_coef_kept))
        
