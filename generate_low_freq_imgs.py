import tensorflow as tf
import jpeg_related_functions as jrf
import pickle as pkl

# Set parameters
jpeg_quality = 25
rho = 1.
alpha = 1.5
noi = 20
lmbda = 0.1
dtype = 'float64'
lmbda_t = 3*lmbda

datasetname = 'BSDS500/'
#datasetname = 'simpleTest/'

# Obtain quantization matrices from chosen quality factor
import PIL
import PIL.Image
import numpy as np
randImgPath = 'data/scratchwork/example/randImg.jpeg'
randimg = np.random.randint(0,256,size=(32,32,3))
encoded_jpeg = tf.image.encode_jpeg(randimg,quality = jpeg_quality)
tf.io.write_file(randImgPath,encoded_jpeg)
loadedRandImg = PIL.Image.open(randImgPath)
qY = np.asarray(loadedRandImg.quantization[0]).astype('uint8')
qUV = np.asarray(loadedRandImg.quantization[1]).astype('uint8')
qY = qY.astype(dtype)/255.
qUV = qUV.astype(dtype)/255.
import os
os.remove(randImgPath)

# setup smoothing layer
fftSz1 = (480,320)
fftSz2 = (320,480)
#smooth_jpeg1 = jrf.Smooth_JPEG(rho,alpha,noi,qY,qUV,lmbda,fftSz1,dtype=dtype)
#smooth_jpeg2 = jrf.Smooth_JPEG(rho,alpha,noi,qY,qUV,lmbda,fftSz2,dtype=dtype)
smooth_jpeg1 = jrf.Smooth_JPEGY(rho,alpha,noi,qY,lmbda,fftSz1,dtype=dtype)
smooth_jpeg2 = jrf.Smooth_JPEGY(rho,alpha,noi,qY,lmbda,fftSz2,dtype=dtype)

Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,dtype = dtype),tf.cast(0.,dtype= dtype))
# Loop through images
dataPath = 'data/original/' + datasetname
filelist = os.listdir(dataPath)
savePath = 'data/scratchwork/' + datasetname + '/whole/'
for datatype in ['val/',]:
#for datatype in ['train/','val/','test/']:
    filelist = os.listdir(dataPath + datatype)
    for filename in filelist:
        loadedImg = PIL.Image.open(dataPath + datatype + filename)
        loadedImg = np.asarray(loadedImg).astype(dtype)/255.
        loadedImgShape = loadedImg.shape
        # crop out a row and a column
        loadedImg = loadedImg[slice(0,loadedImgShape[0] - (loadedImgShape[0] % 8)),slice(0,loadedImgShape[1] - (loadedImgShape[1] % 8)),slice(None)]
        raw = tf.reshape(loadedImg,(1,) + loadedImg.shape)
        if loadedImgShape[0] - (loadedImgShape[0] % 8) == 480 and loadedImgShape[1] - (loadedImgShape[1] % 8) == 320:
           # compressedImg = smooth_jpeg1.Wt(jrf.threeChannelQuantize(smooth_jpeg1.W(tf.reshape(loadedImg,(1,) + loadedImg.shape)),qY,qUV,Yoffset))
            lowpass,compressedImg = smooth_jpeg1(raw)
        elif loadedImgShape[0] - (loadedImgShape[0] % 8) == 320 and loadedImgShape[1] - (loadedImgShape[1] % 8) == 480:
            #compressedImg = smooth_jpeg2.Wt(jrf.threeChannelQuantize(smooth_jpeg2.W(tf.reshape(loadedImg,(1,) + loadedImg.shape)),qY,qUV,Yoffset))
            lowpass,compressedImg = smooth_jpeg2(raw)
        else:
            raise ValueError('Unexpected Shape!')

        # Need to save lowpass, highpass and raw into pickle file
        fid = open(savePath + datatype + filename + '.pckl','wb')

        pkl.dump({'lowpass': tf.reshape(lowpass,lowpass.shape[1:]),'highpass': tf.reshape(compressedImg - lowpass,lowpass.shape[1:]), 'raw': tf.reshape(raw,raw.shape[1:])},fid)
        fid.close()


# Code added to train on raw images.
smooth_jpeg1 = jrf.XUpdate_SmoothJPEG(tf.cast(lmbda_t,'complex128'),fftSz1,tf.reshape(tf.cast([1.,-1.],'float64'),(1,2,1,1)),tf.reshape(tf.cast([1.,-1.],'float64'),(1,1,2,1)),dtype=dtype)
smooth_jpeg2 = jrf.XUpdate_SmoothJPEG(tf.cast(lmbda_t,'complex128'),fftSz2,tf.reshape(tf.cast([1.,-1.],'float64'),(1,2,1,1)),tf.reshape(tf.cast([1.,-1.],'float64'),(1,1,2,1)),dtype=dtype)
rgb2yuv = jrf.RGB2YUV(dtype = dtype)
for datatype in ['train/',]:
    filelist = os.listdir(dataPath + datatype)
    for filename in filelist:
        loadedImg = PIL.Image.open(dataPath + datatype + filename)
        loadedImg = np.asarray(loadedImg).astype(dtype)/255.
        loadedImgShape = loadedImg.shape
        loadedImg = loadedImg[slice(0,loadedImgShape[0] - (loadedImgShape[0] % 8)),slice(0,loadedImgShape[1] - (loadedImgShape[1] % 8)),slice(None)]
        raw = tf.reshape(loadedImg,(1,) + loadedImg.shape)
        if loadedImgShape[0] - (loadedImgShape[0] % 8) == 480 and loadedImgShape[1] - (loadedImgShape[1] % 8) == 320:
           # compressedImg = smooth_jpeg1.Wt(jrf.threeChannelQuantize(smooth_jpeg1.W(tf.reshape(loadedImg,(1,) + loadedImg.shape)),qY,qUV,Yoffset))
            compressedImg = rgb2yuv(raw)
            compressedImg = compressedImg[slice(None),slice(None),slice(None),slice(0,1)]
            lowpass = smooth_jpeg1(compressedImg)
        elif loadedImgShape[0] - (loadedImgShape[0] % 8) == 320 and loadedImgShape[1] - (loadedImgShape[1] % 8) == 480:
            #compressedImg = smooth_jpeg2.Wt(jrf.threeChannelQuantize(smooth_jpeg2.W(tf.reshape(loadedImg,(1,) + loadedImg.shape)),qY,qUV,Yoffset))
            compressedImg = rgb2yuv(raw)
            compressedImg = compressedImg[slice(None),slice(None),slice(None),slice(0,1)]
            lowpass = smooth_jpeg2(compressedImg)
        else:
            raise ValueError('Unexpected Shape!')

        # Need to save lowpass, highpass and raw into pickle file
        fid = open(savePath + datatype + filename + '.pckl','wb')
        pkl.dump({'lowpass': tf.reshape(lowpass,lowpass.shape[1:]),'highpass': tf.reshape(compressedImg - lowpass,lowpass.shape[1:]), 'raw': tf.reshape(raw,raw.shape[1:])},fid)
        fid.close()




fid = open('data/processed/' + datasetname + 'param.pckl','wb')
pkl.dump({'qY': qY,'qUV': qUV,'jpeg_quality': jpeg_quality, 'lmbda': lmbda, 'smoothing_noi': noi,'dtype':dtype},fid)
fid.close()
