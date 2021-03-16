import tensorflow as tf
import jpeg_related_functions as jrf
import pickle as pkl

# Set parameters
jpeg_quality = 20
rho = 1.
alpha = 1.5
noi = 10
lmbda = 1.
dtype = 'float64'

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
smooth_jpeg1 = jrf.Smooth_JPEG(rho,alpha,noi,qY,qUV,lmbda,fftSz1,dtype=dtype)
smooth_jpeg2 = jrf.Smooth_JPEG(rho,alpha,noi,qY,qUV,lmbda,fftSz2,dtype=dtype)
# Loop through images
dataPath = 'data/original/simpleTest/'
filelist = os.listdir(dataPath)
savePath = 'data/scratchwork/simpleTest/whole/'
for datatype in ['train/','val/',]:
    filelist = os.listdir(dataPath + datatype)
    for filename in filelist:
        loadedImg = PIL.Image.open(dataPath + datatype + filename)
        loadedImg = np.asarray(loadedImg).astype(dtype)/255.
        loadedImgShape = loadedImg.shape
        # crop out a row and a column
        loadedImg = loadedImg[slice(0,loadedImgShape[0] - (loadedImgShape[0] % 8)),slice(0,loadedImgShape[1] - (loadedImgShape[1] % 8)),slice(None)]
        if loadedImgShape[0] - (loadedImgShape[0] % 8) == 480 and loadedImgShape[1] - (loadedImgShape[1] % 8) == 320:
            compressedImg = smooth_jpeg1.Wt(smooth_jpeg1.W(tf.reshape(loadedImg,(1,) + loadedImg.shape)))
            lowpass,negC = smooth_jpeg1(compressedImg)
        elif loadedImgShape[0] - (loadedImgShape[0] % 8) == 320 and loadedImgShape[1] - (loadedImgShape[1] % 8) == 480:
            compressedImg = smooth_jpeg2.Wt(smooth_jpeg2.W(tf.reshape(loadedImg,(1,) + loadedImg.shape)))
            lowpass,negC = smooth_jpeg2(compressedImg)
        else:
            raise ValueError('Unexpected Shape!')

        # Need to save lowpass, highpass and raw into pickle file
        fid = open(savePath + datatype + filename + '.pckl','wb')

        pkl.dump({'lowpass': tf.reshape(lowpass,lowpass.shape[1:]),'highpass': tf.reshape(compressedImg - lowpass,lowpass.shape[1:]), 'raw': loadedImg},fid)
        fid.close()    


fid = open('data/processed/simpleTest/param.pckl','wb')
pkl.dump({'qY': qY,'qUV': qUV,'jpeg_quality': jpeg_quality, 'lmbda': lmbda, 'smoothing_noi': noi,'dtype':dtype},fid)
fid.close()
