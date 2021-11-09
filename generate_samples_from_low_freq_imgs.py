import tensorflow as tf
import pickle as pkl
import jpeg_related_functions as jrf
# ************FUNCTION DEFINITION******************
def savePatch(coord,lowpass,highpass,raw,padding,patch_size,qY,qUV,Yoffset,datapath,filename,dtype):
    rowCoord,colCoord = coord
    lowpassPatch = lowpass[slice(rowCoord - padding[0][0],rowCoord + patch_size[0] + padding[0][1]),slice(colCoord - padding[1][0],colCoord + patch_size[1] + padding[1][1]),slice(None)]
    #lowpassPatch = lowpass[slice(rowCoord,rowCoord + patch_size[0]),slice(colCoord,colCoord + patch_size[1]),slice(None)]
    rawPatch =  raw[slice(rowCoord,rowCoord + patch_size[0]),slice(colCoord,colCoord + patch_size[1]),slice(None)]
    #W = jrf.YUV2JPEG_Coef(dtype = dtype)
    #Wt = jrf.JPEG_Coef2YUV(dtype = dtype)
    #W = jrf.RGB2JPEG_Coef(dtype = dtype)
    #Wt = jrf.JPEG_Coef2RGB(dtype = dtype)
    #compressedPatch = Wt(jrf.threeChannelQuantize(W(jrf.RGB2YUV(dtype=dtype)(tf.expand_dims(rawPatch,axis=0))),qY,qUV,Yoffset))
    #compressedPatch = Wt(jrf.threeChannelQuantize(W(tf.expand_dims(rawPatch,axis=0)),qY,qUV,Yoffset))
    #compressedPatch = tf.squeeze(compressedPatch,axis=0)
    compressedPatch = tf.squeeze(jrf.quantize(tf.expand_dims(rawPatch,axis=0),qY,Yoffset),axis = 0)
    highpassPatch = highpass[slice(rowCoord - padding[0][0],rowCoord + patch_size[0] + padding[0][1]),slice(colCoord - padding[1][0],colCoord + patch_size[1] + padding[1][1]),slice(None)]
    fid_raw = open(datapath + 'raw/' + filename,'wb')
    pkl.dump(rawPatch,fid_raw)
    fid_raw.close()
    fid_lowpass = open(datapath + 'lowpass/' + filename,'wb')
    pkl.dump(lowpassPatch,fid_lowpass)
    fid_lowpass.close()
    fid_highpass = open(datapath + 'highpass/' + filename,'wb')
    pkl.dump(highpassPatch,fid_highpass)
    fid_highpass.close()
    fid_compressed = open(datapath + 'compressed/' + filename,'wb')
    pkl.dump(compressedPatch,fid_compressed)
    fid_compressed.close()


# ************CODE START******************

# Choose parameters
patch_size = (32,32)
padding = ((13,13),(13,13))
startOffset1 = (16,16)
startOffset2 = (16,16)
# load previous parameters

datasetname = 'BSDS500/'
#datasetname = 'simpleTest/'
dataloadpath = 'data/scratchwork/' + datasetname + 'whole/'
datasavepath = 'data/scratchwork/' + datasetname + 'patches/'
fid = open('data/processed/' + datasetname + 'param.pckl','rb')
python_dict = pkl.load(fid)
dtype = python_dict['dtype']
qY = python_dict['qY']
qUV = python_dict['qUV']
fid.close()
python_dict['padding'] = padding
python_dict['target_size'] = patch_size
fid = open('data/processed/' + datasetname + 'param.pckl','wb')
pkl.dump(python_dict,fid)
fid.close()

# Loop through saved pickle files
import os


Yoffset = tf.one_hot([[[0]]],64,tf.cast(32.,dtype),tf.cast(0.,dtype))


#for datatype in ['train/','val/']:
for datatype in ['train/','val/','test/']:
    filelist = os.listdir(dataloadpath + datatype)
    for filename in filelist:
        fid = open(dataloadpath + datatype + filename,'rb')
        storedVar = pkl.load(fid)
        fid.close()
        lowpass = storedVar['lowpass']
        highpass = storedVar['highpass']
        raw = storedVar['raw']
        lowpassShape = lowpass.shape
        if lowpassShape[0] == 480 and lowpassShape[1] == 320:
            r = 0
            for rowcoord in range(startOffset1[0],lowpass.shape[0] - patch_size[0] - padding[0][1] + 1,patch_size[0]):
                c = 0
                for colcoord in range(startOffset1[1],lowpass.shape[1] - patch_size[1] - padding[1][1] + 1,patch_size[1]):
                    ind_str = '_' + str(r) + '_' + str(c) + '.pckl'
                    savePatch(coord=(rowcoord,colcoord),lowpass=lowpass,highpass=highpass,raw=raw,padding=padding,patch_size=patch_size,qY=qY,qUV=qUV,Yoffset=Yoffset,datapath=datasavepath + datatype,filename = filename + ind_str,dtype = dtype)
                    c += 1
                r += 1
        elif lowpassShape[0] == 320 and lowpassShape[1] == 480:
            r = 0
            for rowcoord in range(startOffset2[0],lowpass.shape[0] - patch_size[0] - padding[0][1] + 1,patch_size[0]):
                c = 0
                for colcoord in range(startOffset2[1],lowpass.shape[1] - patch_size[1] - padding[1][1] + 1,patch_size[1]):
                    ind_str = '_' + str(c) + '_' + str(r) + '.pckl'
                    savePatch(coord=(rowcoord,colcoord),lowpass=lowpass,highpass=highpass,raw=raw,padding=padding,patch_size=patch_size,qY=qY,qUV=qUV,Yoffset=Yoffset,datapath=datasavepath +datatype,filename = filename + ind_str,dtype = dtype)
                    c += 1
                r += 1
        else:
            raise ValueError('Unexpected shape!')



