import tensorflow as tf
import pickle as pkl
import numpy as np

nof = [32]
noc = 3
stride = []
fltrSz = ((5,5),)
noL = len(fltrSz)
noi =  200

D = []
D_first = np.random.randn(1,fltrSz[0][0],fltrSz[0][1],noc,nof[0]).astype('float64')
D.append(D_first - tf.math.reduce_mean(input_tensor=D_first,axis = (1,2),keepdims=True))
#D.append(np.random.randn(1,fltrSz[0][0],fltrSz[0][1],noc,nof[0]).astype('float64'))
for layer in range(1,noL):
    D.append(np.random.randn(1,fltrSz[layer][0],fltrSz[layer][1],nof[layer - 1],nof[layer]).astype('float64'))

experimentpath = 'data/experiment/simpleTest/experiment1/'
datapath = 'data/processed/simpleTest/'

trainfile = 'train.tfrecord'
valfile = 'val.tfrecord'
testfile = 'test.tfrecord'

fid = open(datapath + 'param.pckl','rb')
data_param = pkl.load(fid)
fid.close()

param = {'data_param': data_param,
         'nof': nof,
         'noc': noc,
         'noi': noi,
         'stride': stride,
         'fltrSz':fltrSz,
         'noL':noL,
         'D': D,
         'datapath': datapath,
         'trainfile': trainfile,
         'valfile': valfile,
         'testfile': testfile}


filename = 'problem_param.pckl'
fid = open(experimentpath + filename,'wb')
pkl.dump(param,fid)
fid.close()

