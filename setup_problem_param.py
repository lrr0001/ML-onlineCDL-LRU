import tensorflow as tf
import pickle as pkl

nof = [32,64,128]
noc = 3
stride = [2,1]
fltrSz = ((5,5),(7,7),(5,5))
noL = 3
noi = 96

D = []
D.append(np.random.randn(1,fltrSz[0][0],fltrSz[0][1],noc,nof[0]))
for layer in range(1,noL):
    D.append(np.random.randn(1,fltrSz[layer][0],fltrSz[layer][1],nof[layer - 1],nof[layer])

experimentpath = 'data/experiments/experiment1/'
datapath = 'data/processed/simpleTest/

trainfile = 'train/data.tfrecord'
valfile = 'val/data.tfrecord'
testfile = 'test/data.tfrecord'

fid = open(datapath + 'param.pckl','rb')
data_param = pkl.dump(fid)
fid.close()

param = {'data_param': data_param,
         'nof': nof,
         'noc': noc,
         'noi': noi,
         'stride': stride,
         'fltrSz':fltrSize,
         'noL':noL,
         'D': D,
         'datapath': datapath
         'trainfile': trainfile
         'valfile': valfile
         'testfile': testfile}


filename = 'problem_param.pckl'
fid = open(experimentpath + filename,'wb')
pkl.dump(param,fid)
fid.close()

