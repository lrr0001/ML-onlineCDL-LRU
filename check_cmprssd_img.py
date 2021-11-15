import tensorflow as tf
import numpy as np
import pickle as pkl
import multilayerCSC_ADMM as mlcsc
fid = open('data/scratchwork/BSDS500/patches/train/compressed/271008.jpg.pckl_12_7.pckl','rb')
cmprssdImg = pkl.load(fid)
fid.close()

print(tf.reduce_max(cmprssdImg))
print(tf.reduce_min(cmprssdImg))


modelname = 'ML_LRA_'
databasename = 'BSDS500/'
#databasename = 'simpleTest/'
experimentname = 'experiment1/'
experimentpath = 'data/experiment/' + databasename + experimentname
checkpointfilename = modelname + 'checkpoint_epoch_{epoch:02d}.ckpt'
timesname = modelname + 'times.pkl'
modelfilename = modelname + 'initial_model.ckpt'
fid = open(experimentpath + 'problem_param.pckl','rb')
problem_param = pkl.load(fid)
fid.close()
data_param = problem_param['data_param']
targetSz = data_param['target_size']
qY = data_param['qY']
qUV = data_param['qUV']
strides = problem_param['stride']
fltrSz = problem_param['fltrSz']
real_dtype = data_param['dtype']
noi = problem_param['noi']
noL = problem_param['noL']
noc = problem_param['noc']
datapath = problem_param['datapath']
trainfile = problem_param['trainfile']
padding = data_param['padding']


#   ******** CROPPING AND PADDING ********
cropAndMerge = mlcsc.CropPadObject(targetSz,strides,[np.asarray(ks) for ks in fltrSz],real_dtype)
paddingTuple = cropAndMerge.paddingTuple
fftSz = cropAndMerge.get_fft_size(targetSz,strides)
paddingDiff = ((padding[0][0] - paddingTuple[0][0],padding[0][1] - paddingTuple[0][1]),(padding[1][0] - paddingTuple[1][0],padding[1][1] - paddingTuple[1][1]))
assert(paddingDiff[0][0] >= 0)
assert(paddingDiff[0][1] >= 0)
assert(paddingDiff[1][0] >= 0)
assert(paddingDiff[1][1] >= 0)
print(paddingDiff)

startr = paddingDiff[0][0]
startc = paddingDiff[1][0]
endr = targetSz[0] + padding[0][0] + padding[0][1] - paddingDiff[0][1]
endc = targetSz[1] + padding[1][0] + padding[1][1] - paddingDiff[1][1]

real_dtype = 'float64'
example_structure = {'highpass': tf.io.FixedLenFeature([], tf.string), 'lowpass': tf.io.FixedLenFeature([], tf.string), 'compressed': tf.io.FixedLenFeature([], tf.string),'raw': tf.io.FixedLenFeature([], tf.string)}

def restore_double(x):
    return tf.io.parse_tensor(x,real_dtype)

def _parse_image_function(example_proto):
    x = tf.io.parse_single_example(example_proto, example_structure)
    highpass = restore_double(x['highpass'])
    lowpass = restore_double(x['lowpass'])
    return ((highpass[slice(startr,endr),slice(startc,endc),slice(None)],lowpass[slice(startr,endr),slice(startc,endc),slice(None)],restore_double(x['compressed'])),restore_double(x['raw']))

datapath = 'data/processed/BSDS500/'
trainfile = 'train.tfrecord'

raw_dataset = tf.data.TFRecordDataset([datapath + trainfile])
dataset = raw_dataset.map(_parse_image_function)
dataset_batch = dataset.batch(batch_size)
dataset_batch = dataset_batch.repeat()

ii = 0
for (x,y) in dataset_batch:
    print(tf.reduce_max(tf.abs(croppedHighpass + croppedLowpass - x[2])))
    print(tf.reduce_max(x[2]))
    print(tf.reduce_min(x[2]))
    ii += 1
    if ii > 8:
        break
