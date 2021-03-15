import tensorflow as tf
import jpeg_related_functions as jrf
import multilayerCSC_ADMM as mlcsc
import numpy as np
import pickle as pkl

#   ******** ALGORITHM-SPECIFIC HYPERPARAMETERS ********
rho = 1.
alpha_init = 1.5
mu_init = 1.
lraParam = {'n_components': 3}
cmplxdtype = tf.complex128 # This should really be elsewhere.
batch_size = 8
noe_per_save = 4
num_of_saves = 24


#   ******** DATA AND EXPERIMENT PARAMETERS ********
experimentpath = 'data/experiment/experiment1/'
def checkpointfilename(ii):
    return 'checkpoint_epoch' + str(ii) + '.ckpt'
modelfilename = 'initial_model.pb'
fid = open(experimentpath + 'problem_param.pckl','rb')
problem_param = pkl.load(fid)
fid.close()
data_param = problem_param['data_param']
targetSz = data_param['target_size']
strides = problem_param['strides']
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


#   ******** SETUP LOADING TFRECORD ********
startr = paddingDiff[0][0]
startc = paddingDiff[1][0]
example_structure = {'highpass': tf.io.FixedLenFeature([], tf.string), 'lowpass': tf.io.FixedLenFeature([], tf.string), 'compressed': tf.io.FixedLenFeature([], tf.string),'raw': tf.io.FixedLenFeature([], tf.string)}

def restore_double(x):
    return tf.io.parse_tensor(x,real_dtype)

def _parse_image_function(example_proto):
    x = tf.io.parse_single_example(example_proto, example_structure)
    highpass = restore_double(x['highpass'])
    endr = highpass.shape[0] - paddingDiff[0][1]
    endc = highpass.shape[1] - paddingDiff[1][1]
    return ((restore_double(x['lowpass']),highpass[slice(startr,endr),slice(startc,endc),slice(None)],restore_double(x['compressed'])),restore_double(x['raw']))

raw_dataset = tf.data.TFRecordDataset([datapath + trainfile])
dataset = raw_dataset.map(_parse_image_function)
dataset_batch = dataset.batch(batch_size)

#   ******** BUILD MODEL ********
CSC = mlcsc.MultiLayerCSC(rho,alpha_init,mu_init,qY,qUV,cropAndMerge,fftSz,problem_param['D'],lraParam,noi,noL,cmplxdtype)

# Build Input Layers
highpassShape = (targetSz[0] + paddingTuple[0][0] + paddingTuple[0][1],targetSz[1] + paddingTuple[1][0] + paddingTuple[1][1],noc)
highpass = tf.keras.Input(shape=(targetSz[0] + padding))
lowpass = tf.keras.Input(shape = highpassShape,dtype = real_dtype)
compressed = tf.keras.Input(shape = (targetSz[0],targetSz[1],noc),dtype= real_dtype)
inputs = (highpass,lowpass,compressed)

reconstruction,itstats = CSC(inputs)
import post_process_grad as ppg
model = ppg.Model_PostProcess(inputs,reconstruction)

#   ******** COMPILE AND TRAIN MODEL ********
import time
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

model.compile(optimizer = tf.keras.optimizers.Adam(step_size),loss = tf.keras.losses.MSE())
model.save(experimentpath + modelfilename)
time_callback = TimeHistory()
for ii in range(num_of_saves):
    model.fit(x = dataset_batch,epochs = noe_per_save,shuffle=False,callbacks=[time_callback])
    model.save_weights(experimentpath + checkpointfilename(ii*noe_per_save))
