import tensorflow as tf
import jpeg_related_functions as jrf
import ML_ADMM_2d as mlcsc
import numpy as np
import pickle as pkl
import datetime
import util

#   ******** ALGORITHM-SPECIFIC HYPERPARAMETERS ********
rho = 1.
alpha_init = 1.5
mu_init = 1.
lraParam = {'n_components': 4}
cmplxdtype = tf.complex128 # This should really be elsewhere.
batch_size = 8
noe_per_save = 1
num_of_saves = 2
step_size = 0.1


#   ******** DATA AND EXPERIMENT PARAMETERS ********
databasename = 'BSDS500/'
experimentpath = 'data/experiment/' + databasename + 'experiment1/'
def checkpointfilename(ii):
    return 'checkpoint_epoch' + str(ii) + '.ckpt'
modelfilename = 'initial_model.ckpt'
fid = open(experimentpath + 'problem_param.pckl','rb')
problem_param = pkl.load(fid)
fid.close()
data_param = problem_param['data_param']
b_init = problem_param['b_init']
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

#   ******** SETUP LOADING TFRECORD ********
startr = paddingDiff[0][0]
startc = paddingDiff[1][0]
endr = targetSz[0] + padding[0][0] + padding[0][1] - paddingDiff[0][1]
endc = targetSz[1] + padding[1][0] + padding[1][1] - paddingDiff[1][1]
example_structure = {'highpass': tf.io.FixedLenFeature([], tf.string), 'lowpass': tf.io.FixedLenFeature([], tf.string), 'compressed': tf.io.FixedLenFeature([], tf.string),'raw': tf.io.FixedLenFeature([], tf.string)}

def restore_double(x):
    return tf.io.parse_tensor(x,real_dtype)

def _parse_image_function(example_proto):
    x = tf.io.parse_single_example(example_proto, example_structure)
    highpass = restore_double(x['highpass'])
    lowpass = restore_double(x['lowpass'])
    return ((highpass[slice(startr,endr),slice(startc,endc),slice(None)],lowpass[slice(startr,endr),slice(startc,endc),slice(None)],restore_double(x['compressed'])),restore_double(x['raw']))

raw_dataset = tf.data.TFRecordDataset([datapath + trainfile])
dataset = raw_dataset.map(_parse_image_function)
dataset_batch = dataset.batch(batch_size)


for (x,y) in dataset_batch:
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)
    print(y.shape)
    print(tf.reduce_max(tf.abs(cropAndMerge.crop(x[0]) + cropAndMerge.crop(x[1]) - x[2])))
    croppedHighpass = x[0][slice(None),slice(paddingTuple[0][0], paddingTuple[0][0] + targetSz[0]),slice(paddingTuple[1][0],paddingTuple[1][0] + targetSz[1]),slice(None)]
    croppedLowpass = x[1][slice(None),slice(paddingTuple[0][0],paddingTuple[0][0] + targetSz[0]),slice(paddingTuple[1][0],paddingTuple[1][0] + targetSz[1]),slice(None)]
    print(croppedHighpass.shape)
    print(croppedLowpass.shape)
    print(x[2].shape)
    print(tf.reduce_max(tf.abs(croppedHighpass + croppedLowpass - x[2])))
    print(tf.reduce_max(x[2]))
    print(tf.reduce_min(x[2]))
    break

#   ******** BUILD MODEL ********
CSC = mlcsc.MultiLayerCSC(rho,alpha_init,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,problem_param['D'],lraParam,noi,noL,cmplxdtype)

# Build Input Layers
highpassShape = (targetSz[0] + paddingTuple[0][0] + paddingTuple[0][1],targetSz[1] + paddingTuple[1][0] + paddingTuple[1][1],noc)
highpass = tf.keras.Input(shape=highpassShape,dtype=real_dtype)
lowpass = tf.keras.Input(shape = highpassShape,dtype = real_dtype)
compressed = tf.keras.Input(shape = (targetSz[0],targetSz[1],noc),dtype= real_dtype)
inputs = (highpass,lowpass,compressed)

reconstruction,itstats = CSC(inputs)
rgb_reconstruction = jrf.YUV2RGB(dtype=real_dtype)(reconstruction)
clipped_reconstruction = util.clip(a = 0.,b = 1.,dtype=real_dtype)(rgb_reconstruction)
import post_process_grad as ppg
model = ppg.Model_PostProcess(inputs,clipped_reconstruction)

#   ******** COMPILE AND TRAIN MODEL ********
import time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_times = []

    def on_test_begin(self, logs={}):
        self.test_times = []

    def on_test_batch_begin(self, batch, logs={}):
        self.test_batch_start_time = time.time()

    def on_test_batch_end(self, batch,logs={}):
        self.test_times.append(time.time() - self.test_batch_start_time)

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.train_times.append(time.time() - self.epoch_time_start)
log_dir = "logs/logs_test/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S.log")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir = log_dir,
      histogram_freq = 1,
      profile_batch = '2,5'
)

model.compile(optimizer = tf.keras.optimizers.Adam(step_size),loss = tf.keras.losses.MSE,run_eagerly=False)
model.save_weights(experimentpath + modelfilename)
sha_name = "SHA.txt"
log_sha_command = "git log --pretty=format:'%h' -n 1 >> "
import os
os.system(log_sha_command + experimentpath + sha_name)
time_callback = TimeHistory()
for ii in range(num_of_saves):
    if ii == 1:
        with tf.profiler.experimental.Profile(log_dir):
            model.fit(x=dataset_batch,epochs= noe_per_save,steps_per_epoch=8,shuffle=False,callbacks = [time_callback,])
    else:
        model.fit(x=dataset_batch,epochs= noe_per_save,steps_per_epoch=8,shuffle=False,callbacks = [time_callback,])
    #model.evaluate(x = dataset_batch,verbose=True,steps=5,callbacks=[TimeHostory)
    model.save_weights(experimentpath + checkpointfilename(ii*noe_per_save))
