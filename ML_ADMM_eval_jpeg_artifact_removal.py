import tensorflow as tf
import jpeg_related_functions as jrf
import multilayerCSC_ADMM as mlcsc
import multilayerCSC_FISTA as mlcscf
import numpy as np
import pickle as pkl
import datetime
import util
import time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_predict_begin(self,logs={}):
        self.predict_times = []

    def on_predict_batch_begin(self,batch,logs={}):
        self.predict_batch_start_time = time.time()

    def on_predict_batch_end(self,batch,logs = {}):
        self.predict_times.append(time.time() - self.predict_batch_start_time)



def test_ADMM_CSC(rho,alpha_init,noi,databasename,steps_per_epoch,num_of_epochs):
    #   ******** ALGORITHM-SPECIFIC HYPERPARAMETERS ********
    #rho = 1.
    #lpstz = 100.
    #alpha_init = 1.5
    mu_init = 1.
    b_init = 0.1
    n_components = 4
    cmplxdtype = tf.complex128 # This should really be elsewhere.
    batch_size = 1
    #steps_per_epoch = 4
    step_size = 0.01
    #num_of_epochs = 2

    #   ******** BASE NAMES AND DIRECTORIES  ********
    modelname = 'ML_ADMM_'
    #databasename = 'BSDS500/'
    #databasename = 'simpleTest/'
    experimentname = 'experiment1/'

    #   ******** DEPENDENT NAMES AND DIRECTORIES
    experimentpath = 'data/experiment/' + databasename + experimentname
    checkpointfilename = modelname + 'checkpoint_epoch_{epoch:02d}.ckpt'
    timesname = 'times/' + modelname + 'rho' + str(rho) + '_iter' + str(noi) + '_times.pkl'
    modelfilename = modelname + 'initial_model.ckpt'

    #   ******** DATA AND EXPERIMENT PARAMETERS ********
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
    #noi = problem_param['noi']
    noL = problem_param['noL']
    noc = problem_param['noc']
    datapath = problem_param['datapath']
    trainfile = problem_param['trainfile']
    valfile = problem_param['valfile']
    padding = data_param['padding']


    #   ******** CROPPING AND PADDING ********
    cropAndMerge = mlcsc.CropPadObject(targetSz,strides,[np.asarray(ks) for ks in fltrSz],real_dtype)
    paddingTuple = cropAndMerge.paddingTuple
    fftSz = cropAndMerge.get_fft_size(targetSz,strides)
    paddingDiff = ((padding[0][0] - paddingTuple[0][0],padding[0][1] - paddingTuple[0][1]),(padding[1][0] - paddingTuple[1][0],padding[1][1] -  paddingTuple[1][1]))
    assert(paddingDiff[0][0] >= 0)
    assert(paddingDiff[0][1] >= 0)
    assert(paddingDiff[1][0] >= 0)
    assert(paddingDiff[1][1] >= 0)
    
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

    raw_dataset = tf.data.TFRecordDataset([datapath + valfile])
    dataset = raw_dataset.map(_parse_image_function)
    dataset_batch = dataset.batch(batch_size)



    #   ******** BUILD MODEL ********
    CSC = mlcsc.Wrap_ML_ADMM(rho,alpha_init,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,problem_param['D'],n_components,noi,noL,cmplxdtype)
    Get_Obj = mlcsc.Get_Obj(CSC)
    #CSC = mlcscf.Wrap_ML_FISTA(lpstz,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,problem_param['D'],noi,noL,cmplxdtype)

    # Build Input Layers
    highpassShape = (targetSz[0] + paddingTuple[0][0] + paddingTuple[0][1],targetSz[1] + paddingTuple[1][0] + paddingTuple[1][1],noc)
    highpass = tf.keras.Input(shape=highpassShape,dtype=real_dtype)
    lowpass = tf.keras.Input(shape = highpassShape,dtype = real_dtype)
    compressed = tf.keras.Input(shape = (targetSz[0],targetSz[1],noc),dtype= real_dtype)
    inputs = (highpass,lowpass,compressed)

    y,negC = CSC(inputs)
    output = Get_Obj((y,negC))
    model = tf.keras.Model(inputs,output)
    model.compile(optimizer = tf.keras.optimizers.SGD(step_size),loss = tf.keras.losses.MSE,run_eagerly=False)
    for tv in model.trainable_variables:
        print(tv.name)

    time_callback = time_callback = TimeHistory()
    outputs = model.predict(x=dataset_batch,steps=num_of_epochs*steps_per_epoch,verbose=0,callbacks = [time_callback])

    fid = open(experimentpath + timesname,'wb')
    pkl.dump(outputs,fid)
    pkl.dump(time_callback.predict_times,fid)
    fid.close()

    tf.keras.backend.clear_session()

def test_ADMM_CSC_saved_dict(rho,alpha_init,noi,databasename,steps_per_epoch,num_of_epochs,pklfile):
    #   ******** ALGORITHM-SPECIFIC HYPERPARAMETERS ********
    #rho = 1.
    #lpstz = 100.
    #alpha_init = 1.5
    #mu_init = 1.
    #b_init = 0.1
    n_components = 4
    cmplxdtype = tf.complex128 # This should really be elsewhere.
    batch_size = steps_per_epoch
    #steps_per_epoch = 4
    step_size = 0.01
    #num_of_epochs = 2

    #   ******** BASE NAMES AND DIRECTORIES  ********
    modelname = 'ML_ADMM_'
    #databasename = 'BSDS500/'
    #databasename = 'simpleTest/'
    experimentname = 'experiment1/'

    #   ******** DEPENDENT NAMES AND DIRECTORIES
    experimentpath = 'data/experiment/' + databasename + experimentname


    #   ******** DATA AND EXPERIMENT PARAMETERS ********
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
    #noi = problem_param['noi']
    noL = problem_param['noL']
    noc = problem_param['noc']
    datapath = problem_param['datapath']
    trainfile = problem_param['trainfile']
    valfile = problem_param['valfile']
    padding = data_param['padding']

    fid = open(experimentpath + pklfile,'rb')
    mu_init = pkl.load(fid)
    D = pkl.load(fid)
    b_init = pkl.load(fid)
    fid.close()
    timesname = 'times/' + pklfile + modelname + 'rho' + str(rho) + '_iter' + str(noi) + '_times.pkl'

    #   ******** CROPPING AND PADDING ********
    cropAndMerge = mlcsc.CropPadObject(targetSz,strides,[np.asarray(ks) for ks in fltrSz],real_dtype)
    paddingTuple = cropAndMerge.paddingTuple
    fftSz = cropAndMerge.get_fft_size(targetSz,strides)
    paddingDiff = ((padding[0][0] - paddingTuple[0][0],padding[0][1] - paddingTuple[0][1]),(padding[1][0] - paddingTuple[1][0],padding[1][1] -  paddingTuple[1][1]))
    assert(paddingDiff[0][0] >= 0)
    assert(paddingDiff[0][1] >= 0)
    assert(paddingDiff[1][0] >= 0)
    assert(paddingDiff[1][1] >= 0)
    
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

    raw_dataset = tf.data.TFRecordDataset([datapath + valfile])
    dataset = raw_dataset.map(_parse_image_function)
    dataset_batch = dataset.batch(batch_size)



    #   ******** BUILD MODEL ********
    #CSC = mlcsc.MultiLayerCSC(rho,alpha_init,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,problem_param['D'],n_components,noi,noL,cmplxdtype)
    CSC = mlcsc.Wrap_ML_ADMM(rho,alpha_init,mu_init,b_init,qY,qUV,cropAndMerge,fftSz,strides,[tf.reshape(D[ii],(1,) + D[ii].shape) for ii in range(len(D))],n_components,noi,noL,cmplxdtype)
    Get_Obj = mlcsc.Get_Obj(CSC_Wrap)

    # Build Input Layers
    highpassShape = (targetSz[0] + paddingTuple[0][0] + paddingTuple[0][1],targetSz[1] + paddingTuple[1][0] + paddingTuple[1][1],noc)
    highpass = tf.keras.Input(shape=highpassShape,dtype=real_dtype)
    lowpass = tf.keras.Input(shape = highpassShape,dtype = real_dtype)
    compressed = tf.keras.Input(shape = (targetSz[0],targetSz[1],noc),dtype= real_dtype)
    inputs = (highpass,lowpass,compressed)

    z,x,negC = CSC_Wrap(inputs)
    output = Get_Obj((x,negC))
    reconstruction1,reconstruction2,itstats = CSC_Wrap.admm(inputs)
    clipped_reconstruction2 = util.clip(a=0.,b = 1.,dtype=real_dtype)(reconstruction2)
    model = tf.keras.Model(inputs,output)
    model2 = tf.keras.Model(inputs,clipped_reconstruction2)
    model.compile(optimizer = tf.keras.optimizers.SGD(step_size),loss = tf.keras.losses.MSE,run_eagerly=False)
    model2.compile(optimizer = tf.keras.optimizers.SGD(step_size),loss = tf.keras.losses.MSE,run_eagerly=False)
    for tv in model.trainable_variables:
        print(tv.name)


    time_callback = TimeHistory()

    outputs = model.predict(x=dataset_batch,steps=num_of_epochs,verbose=0,callbacks = [time_callback])
    outputs2 = model2.evaluate(x= dataset_batch,steps = num_of_epochs,verbose=0)

    fid = open(experimentpath + timesname,'wb')
    pkl.dump(outputs,fid)
    pkl.dump(outputs2,fid)
    pkl.dump(time_callback.predict_times,fid)
    fid.close()

    tf.keras.backend.clear_session()
