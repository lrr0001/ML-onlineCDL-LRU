import tensorflow as tf
import numpy as np
import pickle as pkl
fid = open('data/scratchwork/BSDS500/patches/train/compressed/271008.jpg.pckl_12_7.pckl','rb')
cmprssdImg = pkl.load(fid)
fid.close()

print(tf.reduce_max(cmprssdImg))
print(tf.reduce_min(cmprssdImg))


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

dataset_iterator = dataset_batch.__iter__()

(x,y) = dataset_iterator.get_next()
print(tf.reduce_max(x[2]))
print(tf.reduce_min(x[2]))

(x,y) = dataset_iterator.get_next()
print(tf.reduce_max(x[2]))
print(tf.reduce_min(x[2]))

(x,y) = dataset_iterator.get_next()
print(tf.reduce_max(x[2]))
print(tf.reduce_min(x[2]))

(x,y) = dataset_iterator.get_next()
print(tf.reduce_max(x[2]))
print(tf.reduce_min(x[2]))


(x,y) = dataset_iterator.get_next()
print(tf.reduce_max(x[2]))
print(tf.reduce_min(x[2]))


(x,y) = dataset_iterator.get_next()
print(tf.reduce_max(x[2]))
print(tf.reduce_min(x[2]))


(x,y) = dataset_iterator.get_next()
print(tf.reduce_max(x[2]))
print(tf.reduce_min(x[2]))


(x,y) = dataset_iterator.get_next()
print(tf.reduce_max(x[2]))
print(tf.reduce_min(x[2]))

