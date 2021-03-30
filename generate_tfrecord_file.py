import tensorflow as tf
import random
import pickle as pkl
import numpy as np
datatype = 'float64'
def convert_to_binary_str(input_array):
    tensor = tf.convert_to_tensor(input_array)
    return tf.io.serialize_tensor(tensor)

def convert_from_binary_str(binary_str,dtype):
    tf.io.parse_tensor(binary_str,dtype)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))

def serialize_example(highpass, lowpass, compressed, raw):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'highpass': _bytes_feature(convert_to_binary_str(highpass)),
        'lowpass': _bytes_feature(convert_to_binary_str(lowpass)),
        'compressed': _bytes_feature(convert_to_binary_str(compressed)),
        'raw': _bytes_feature(convert_to_binary_str(raw)),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

import os
for datatype in ['train','val']:
    datasetname = 'BSDS500/'
    datapath = 'data/scratchwork/' + datasetname + 'patches/' + datatype + '/'
    recordpath = 'data/processed/' + datasetname
    recordname = datatype + '.tfrecord'
    writer = tf.io.TFRecordWriter(recordpath + recordname)


    filelist = os.listdir(datapath + 'raw/')
    random.shuffle(filelist)
    for filename in filelist:
        raw_fid = open(datapath + 'raw/' + filename,'rb')
        raw = pkl.load(raw_fid)
        raw_fid.close()
        highpass_fid = open(datapath + 'highpass/' + filename,'rb')
        highpass = pkl.load(highpass_fid)
        highpass_fid.close()
        lowpass_fid = open(datapath + 'lowpass/' + filename,'rb')
        lowpass = pkl.load(lowpass_fid)
        lowpass_fid.close()
        compressed_fid = open(datapath + 'compressed/' + filename, 'rb')
        compressed = pkl.load(compressed_fid)
        compressed_fid.close()
        writer.write(serialize_example(highpass,lowpass,compressed,raw))
    writer.close()

