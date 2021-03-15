import tensorflow as tf
dtype = 'float64'


example_structure = {'highpass': tf.io.FixedLenFeature([], tf.string), 'lowpass': tf.io.FixedLenFeature([], tf.string), 'compressed': tf.io.FixedLenFeature([], tf.string),'raw': tf.io.FixedLenFeature([], tf.string)}

def restore_double(x):
    return tf.io.parse_tensor(x,'float64')

def _parse_image_function(example_proto):
    x = tf.io.parse_single_example(example_proto, example_structure)
    return ((restore_double(x['lowpass']),restore_double(x['highpass']),restore_double(x['compressed'])),restore_double(x['raw']))


filenames = ['data/processed/simpleTest/val/data.tfrecord']
raw_dataset = tf.data.TFRecordDataset(filenames)
dataset = raw_dataset.map(_parse_image_function)
dataset_batch = dataset.batch(8)
for ii in dataset_batch:
    hlc,raw = ii
    lowpass,highpass,compressed = hlc
    print('highpass',highpass.shape)
    print('lowpass',lowpass.shape)
    print('compressed',compressed.shape)
    print('raw',raw.shape)
