import tensorflow as tf
import numpy as np
import pickle as pkl
fid = open('data/scratchwork/BSDS500/patches/train/compressed/271008.jpg.pckl_12_7.pckl','rb')
cmprssdImg = pkl.load(fid)
fid.close()

print(tf.reduce_max(cmprssdImg))
print(tf.reduce_min(cmprssdImg))
