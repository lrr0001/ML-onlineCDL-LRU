import tensorflow as tf
x = tf.random.uniform(shape=(3,4))
y = tf.random.uniform(shape=(4,3))

def myFun(ii):
    return ((tf.cast(ii,'float32')*x,(tf.cast(ii,'float32') + 1.)*x),tf.cast(ii,'float32')*y)
