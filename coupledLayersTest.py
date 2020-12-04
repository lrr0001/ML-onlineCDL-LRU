import tensorflow as tf
import numpy as np
import coupledLayers as cL
inputs = tf.keras.layers.Input(shape=(1,))
mm = cL.mastermindLayer(-1.)
hench = cL.henchmanLayer(mm)
x1 = mm(inputs)
outputs = hench(x1)
model = tf.keras.Model(inputs,outputs)
model.compile(optimizer=tf.keras.optimizers.SGD(0.00005),loss=tf.keras.losses.MeanSquaredError())
x = np.random.randn(1000,1)
y = -2*x + 4
xval = np.random.randn(100,1)
yval = -2*xval + 4
model.fit(x=x,y=y,batch_size=10,epochs=200,validation_data=(xval,yval))

