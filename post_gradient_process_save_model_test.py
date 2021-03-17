import tensorflow as tf
def custom_loss(y_actual, y_pred):
    pred_rank = tf.rank(y_pred)
    if pred_rank > 1:
        return tf.math.reduce_sum(1/((y_actual - y_pred - 1)**2 + 0.6) - 1/((y_actual - y_pred + 1)**2 + 0.6),axis=tf.range(1,tf.rank(y_pred)))
    else:
        return 1/((y_actual - y_pred - 1)**2 + 0.6) - 1/((y_actual - y_pred + 1)**2 + 0.6)

import tensorflow as tf
import numpy as np

class PostProcess:
    update = {}
    def add_update(varName,update_fun):
        assert varName not in PostProcess.update, "Update function already exists for %r; may be duplicate" % varName
        PostProcess.update[varName] = update_fun


class AddVar(tf.keras.layers.Layer,PostProcess):
    def __init__(*args,**kwargs):
        tf.keras.layers.Layer.__init__(*args,**kwargs)
    def build(self,input_shape):
        self.bias = tf.Variable(initial_value=tf.zeros(input_shape[-1]),trainable=False,dtype=self.dtype)
        self.bias2 = tf.Variable(initial_value=tf.identity(self.bias),trainable=True,dtype=self.dtype)
        PostProcess.add_update(self.bias2.name, self.post_update)
    def call(self,inputs):
        return inputs + self.bias2
    def post_update(self):

        theupdate = self.bias2 - self.bias
        # If "=" is used for assignment, differentiation will fail (in eager execution) or throw an error.
        # The assign operator is required.        
        newBias2 = self.bias2.assign(self.bias - theupdate)
        return [self.bias.assign(newBias2)]

class CustomTrainModel(tf.keras.Model):
    def train_step(self,data):
        myoutputs = tf.keras.Model.train_step(self,data)
        for tv in self.trainable_variables:
            if tv.name in PostProcess.update:
                PostProcess.update[tv.name]()
        return myoutputs

inputs = tf.keras.layers.Input(shape=(1,))
outputs = AddVar()(inputs)
model = CustomTrainModel(inputs,outputs)
model.compile(loss = custom_loss,run_eagerly=False)
x = np.random.randn(1000,1)
y = x
xval = np.random.randn(100,1)
yval = xval
model.fit(x=x,y=y,batch_size=10,epochs=16,validation_data = (xval,yval))
print(model.trainable_variables)
model.save("/home/lrr/STORAGE/Research/Phase 2/data/test_save_output.pb")
print('model should be saved now!')




