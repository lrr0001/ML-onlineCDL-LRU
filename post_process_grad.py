import tensorflow as tf

class PostProcess:
    update = {}

class Model_PostProcess(tf.keras.Model):
    def train_step(self,data):
        myoutputs = tf.keras.Model.train_step(self,data)
        for tv in self.trainable_variables:
            if tv.name in PostProcess.update:
                PostProcess.update[tv.name]()
        return myoutputs
