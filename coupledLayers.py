import numpy as np
import tensorflow as tf


class outerClass:
# This class fails because variable a is never registered to the model's trainable variables
    def __init__(self,a_init):
        self.a = tf.Variable(initial_value=a_init,trainable=True)
        self.plusa = self.PLUSA(self)
        self.timesa = self.TIMESA(self)

    class PLUSA(tf.keras.layers.Layer):
        def __init__(self,outerclass):
            super().__init__()
            self.outerclass = outerclass
        def build(self, input_shape):
            pass
        def call(self, inputs):
            return tf.math.add(self.outerclass.a,inputs)
        def output_shape(self, input_shape):
            input_shape

    class TIMESA(tf.keras.layers.Layer):
        def __init__(self,outerclass):
            super().__init__()
            self.outerclass = outerclass
        def build(self, input_shape):
            pass
        def call(self, inputs):
            return tf.math.multiply(self.outerclass.a,inputs)
        def output_shape(self, input_shape):
            input_shape

class captainLayer(tf.keras.layers.Layer):
# This class fails because when asked to list the trainable variables model enters an infinite recursive loop.
    def __init__(self,a_init):
        self.var_a = tf.Variable(initial_value=a_init,trainable=True)
        self.subLay = self.crewLayer(self)
        super().__init__()
    def call(self, inputs):
        return tf.math.add(self.var_a,inputs)
    class crewLayer(tf.keras.layers.Layer):
        def __init__(self,captain):
            self.captain=captain
            super().__init__()
        def call(self,inputs):
            return tf.math.multiply(self.captain.var_a,inputs)

class mastermindLayer(tf.keras.layers.Layer):
    def __init__(self,a_init):
        self.var_a = tf.Variable(initial_value=a_init,trainable=True)
        super().__init__()
    def call(self,inputs):
        return tf.math.add(self.var_a,inputs)

class henchmanLayer(tf.keras.layers.Layer):
    def __init__(self,mastermind):
        self.mastermind = mastermind
        super().__init__()
    def call(self,inputs):
        return tf.math.multiply(self.mastermind.var_a,inputs)

class landlordLayer(tf.keras.layers.Layer):
    def __init__(self,a_init):
        self.var_a = tf.Variable(initial_value=a_init,trainable=True)
        super().__init__()
    def call(self,inputs):
        return tf.math.add(self.var_a,inputs)

class tenantLayer(tf.keras.layers.Layer):
    def __init__(self,a):
        self.a = a
        super().__init__()
    def call(self,inputs):
        return tf.math.multiply(self.a,inputs)

