import tensorflow as tf

class FlipGrad(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def build(self,input_shape):
        pass
    def call(self,inputs):
       return flip_grad(inputs)

class FlipGrad2(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        @tf.custom_gradient
        def flip_grad2(x):
            def grad(dy):
                return -dy
            return tf.identity(x), grad
        super().__init__(*args,**kwargs)
        self.f = lambda x: flip_grad2(x)
    def call(self,inputs):
        return self.f(inputs)

class FlipGrad3(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        @tf.custom_gradient
        def flip_grad3(x):
            def grad(dy):
                return None
            return tf.identity(x), grad
        super().__init__(*args,**kwargs)
        self.f = lambda x: flip_grad3(x)
    def call(self,inputs):
        return self.f(inputs)



@tf.custom_gradient
def flip_grad(x):
    def grad(dy):
        return -dy
    return tf.identity(x), grad


def custom_loss(y_actual, y_pred):
    return tf.math.reduce_sum(1/((y_actual - y_pred - 5)**2 + 3) - 1/((y_actual - y_pred + 5)**2 + 3))

def custom_loss1(y_actual, y_pred):
    return tf.math.reduce_sum(1/((y_actual - y_pred)**2 + 3) - 1/((y_actual - y_pred + 10)**2 + 3))

def custom_loss2(y_actual, y_pred):
    return tf.math.reduce_sum(1/((y_actual - y_pred - 10)**2 + 3) - 1/((y_actual - y_pred)**2 + 3))
