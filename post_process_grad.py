import tensorflow as tf

class PostProcess:
    update = {}
    def add_update(varName,update_fun):
        assert varName not in PostProcess.update, "Update function already exists for %r; variable name must be unique." % varName
        PostProcess.update[varName] = update_fun


class Model_PostProcess(tf.keras.Model):
    def train_step(self,data):
        myoutputs = tf.keras.Model.train_step(self,data)
        update_ops = []
        for tv in self.trainable_variables:
            if tv.name in PostProcess.update:
                update_ops += PostProcess.update[tv.name]()
        with tf.control_dependencies(update_ops):
            return myoutputs

class Model_record_grad(tf.keras.Model):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gradient_record = []
    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        This method is called by `Model.make_train_function`.
        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Arguments:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        data = tf.python.keras.engine.data_adapter.expand_1d(data)
        x, y, sample_weight = tf.python.keras.engine.data_adapter.unpack_x_y_sample_weight(data)

        with tf.python.eager.backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
        y, y_pred, sample_weight, regularization_losses=self.losses)
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    #    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        self.gradient_record.append(gradients)
        return {m.name: m.result() for m in self.metrics}

class Model_passenger(tf.keras.Model):
    def __init__(self,gradInstructions,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gradient_record = []
        self.iter = 0
        self.gradInstructions = gradInstructions
    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        This method is called by `Model.make_train_function`.
        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Arguments:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        data = tf.python.keras.engine.data_adapter.expand_1d(data)
        x, y, sample_weight = tf.python.keras.engine.data_adapter.unpack_x_y_sample_weight(data)

        with tf.python.eager.backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
        y, y_pred, sample_weight, regularization_losses=self.losses)
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(self.gradInstructions[self.iter], trainable_variables))
    #    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        self.gradient_record.append(gradients)
        self.iter +=1
        return {m.name: m.result() for m in self.metrics}
