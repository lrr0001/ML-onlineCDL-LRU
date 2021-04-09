import tensorflow as tf

class PostProcess:
    update = {}
    def add_update(varName,update_fun):
        assert varName not in PostProcess.update, "Update function already exists for %r; variable name must be unique." % varName
        PostProcess.update[varName] = update_fun


class CondPostProcess:
    cond = {}
    condupdate = {}
    def add_cupdate(varName,ccheck,cupdate):
        assert varName not in CondPostProcess.cond, "Conditional update function already exists for %r; variable name must be unique." % varName
        CondPostProcess.cond[varName] = ccheck
        CondPostProcess.condupdate[varName] = cupdate

class StateSaveProcess:
    save_state = {}
    def add_save(varName,save_fun):
        assert varName not in StateSaveProcess.save_state, "Save state function already exists for %r; name must be unique." % varName
        StateSaveProcess.save_state[varName] = save_fun

class PostProcessCallback(tf.keras.callbacks.Callback,PostProcess):
    def __init__(self):
        super().__init__()
        self.history = {}


    def on_train_begin(self,logs):
        logs = logs or {}
        self.updates = []
        for tv in self.model.trainable_variables:
            if tv.name in PostProcess.update:
                self.updates.append(PostProcess.update[tv.name])

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}

        for an_update in self.updates:
            an_update()

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

class DriftTracker(tf.keras.callbacks.Callback,CondPostProcess):
    def __init__(self,eps=5e-5)#,savestuff=False):
        super().__init__()
        
        self.itrtn = 0
        #self.lastreset = -10
        #self.saveCounter = 0
        #self.savestuff = savestuff
        self.drift_eps = eps
        self.history = {}


    def on_train_begin(self,logs):
        logs = logs or {}
        self.condupdate_keys = []
        for tv in self.model.trainable_variables:
            if tv.name in CondPostProcess.cond:
                self.condupdate_keys.append(tv.name)
        
        
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.itrtn += 1
        self.history.setdefault('itrtns', []).append(self.itrtn)
        for tvname in self.condupdate_keys:
            drift = CondPostProcess.cond[tvname]()
            self.history.setdefault('drift_' + tvname,[]).append(drift)
            if drift > self.drift_eps:
                CondPostProcess.condupdate[tvname]()
                #if self.lastreset + 1 == self.itrtn and self.savestuff and self.saveCounter < 10:
                #    print(drift)
                #    self.model.save_weights('iter_' + str(self.itrtn) + '_weights.ckpt')
                #    self.saveCounter = self.saveCounter + 1
                #self.lastreset = self.itrtn
            drift = CondPostProcess.cond[tvname]()
            self.history.setdefault('afterdrift_' + tvname,[]).append(drift)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def output_summary(self):
        output = {'iterations': self.history['itrtns']}
        for tvname in self.condupdate_keys:
            output['drift_' + tvname] = self.history['drift_' + tvname]
            output['afterdrift_' + tvname] = self.history['afterdrift_' + tvname]
        return output



class StateTracker(tf.keras.callbacks.Callback,PostProcess,StateSaveProcess):
    def __init__(self):
        super().__init__()
        
        self.trtn = 0
        self.history = {}


    def on_train_begin(self,logs):
        logs = logs or {}
        self.update_keys = []
        for tv in self.model.trainable_variables:
            if tv.name in PostProcess.update:
                self.update_keys.append(tv.name)
        
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trtn += 1
        self.history.setdefault('trtns', []).append(self.trtn)
        for statesavename in StateSaveProcess.save_state:
            self.history.setdefault('state_' + statesavename,[]).append(StateSaveProcess.save_state[statesavename]())

        for tvname in self.update_keys:
            PostProcess.update[tvname]()

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def output_summary(self):
        output = {'iterations': self.history['trtns']}
        for tvname in StateSaveProcess.save_state:
            output['state_' + tvname] = self.history['state_' + tvname]
        return output


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
