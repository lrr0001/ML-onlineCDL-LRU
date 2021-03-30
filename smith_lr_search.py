import matplotlib.pyplot as plt
import tensorflow as tf


class LearningRateFinder(tf.keras.callbacks.Callback):
    
    ''' References:
            https://www.jeremyjordan.me/nn-learning-rate
            https://arxiv.org/abs/1506.01186
    '''
    
    def __init__(self, lrMin=0.005, lrMax=50, steps=128):
        super().__init__()
        
        self.lrMin = lrMin
        self.lrMax = lrMax
        self.total_iterations = steps
        self.iteration = 0
        self.history = {}
        
    def get_learning_rate(self):
        return self.lrMin*(self.lrMax/self.lrMin)**(self.iteration/self.total_iterations)
        
    def on_train_begin(self, logs=None):
        logs = logs or {}
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lrMin)
        
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('iterations', []).append(self.iteration)
        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))
        

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        tf.keras.backend.set_value(self.model.optimizer.lr, self.get_learning_rate())

    def output_summary(self):
        return {'iterations': self.history['iterations'],'learning_rate': self.history['lr'],'loss': self.history['loss']}
 
    #def plot_lr(self):
    #    '''Visualize learning rate schedule.'''
    #    plt.plot(self.history['iterations'], self.history['lr'])
    #    plt.yscale('log')
    #    plt.xlabel('Iteration')
    #    plt.ylabel('Learning rate')
    #    plt.show()
        
    #def plot_loss(self):
    #    '''Plot loss vs learning rate.'''
    #    plt.plot(self.history['lr'], self.history['loss'])
    #    plt.xscale('log')
    #    plt.xlabel('Learning rate')
    #    plt.ylabel('Loss')
    #    plt.show()
