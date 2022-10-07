import sys
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras as keras

class PrintEpochNoBasic(tf.keras.callbacks.Callback):
    def __init__(self, print_freq, **kwargs):
        self.print_freq = print_freq
    def on_epoch_end(self, epoch, logs={}):
        if  (epoch+1)%self.print_freq == 0:
            tf.print("Epoch no " + str(epoch+1)+ " loss " + str(logs.get('loss'))  + " val_loss " + str(logs.get('val_loss')) + " error " + str(logs.get('error'))+  " val_error " + str(logs.get('val_error')), output_stream=sys.stdout)

class MAE(tf.keras.metrics.Metric):
    def __init__(self, name='mae', **kwargs):
        super(MAE, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)

        err = tf.math.abs(y_pred-y_true)

        self.accuracy.assign_add(tf.reduce_mean(tf.cast(err, tf.float64)))
        self.count.assign_add(1)

    def result(self):
      return self.accuracy/self.count

    def reset_states(self):
      self.accuracy.assign(0.)
      self.count.assign(0.)


def mse(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.cast(y_true, tf.float64)
    return  K.mean((K.sum(K.square(y_true-y_pred), axis=-1)))

