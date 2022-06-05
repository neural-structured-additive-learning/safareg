import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
import tensorflow.keras.regularizers as regularizers

### Helper functions matrix algebra
def tf_crossprod(a,b):
    return(tf.matmul(tf.transpose(a), b))
def tf_incross(a,b):
    return(tf.matmul(tf_crossprod(a,b),a))
def tf_unitvec(n,j):
    return(tf.transpose(tf.eye(n)[None,j,:]))
def tf_operator_multiply(scalar, operator):
    return(operator.matmul(tf.linalg.diag(tf.repeat(scalar, operator.shape[0]))))
def vecmatvec(a, B, c=None, sparse_mat = False):
    if c is None:
        c = a
    #return(tf.matmul(tf.transpose(a),tf.linalg.matvec(B, tf.squeeze(c, [1]), a_is_sparse = sparse_mat)))
    return(tf.keras.backend.sum(tf.keras.backend.batch_dot(a, tf.keras.backend.dot(B, c), axes=1)))


def layer_splineVC(P, units, name):
    return(tf.keras.layers.Dense(units = units, name = name, use_bias=False, kernel_regularizer = squaredPenaltyVC(P, 1)))

class squaredPenaltyVC(regularizers.Regularizer):

    def __init__(self, P, strength, nlev):
        self.strength = strength
        self.P = tf.cast(P, dtype="float32")
        self.nlev = nlev

    def __call__(self, x):
        x_splitted = tf.split(x, self.nlev)
        pen = 0
        for x_k in x_splitted:
            pen += tf.reduce_sum(vecmatvec(x_k, self.P, sparse_mat = True))
        return self.strength * pen

    def get_config(self):
        return {'strength': self.strength, 'P': self.P}


class LinearArrayRWT(keras.layers.Layer):
    def __init__(self, units=(1, 1), P=None, name=None):
        super(LinearArrayRWT, self).__init__()
        self.units = units
        self.P = tf.cast(P, dtype="float32")
        self._name = name

    def build(self, input_shape):
        self.w = self.add_weight(
            shape = self.units,
            initializer="random_normal",
            trainable=True,
        )


    def call(self, inputs):
        self.add_loss(tf.reduce_sum(tf.multiply(self.w, tf.matmul(self.P, self.w))))
        return tf.reduce_sum(tf.multiply(tf.matmul(inputs[0], self.w), inputs[1]), 1)

def tf_row_tensor_left_part(a,b):
    return tf_repeat(a, b$shape[1])

def tf_row_tensor_right_part(a,b):
    return tf$tile(b, c(1, a$shape[1]))

def tf_row_tensor(a,b):
    return tf.multiply(tf_row_tensor_left_part(a,b), tf_row_tensor_right_part(a,b))

class OneHotFac(keras.layers.Layer):
    def __init__(self, lev, **kwargs):
        super(OneHotFac, self).__init__(**kwargs)
        self.lev = lev
 
    def call(self, input):
        return tf.squeeze(tf.one_hot(tf.cast(x, dtype="int32"), depth = lev), axis=1)
        
class OneHotIA(keras.layers.Layer):
    def __init__(self, xlev, ylev, **kwargs):
        super(OneHotFac, self).__init__(**kwargs)
        self.xlev = xlev
        self.ylev = ylev
 
    def call(self, input):
        y = x[:,1]
        x = x[:,0]
        xoh = OneHotFac(xlev)(x)
        yoh = OneHotFac(ylev)(y)
        return tf_row_tensor(xoh, yoh)
        
