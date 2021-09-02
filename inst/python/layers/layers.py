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
        self.P = P
        self.nlev = nlev

    def __call__(self, x):
        x_splitted = tf.split(x, nlev)
        pen = 0
        for x_k in x_splitted:
            pen += tf.reduce_sum(vecmatvec(x_k, tf.cast(self.P, dtype="float32"), sparse_mat = True))
        return self.strength * pen

    def get_config(self):
        return {'strength': self.strength, 'P': self.P}
    
