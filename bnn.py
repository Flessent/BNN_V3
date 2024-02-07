from larq.layers import QuantDense
import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers, constraints, initializers, models
import numpy as np
from tensorflow.keras.initializers import RandomNormal

#from activations import*
from tensorflow.keras.models import Sequential

import larq as lq
from keras.layers import Dense, Activation, BatchNormalization,Dropout,InputLayer,Lambda

class Binarizer(layers.Layer):
    def __init__(self, **kwargs):
        super(Binarizer, self).__init__(**kwargs)

    def binarize(self, inputs):
        return tf.where(inputs < 0, -1.0, 1.0)

    def call(self, inputs):
        return self.binarize(inputs)

def describe_network(model):
    for layer in model.layers:
        if isinstance(layer, QuantDense):
            print(f'for QuantDense Layer  weights are  : {layer.get_weights()[0]} shape : {len(layer.get_weights()[0])}')
        
       # else:
        #    print(f'Unknown layer type: {type(layer)}')
"""class BooleanActivation(layers.Layer):
    def __init__(self, units, **kwargs):
        super(BooleanActivation, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)
        super(BooleanActivation, self).build(input_shape)

    def call(self, inputs):
        print('BooleanActivation is called...')
        tau = self.bias + 0.5 * tf.reduce_sum(self.kernel, axis=0)
        theta_i_o_1 = 1 / (1 + tf.exp(-0.5 * self.kernel))
        theta_i_o_0 = 1 / (1 + tf.exp(0.5 * self.kernel))

        prob_o_1 = tf.exp(tau) / (1 + tf.exp(tau))
        prob_i_1_o_1 = tf.reduce_prod(theta_i_o_1 * inputs, axis=1)
        prob_i_1_o_0 = tf.reduce_prod(theta_i_o_0 * (1 - inputs), axis=1)

        probability_o_1_given_x = prob_o_1 * prob_i_1_o_1 / (prob_o_1 * prob_i_1_o_1 + (1 - prob_o_1) * prob_i_1_o_0)
        return tf.where(probability_o_1_given_x >= 0.5, 1, 0)"""
class BooleanActivation(layers.Layer):
    def __init__(self, units, **kwargs):
        super(BooleanActivation, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)
        super(BooleanActivation, self).build(input_shape)

    def call(self, inputs):
        tau = self.bias + 0.5 * tf.reduce_sum(self.kernel, axis=0)
        theta_i_o_1 = 1 / (1 + tf.exp(-0.5 * self.kernel))
        theta_i_o_0 = 1 / (1 + tf.exp(0.5 * self.kernel))

        prob_o_1 = tf.exp(tau) / (1 + tf.exp(tau))
        prob_i_1_o_1 = tf.reduce_prod(theta_i_o_1 * inputs + (1 - theta_i_o_1) * (1 - inputs), axis=1)
        prob_i_1_o_0 = tf.reduce_prod(theta_i_o_0 * inputs + (1 - theta_i_o_0) * (1 - inputs), axis=1)

        probability_o_1_given_x = prob_i_1_o_1 * prob_o_1 / (prob_i_1_o_1 * prob_o_1 + prob_i_1_o_0 * (1 - prob_o_1))
        print('shape return BooleanActivation :',tf.expand_dims(tf.where(probability_o_1_given_x >= 0.5, 1.0, 0.0), axis=1).shape)
        return tf.expand_dims(tf.where(probability_o_1_given_x >= 0.5, 1.0, 0.0), axis=1)


class BNN(models.Sequential):
    def __init__(self, input_dim, output_dim, num_neuron_in_hidden_dense_layer=18, num_neuron_output_layer=16):
        super(BNN, self).__init__()

        kwargs = dict(
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=True
        )
        print('num_neuron_in_hidden_dense_layer :',num_neuron_in_hidden_dense_layer, 'num_neuron_output_layer :',num_neuron_output_layer)
        
        self.add(QuantDense(num_neuron_in_hidden_dense_layer, input_dim=18, kernel_quantizer="ste_sign", kernel_constraint="weight_clip", use_bias=True)),
        self.add(BatchNormalization(momentum=0.999, scale=False)),
        self.add(QuantDense(num_neuron_in_hidden_dense_layer, **kwargs)),
        self.add(BatchNormalization(momentum=0.999, scale=False)),
        self.add(QuantDense(num_neuron_in_hidden_dense_layer, **kwargs)),
        self.add(BatchNormalization(momentum=0.999, scale=False)),
        self.add(QuantDense(num_neuron_in_hidden_dense_layer, **kwargs)),
        self.add(BatchNormalization(momentum=0.999, scale=False)),
        self.add(QuantDense(num_neuron_output_layer, activation='sigmoid', **kwargs))