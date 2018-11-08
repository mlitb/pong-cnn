"""
    A policy gradient agent with convolutional neural network.

    More Info: https://github.com/mlitb/pong-cnn
    Authors:
        1. Faza Fahleraz https://github.com/ffahleraz
        2. Nicholas Rianto Putra https://github.com/nicholaz99
        3. Abram Perdanaputra https://github.com/abrampers
"""


import numpy as np
import tensorflow as tf
from typing import Tuple


class Model(tf.keras.Model):
    """TODO: Docs"""
    def __init__(self, input_shape: Tuple[int, int, int]):
        self.convolution_1 = tf.keras.layers.Conv2D(input_shape=input_shape, filters=16, 
                kernel_size=8, strides=4, activation='relu', data_format="channels_last")
        self.convolution_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, 
                activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fully_connected = tf.keras.layers.Dense(256, activation='relu')
        self.move_probability = tf.keras.layers.Dense(1, activation='sigmoid')


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute the probability distribution of actions according to the policy."""
        l1 = self.convolution_1(state)
        l2 = self.convolution_2(l1)
        l3 = self.flatten(l2)
        l4 = self.fully_connected(l3)
        l5 = self.move_probability(l4)
        return l5
