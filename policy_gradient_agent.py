"""
    Implementation of Policy gradient model with Convolutional Neural Network.
    Authors:
    1. Faza Fahleraz https://github.com/ffahleraz
    2. Nicholas Rianto Putra https://github.com/nicholaz99
    3. Abram Perdanaputra https://github.com/abrampers
"""

import tensorflow as tf
import numpy as np
from typing import Tuple

class PolicyGradientAgent:
    def __init__(self, input_shape: Tuple[int, int]):
        self.conv1 = tf.keras.layers.Conv2D(input_shape=input_shape, filters=16, kernel_size=8, strides=4, activation='relu', data_format="channels_last") # 16 8x8 conv, stride 4
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation='relu') # 32 4x4 conv, stride 2
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(256, activation='relu') # 256
        self.val = tf.keras.layers.Dense(1, activation='sigmoid') # Last layer

    
    def _forward_pass(self, state: tf.Tensor) -> tf.Tensor:
        """Forward pass."""
        l1 = self.conv1(state)
        l2 = self.conv2(l1)
        l3 = self.flatten(l2)
        l4 = self.fc(l3)
        l5 = self.val(l4)
        return l5

    def sample_action(self, state: tf.Tensor) -> tf.Tensor:
        """Sampling action from a given state."""
        prob = self._forward_pass(state)
        return 2 if np.random.uniform() < prob[0][0] else 5
        
        