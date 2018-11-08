"""
    A policy gradient agent with convolutional neural network.

    More Info : https://github.com/mlitb/pong-cnn
    Authors:
        1. Faza Fahleraz https://github.com/ffahleraz
        2. Nicholas Rianto Putra https://github.com/nicholaz99
        3. Abram Perdanaputra https://github.com/abrampers
"""
import numpy as np
import tensorflow as tf
from typing import Tuple

class Memory:
    """TODO: Docs"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    
    def store(self, state: tf.Tensor, action: int, reward: int):
        """Add state, action, reward per timeframe."""
        self.states.append(state)
        self.actions.append(action)
        if reward is not None:
            self.rewards.append(reward)


    def clear(self):
        """Clear the memory."""
        self.states = []
        self.actions = []
        self.rewards = []


class Agent:
    """TODO: Docs"""
    def __init__(self, input_shape: Tuple[int, int, int]):
        self.convolution_1 = tf.keras.layers.Conv2D(input_shape=input_shape, filters=16, 
                kernel_size=8, strides=4, activation='relu', data_format="channels_last")
        self.convolution_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, 
                activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fully_connected = tf.keras.layers.Dense(256, activation='relu')
        self.probability = tf.keras.layers.Dense(1, activation='sigmoid')


    def forward_pass(self, state: tf.Tensor) -> tf.Tensor:
        """Compute the probability distribution of actions according to the policy."""
        l1 = self.convolution_1(state)
        l2 = self.convolution_2(l1)
        l3 = self.flatten(l2)
        l4 = self.fully_connected(l3)
        l5 = self.probability(l4)
        return l5


    def sample_action(self, state: tf.Tensor) -> tf.Tensor:
        """Sample action according to the policy from a given state."""
        prob = self._forward_pass(state)
        if np.random.uniform() < prob[0][0]:

        return 2 if np.random.uniform() < prob[0][0] else 5


    def register(self, reward: float, reset_discounted_reward: bool, episode_done: bool):
        """TODO: Docs"""
        pass


    def update_policy(self):
        """TODO: Docs"""
        pass
