"""
    An implementation of a policy gradient agent that learns to play the 
    mighty game of Pong.
    
    More Info : https://github.com/mlitb/pong-cnn
    Authors:
        1. Faza Fahleraz https://github.com/ffahleraz
        2. Nicholas Rianto Putra https://github.com/nicholaz99
        3. Abram Perdanaputra https://github.com/abrampers
"""

import gym
import argparse
import numpy as np
import tensorflow as tf
from model import Model

from typing import Dict, List

tf.enable_eager_execution()


# Type shorthands
EpisodeBuffer = Dict[str, List]
Gradient = Dict[str, np.ndarray]


# Training hyperparameters
BATCH_SIZE = 10


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
        Preprocess 210x160x3 uint8 frame into 1x80x80x1 4D float32 tensor.
    """
    frame = frame[35:195]
    frame = frame[::2,::2,0]
    frame[(frame == 144) | (frame == 109)] = 0
    frame[frame != 0] = 1
    frame = np.expand_dims(frame, 0)
    frame = np.expand_dims(frame, -1)
    return tf.convert_to_tensor(frame, dtype=tf.float32)


def main(render: bool = False):
    """
        Main training loop.
    """
    policy = Model((80, 80, 1))
    optimizer = tf.train.RMSPropOptimizer(1e-3)
    env = gym.make('Pong-v0')
    episode_count = 0

    while True:
        observation = env.reset()
        prev_frame = tf.zeros((1, 80, 80, 1), dtype=tf.float32)
        episode_done = False
        episode_buffer = {
            'gradient': [],
            'reward': []
        }

        while not episode_done:
            if render:
                env.render()

            # preprocess input
            frame = preprocess(observation)
            x = frame - prev_frame
            prev_frame = frame

            # forward pass
            with tf.GradientTape() as tape:
                y = policy.call(x)
                action, y_true = (2, 1.0) if np.random.uniform() < y[0][0] else (5, 0.0)
                error = y_true * tf.log(y) + (1 - y_true) * tf.log(y)
                loss_value = tf.reduce_mean(error)
            gradient = tape.gradient(loss_value, policy.variables)
            episode_buffer['gradient'].append(gradient)
            
            # perform action and get new observation
            observation, reward, episode_done, info = env.step(action)
            episode_buffer['reward'].append(reward)

            if episode_done:
                print('Episode: {}'.format(episode_count))
                episode_count += 1

                # optimizer.apply_gradients(zip(gradients, policy.variables),
                #         global_step=tf.train.get_or_create_global_step())

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an RL agent to play the mighty game of Pong.')
    parser.add_argument('-r', '--render', action="store_true", default=False, help='whether to render the environment or not')
    args = parser.parse_args()
    main(render=args.render)
    