"""
    Reinforcement Learning for Pong!
    More Info : https://github.com/mlitb/pong
    Authors:
    1. Faza Fahleraz https://github.com/ffahleraz
    2. Nicholas Rianto Putra https://github.com/nicholaz99
    3. Abram Perdanaputra https://github.com/abrampers
"""

import os
import gym
import argparse
import numpy as np
import pickle as pkl
import tensorflow as tf
from model import Model

from typing import Dict, List


# Type shorthands
EpisodeBuffer = Dict[str, List]
Gradient = Dict[str, np.ndarray]


def preprocess(frame: np.ndarray) -> tf.Tensor:
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


def normal_discounted_reward(episode_buffer: EpisodeBuffer, discount_factor: float) -> float:
    """
        Calculate the normalized and discounted reward for the current episode.
    """
    reward = episode_buffer['reward']
    discounted_reward = np.zeros((len(reward), 1))
    future_reward = 0
    for i in range(len(reward) - 1, -1, -1):
        if reward[i] != 0: # reset future reward after each score
            future_reward = 0
        discounted_reward[i][0] = reward[i] + discount_factor * future_reward
        future_reward = discounted_reward[i][0]
    discounted_reward -= np.mean(discounted_reward)
    discounted_reward /= np.std(discounted_reward)
    return discounted_reward


def main(load_fname: str, save_dir: str, render: bool) -> None:
    """
        Main training loop.
    """
    model = Model((80, 80, 1))
    batch_size = 10
    input_layer_size = 6400
    hidden_layer_size_1 = 800
    hidden_layer_size_2 = 200
    learning_rate = 1e-3
    discount_factor = .99
    rmsprop_decay = .90
    rmsprop_smoothing = 1e-5

    if load_fname is not None:
        saved = pkl.load(open(load_fname, 'rb'))
        model = saved['model']
        moving_grad_rms = saved['moving_grad_rms']
        episode_number = saved['episode_number']
        print('Resuming saved model in \'{}\'.'.format(load_fname))
    else:
        model = {
            'wh1': np.random.randn(hidden_layer_size_1, input_layer_size) / np.sqrt(input_layer_size),
            'wh2': np.random.randn(hidden_layer_size_2, hidden_layer_size_1) / np.sqrt(hidden_layer_size_1),
            'wo': np.random.randn(1, hidden_layer_size_2) / np.sqrt(hidden_layer_size_2),
        }
        moving_grad_rms = {
            'wh1': np.zeros_like(model['wh1']),
            'wh2': np.zeros_like(model['wh2']),
            'wo': np.zeros_like(model['wo']),
        }
        episode_number = 0

    batch_gradient_buffer = {
        'wh1': np.zeros_like(model['wh1']),
            'wh2': np.zeros_like(model['wh2']),
            'wo': np.zeros_like(model['wo']),
    }
    batch_rewards = []

    env = gym.make('Pong-v0')
    while True:
        observation = env.reset()
        prev_frame = np.zeros(input_layer_size)
        episode_done = False
        timestep = 0
        
        episode_buffer = {
            'x': [], # input vector
            'ph1': [], # product of hidden layer
            'h1': [], # activation of hidden layer
            'ph2': [], # product of hidden layer
            'h2': [], # activation of hidden layer
            'py': [], # product of output layer
            'y': [], # activation of output layer (prob of moving up)
            'y_true': [], # fake label
            'reward': [] # rewards
        }

        while not episode_done:
            if render:
                env.render()

            # generate input vector
            frame = preprocess(observation)
            x = frame - prev_frame
            prev_frame = frame

            # forward pass
            y = model.call(x)
            action, y_true = (2, 1.0) if np.random.uniform() < y[0][0] else (5, 0.0)
            episode_buffer['y_true'].append(y_true)

            with tf.GradientTape() as tape:
                error = y_true * tf.log(y) + (1 - y_true) * tf.log(y)
            
            # perform action and get new observation
            observation, reward, episode_done, info = env.step(action)
            episode_buffer['reward'].append(reward)
            timestep += 1
            
            if episode_done:
                # backward pass
                episode_reward = normal_discounted_reward(episode_buffer, discount_factor)
                gradient = model.backward_pass(model, episode_buffer, episode_reward)
                for key in model:
                    batch_gradient_buffer[key] += gradient[key]

                # bookeeping
                batch_rewards.append(sum(episode_buffer['reward']))
                episode_number += 1

                # training info
                # print('Episode: {}, rewards: {}'.format(episode_number, sum(episode_buffer['reward'])))

                # parameter update (rmsprop)
                if episode_number % batch_size == 0:
                    for key in model:
                        moving_grad_rms[key] = rmsprop_decay * moving_grad_rms[key] + \
                                (1 - rmsprop_decay) * (batch_gradient_buffer[key] ** 2)
                        model[key] += batch_gradient_buffer[key] * learning_rate / \
                                (np.sqrt(moving_grad_rms[key]) + rmsprop_smoothing)
                        batch_gradient_buffer[key] = np.zeros_like(model[key])
                    
                    # training info
                    print('Batch: {}, avg episode rewards: {}'.format(episode_number // batch_size, 
                            sum(batch_rewards) / len(batch_rewards)))
                    batch_rewards = []

                # save model
                if episode_number % 50 == 0 and save_dir is not None:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_fname = os.path.join(save_dir, 'save_{}.pkl'.format(episode_number))
                    pkl.dump({'model': model, 'moving_grad_rms': moving_grad_rms, 
                            'episode_number': episode_number}, open(save_fname, 'wb'))
                    print('Model saved to \'{}\'!'.format(save_fname))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an RL agent to play the mighty game of Pong.')
    parser.add_argument('-l', '--load', action="store", default=None, help='path to the saved model to load from')
    parser.add_argument('-s', '--save', action="store", default=None, help='path to the folder to save model')
    parser.add_argument('-r', '--render', action="store_true", default=False, help='whether to render the environment or not')
    args = parser.parse_args()
    main(load_fname=args.load, save_dir=args.save, render=args.render)
