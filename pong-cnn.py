"""
    Reinforcement Learning for Pong!
    More Info : https://github.com/mlitb/pong-cnn
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

from typing import Dict, List


# Type shorthands
Model = Dict[np.ndarray, np.ndarray]
EpisodeBuffer = Dict[str, List]
Gradient = Dict[str, np.ndarray]


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
        Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector.
    """
    frame = frame[35:195] # crop
    frame = frame[::2,::2,0] # downsample by factor of 2
    frame[frame == 144] = 0 # erase background (background type 1)
    frame[frame == 109] = 0 # erase background (background type 2) 
    frame[frame != 0] = 1 # everything else (paddles, ball) just set to 1
    return frame.astype(np.float).ravel()


def relu(x: np.ndarray) -> np.ndarray:
    x[x < 0] = 0
    return x


def relu_prime(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)[x > 0] = 1
    return y


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def conv_layer(input: tf.ndarray, num_input_channels: int, filter_size: int, num_filters: int, name: str) -> (tf.nn.conv2d, tf.Variable):
    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # Add the biases to the results of the convolution.
        layer += biases

        # Relu
        layer = tf.nn.relu(layer)

        return layer, weights


def pool_layer(input: tf.ndarray, name: str) -> tf.nn.max_pool:
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        return layer


def fully_connected_layer(input: tf.ndarray, num_inputs: int, num_outputs: int, name: str):
    with tf.variable_scope(name) as scope:
        # Create new weights and biases.
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        
        # Multiply the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases
        
        return layer

def convnet():
    # Hyperparameters
    x = 
    num_input_channels = 
    filter_size = 
    num_filters = 

    layer_conv1, weights_conv1 = conv_layer(x, num_input_channels, filter_size, num_filters, 'conv1')



def forward(x: np.ndarray, model: Model, episode_buffer: EpisodeBuffer) -> float:
    """
        Do a forward pass to get the probability of moving the paddle up.
    """
    ph1 = np.dot(model['wh1'], x)
    h1 = relu(ph1)
    ph2 = np.dot(model['wh2'], h1)
    h2 = relu(ph2)
    py = np.dot(model['wo'], h2)
    y = sigmoid(py)
    episode_buffer['x'].append(x)
    episode_buffer['ph1'].append(ph1)
    episode_buffer['h1'].append(h1)
    episode_buffer['ph2'].append(ph2)
    episode_buffer['h2'].append(h2)
    episode_buffer['py'].append(py)
    episode_buffer['y'].append(y)
    return y


def backward(model: Model, episode_buffer: EpisodeBuffer, episode_reward: np.ndarray) -> Gradient:
    """
        Do a backward pass to get the gradient of the network weights.
    """
    y_true = np.vstack(episode_buffer['y_true'])
    y = np.vstack(episode_buffer['y'])
    py = np.vstack(episode_buffer['py'])
    h2 = np.vstack(episode_buffer['h2'])
    ph2 = np.vstack(episode_buffer['ph2'])
    h1 = np.vstack(episode_buffer['h1'])
    ph1 = np.vstack(episode_buffer['ph1'])
    x = np.vstack(episode_buffer['x'])

    # the objective here is to maximize the log likelihood of y_true being chosen
    # (given the probability y) (see http://cs231n.github.io/neural-networks-2/#losses 
    # section 'Attribute classification' for more details), so the gradient of the 
    # log likelihood function on py should be:
    grad_py = y_true - y

    adv_grad_py = grad_py * episode_reward # advantage based on reward
    grad_wo = np.dot(adv_grad_py.T, h2)
    grad_h2 = np.dot(adv_grad_py, model['wo'])
    grad_ph2 = relu_prime(ph2) * grad_h2
    grad_wh2 = np.dot(grad_ph2.T, h1)
    grad_h1 = np.dot(grad_ph2, model['wh2'])
    grad_ph1 = relu_prime(ph1) * grad_h1
    grad_wh1 = np.dot(grad_ph1.T, x)

    return {'wh1': grad_wh1, 'wh2': grad_wh2, 'wo': grad_wo}


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
            y = forward(x, model, episode_buffer)
            action, y_true = (2, 1.0) if np.random.uniform() < y else (5, 0.0)
            episode_buffer['y_true'].append(y_true)
            
            # perform action and get new observation
            observation, reward, episode_done, info = env.step(action)
            episode_buffer['reward'].append(reward)
            timestep += 1
            
            if episode_done:
                # backward pass
                episode_reward = normal_discounted_reward(episode_buffer, discount_factor)
                gradient = backward(model, episode_buffer, episode_reward)
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
