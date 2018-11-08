"""
    An implementation of a policy gradient agent that learns to play the 
    mighty game of Pong.
    
    More Info : https://github.com/mlitb/pong-cnn
    Authors:
        1. Faza Fahleraz https://github.com/ffahleraz
        2. Nicholas Rianto Putra https://github.com/nicholaz99
        3. Abram Perdanaputra https://github.com/abrampers
"""

import os
import gym
import argparse
import tensorflow as tf
import numpy as np
from model import Model

from typing import Dict, List

tf.enable_eager_execution()


# Type shorthands
EpisodeBuffer = Dict[str, List]
Gradient = Dict[str, np.ndarray]


# Training hyperparameters
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = .99

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


def normal_discounted_rewards(episode_buffer: EpisodeBuffer, discount_factor: float) -> float:
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
    policy = Model((80, 80, 1))
    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, name='RMSProp')
    env = gym.make('Pong-v0')
    episode_number = 0

    # if load_fname is not None:
    #     # load json and create model
    #     json_file = open('model.json', 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     policy.load_weights(load_fname)

    batch_gradient = None
    batch_rewards = []

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
                # calculate discounted reward of every step
                episode_rewards = normal_discounted_rewards(episode_buffer, 
                        DISCOUNT_FACTOR)
                episode_gradient = None
                for idx, step_gradient in enumerate(episode_buffer['gradient']):
                    step_gradient = [tf.scalar_mul(episode_rewards[idx][0], var_grad) \
                            for var_grad in episode_buffer['gradient'][idx]]
                    if episode_gradient is None:
                        episode_gradient = step_gradient
                    else:
                        episode_gradient = [tf.add(episode_gradient[i], 
                                step_gradient[i]) for i in range(len(episode_gradient))]

                # bookeeping
                batch_rewards.append(sum(episode_buffer['reward']))
                episode_number += 1

                # training info
                print('Episode: {}, rewards: {}'.format(episode_number, 
                        sum(episode_buffer['reward'])))
                
                # parameter update (rmsprop)
                if episode_number % BATCH_SIZE == 0:
                    if batch_gradient is None:
                        batch_gradient = episode_gradient
                    else:
                        batch_gradient = [tf.add(batch_gradient[i], 
                                episode_gradient[i]) for i in len(batch_gradient)]
                    
                    optimizer.apply_gradients(zip(episode_gradient, policy.variables),
                            global_step=tf.train.get_or_create_global_step())
                    batch_gradient = None

                    # training info
                    print('Batch: {}, avg episode rewards: {}'.format(episode_number // BATCH_SIZE, 
                            sum(batch_rewards) / len(batch_rewards)))
                    batch_rewards = []

                # # save model
                # if episode_number % 1 == 0 and save_dir is not None:
                #     if not os.path.exists(save_dir):
                #         os.makedirs(save_dir)
                #     save_fname = os.path.join(save_dir, 'save_{}.h5'.format(episode_number))
                #     # serialize model to JSON
                #     model_json = policy.to_json()
                #     with open("model.json", "w") as json_file:
                #         json_file.write(model_json)
                #     print("Saved model to disk")
                #     policy.save_weights(save_fname)
                #     print('Model saved to \'{}\'!'.format(save_fname))

        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an RL agent to play the mighty game of Pong.')
    parser.add_argument('-r', '--render', action="store_true", default=False, help='whether to render the environment or not')
    parser.add_argument('-l', '--load', action="store", default=None, help='path to the saved model to load from')
    parser.add_argument('-s', '--save', action="store", default=None, help='path to the folder to save model')
    args = parser.parse_args()
    main(load_fname=args.load, save_dir=args.save, render=args.render)
    