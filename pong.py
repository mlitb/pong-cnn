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
from agent import Agent

tf.enable_eager_execution()


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
    episode_count = 0
    agent = Agent((80, 80, 1))
    env = gym.make('Pong-v0')
    observation = env.reset()

    while True:
        state = preprocess(observation)
        action = agent.sample_action(state=state)
        observation, reward, episode_done, info = env.step(action)
        agent.register(reward=reward, reset_discounted_reward=(reward == 1), episode_done=episode_done)

        if episode_done:
            env.reset()
            episode_count += 1
            print('Episode: {}'.format(episode_count))

        if episode_count % BATCH_SIZE == 0:
            agent.update_policy()

        if render:
            env.render()

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an RL agent to play the mighty game of Pong.')
    parser.add_argument('-r', '--render', action="store_true", default=False, help='whether to render the environment or not')
    args = parser.parse_args()
    main(render=args.render)
    