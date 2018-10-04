"""
    An implementation of a policy gradient agent that learns to play Atari games.
    More Info : https://github.com/mlitb/pong-cnn
    Authors:
        1. Faza Fahleraz https://github.com/ffahleraz
        2. Nicholas Rianto Putra https://github.com/nicholaz99
        3. Abram Perdanaputra https://github.com/abrampers
"""

import gym
import tensorflow as tf

from agent import Agent


def main():
    batch_size = 10
    episode_count = 0
    agent = Agent()
    env = gym.make('Pong-v0')
    
    while True:
        observation = env.reset()
        action = agent.sample_action(observation=observation)

        observation, reward, episode_done, info = env.step(action)
        episode_count += 1 * episode_done

        agent.update(reward=reward, reset_value=(reward == 1), reset_episode=episode_done)

        if episode_count % batch_size == 0:
            agent.update_policy()


if __name__ == '__main__':
    main()