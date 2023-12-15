# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code authors: []
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, DQNAgent, ExperienceReplayBuffer

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize enviroment
env = gym.make('LunarLander-v2')

# Parameters
N_episodes = 1000                             # Number of episodes [100,1000]
discount_factor = 0.99                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes

batch_size = 64                              # N, between 4-128,number of elements to sample from the exp buffer [4,128]
buffer_length = 20000                        # L, between 5000-30000

learning_rate = 0.001                        # usually 10^-3, 10^-4


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# net_name is a parameter used to save the network weight and the plots with a specific name
def train_enviroment(net_name, env, N_episodes, n_ep_running_average, buffer_length, batch_size, discount_factor, learning_rate, dueling = True):
  # Initialize the discrete Lunar Laner Environment
  env.reset()

  # PARAMETERS
  n_actions = env.action_space.n               # Number of available actions
  dim_state = len(env.observation_space.high)  # State dimensionality
  maximum_steps_episode = 1000                 # we stop the episode if we reach more than this number of steps

  max_steps = int(buffer_length/batch_size)    # C = L/N
  n_episodes_decay = 0.95*N_episodes           # Z, this is usually 90-95% of the number of episodes
  epsilon_max = 0.99
  epsilon_min = 0.05
  early_stopping_threshold = 150               # value for which we stop the training and save the value of the net

  # We will use these variables to compute the average episodic reward and
  # the average number of steps per episode
  episode_reward_list = []       # this list contains the total reward per episode
  episode_number_of_steps = []   # this list contains the number of steps per episode

  agent = DQNAgent(n_actions, dim_state, max_steps, discount_factor, learning_rate, dueling)

  random_agent = RandomAgent(n_actions)
  ### Create Experience replay buffer ###
  buffer = ExperienceReplayBuffer(buffer_length)
  ## fill the buffer with random experiences
  for i in range(10):
      # Reset enviroment data and initialize variables
      done = False
      state = env.reset()
      state = state[0]
      while not done and len(buffer) < buffer_length:
          # Take a random action
          action = random_agent.forward(state)

          next_state, reward, done, _, _ = env.step(action)
          #append to the buffer B
          exp = (state, action, reward, next_state, done)
          buffer.append(exp)
          state = next_state

  ### Training process

  # trange is an alternative to range in python, from the tqdm library
  # It shows a nice progression bar that you can update with useful information
  EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

  epsilon = epsilon_max

  for i in EPISODES:
      # Reset enviroment data and initialize variables
      done = False
      state = env.reset()
      state = state[0]
      total_episode_reward = 0.
      t = 0

      # epsilon exponential decay over number of episodes
      epsilon = max(epsilon_min, epsilon_max*(epsilon_min/epsilon_max)**((i-1)/(n_episodes_decay-1)))
      while not done and t < maximum_steps_episode:
          # pdb.set_trace()
          # Take a random action -> epsilon greedy 
          action = agent.forward(state, epsilon)

          # Get next state and reward.  The done variable
          # will be True if you reached the goal position,
          # False otherwise
          next_state, reward, done, _, _ = env.step(action)

          #append to the buffer B
          buffer.append((state, action, reward, next_state, done))

          # Update episode reward
          total_episode_reward += reward

          # Update state for next iteration
          state = next_state
          t+= 1

          # sample a random batch
          # Perform training only if we have more than batch_size elements in the buffer
          if len(buffer) >= batch_size:
              # combined experience replay
              exp = buffer.sample_batch(batch_size)
              agent.backward(exp)

      # Append episode reward and total number of steps
      episode_reward_list.append(total_episode_reward)
      episode_number_of_steps.append(t)

      # Close environment
      env.close()

      # Updates the tqdm update bar with fresh information
      # (episode number, total reward of the last episode, total number of Steps
      # of the last episode, average reward, average number of steps)
      EPISODES.set_description(
          "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
          i, total_episode_reward, t,
          running_average(episode_reward_list, n_ep_running_average)[-1],
          running_average(episode_number_of_steps, n_ep_running_average)[-1]))

      # early stopping
      if  running_average(episode_reward_list, n_ep_running_average)[-1] > early_stopping_threshold:
        print('The average over the last {} episodes was: {}'.format(n_ep_running_average,running_average(episode_reward_list, n_ep_running_average)[-1]))
        agent.save_net(net_name)
        break



  # Plot Rewards and steps
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
  ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], episode_reward_list, label='Episode reward')
  ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], running_average(
      episode_reward_list, n_ep_running_average), label='Avg. episode reward')
  ax[0].set_xlabel('Episodes')
  ax[0].set_ylabel('Total reward')
  ax[0].set_title('Total Reward vs Episodes')
  ax[0].legend()
  ax[0].grid(alpha=0.3)

  ax[1].plot([i for i in range(1, len(episode_reward_list)+1)], episode_number_of_steps, label='Steps per episode')
  ax[1].plot([i for i in range(1, len(episode_reward_list)+1)], running_average(
      episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
  ax[1].set_xlabel('Episodes')
  ax[1].set_ylabel('Total number of steps')
  ax[1].set_title('Total number of steps vs Episodes')
  ax[1].legend()
  ax[1].grid(alpha=0.3)
  plt.show()

  path = net_name + '.png'
  plt.savefig(path)

  return n_ep_running_average

train_enviroment('neural-network-1', env, N_episodes, n_ep_running_average, buffer_length, batch_size, discount_factor, learning_rate, dueling = True)

# try different discount factors
df = 1
train_enviroment('nn-df1', env, 200, n_ep_running_average, buffer_length, batch_size, df, learning_rate, dueling = True)

df = 0.5
train_enviroment('nn-df1',env, 200, n_ep_running_average, buffer_length, batch_size, df, learning_rate, dueling = True)

# try different n of episodes and memory sizes
avg_diff_n_episods = ()
avg_diff_m_size = ()

n_ep_list = [100, 200, 300, 400, 500, 600, 700]
m_sizes = [5000, 10000, 15000, 20000, 25000, 30000]

for n_ep in n_ep_list:
  name = 'n_ep' + n_ep
  avg_diff_n_episods.append(train_enviroment(name,env, n_ep, n_ep_running_average, buffer_length, batch_size, discount_factor, learning_rate, dueling = True))

for m_size in m_sizes:
  name = 'm_size' + n_ep
  avg_diff_m_size.append(train_enviroment(name,env, N_episodes, n_ep_running_average, m_size, batch_size, discount_factor, learning_rate, dueling = True))