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
# Code authors: [Valeria grotto 200101266021, Dalim Wahby 19970606-T919]
#

# Load packages
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # combined experience replay --> the last element is the newest
        batch[n-1] = self.buffer[len(self.buffer)-1]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)

### Deep Q Network ###
class DQNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, input_size, output_size):
        super().__init__()

        hidden_size = 12
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(device)

        # Create output layer
        self.output_layer = nn.Linear(8, output_size)

    def forward(self, x):
        return self.net(x)

### Dueling Deep Q Network ###
class DuelingDQNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, input_size, output_size):
        super().__init__()

        hidden_sizes = [32, 16]
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], output_size)
        ).to(device)

        # V(s) is a scalar
        self.value_function_layer = nn.Sequential(
            nn.Linear(output_size, hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        ).to(device)

        #  and A(s,a) layers
        self.advantage_layer = nn.Sequential(
            nn.Linear(output_size, hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)
        ).to(device)

    def forward(self, x):
        x = self.net(x)
        value = self.value_function_layer(x)
        advantage = self.advantage_layer(x)
        advAvg = advantage.mean()
        q = value + (advantage - advAvg)
        return q



class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action

class DQNAgent(Agent):
    def __init__(self, n_actions, n_states, gamma, learning_rate, dueling = False):
        super(DQNAgent, self).__init__(n_actions)

        self.n_states = n_states

        # Initialize your neural network
        if not dueling:
            self.q_net = DQNetwork(n_states, n_actions)
            self.target_q_net = DQNetwork(n_states, n_actions)
        else:
            self.q_net = DuelingDQNetwork(n_states, n_actions)
            self.target_q_net = DuelingDQNetwork(n_states, n_actions)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # Copy the weights from q_net to target_q_net
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.step_count = 0
        self.gamma = gamma

    def forward(self, state: np.ndarray, epsilon = 0.5):
        # Convert the state to a PyTorch tensor
        if not isinstance(state, np.ndarray):
          state_tensor = torch.tensor(state).to(device)
        else:
          state_tensor = torch.from_numpy(state).to(device)

        # Forward pass through the neural network
        q_values = self.q_net(state_tensor)

        # Use an epsilon-greedy strategy to choose an action
        if np.random.rand() < epsilon:
            # Explore: choose a random action
            self.last_action = np.random.randint(0, self.n_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            self.last_action = torch.argmax(q_values).item()

        return self.last_action

    def backward(self, exp, max_steps):
        state, action, reward, next_state, done = exp
        # Convert states to PyTorch tensors
        if not isinstance(state, np.ndarray):
          state_tensor = torch.tensor(state).to(device)
        else:
          state_tensor = torch.from_numpy(state).to(device)
        next_state_tensor = torch.tensor(next_state).to(device)
        action_tensor = torch.tensor(action, dtype=torch.long).to(device)

        target_q_values = []
        curr_q_vals = []

        # Forward pass to get Q-values for the current state
        curr_q_vals = self.q_net(state_tensor)[torch.arange(state_tensor.size(0)), action_tensor]

        # Calculate the target Q-value using the Bellman equation
        with torch.no_grad():
            next_q_values = self.target_q_net(next_state_tensor)
            max_next_q_value, _ = torch.max(next_q_values, 1)
            for i in range(len(done)):
                if done[i]:
                    target = torch.tensor(reward[i]).to(device)
                else:
                    target = reward[i] + self.gamma * max_next_q_value[i]

                target_q_values.append(target)


            #curr_q_vals = current_q_values[torch.arange(current_q_values.size(0)), action_tensor]
            # curr_q_vals.requires_grad=True
            # Use torch.gather to select the Q-values corresponding to the taken actions
            target_q_values = torch.stack(target_q_values).to(device)

        # Calculate the loss (mean squared error between current Q-value and target Q-value)
        loss = F.mse_loss(curr_q_vals, target_q_values.clone().detach().requires_grad_(False)).to(device)

        # Perform a backward pass and update the weights
        self.q_net.zero_grad()
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.)
        self.optimizer.step()

        self.step_count += 1

        if (self.step_count % max_steps == 0):
            self.target_q_net.load_state_dict(self.q_net.state_dict())


    def save_net(self, net_name = 'neural-network-1'):
      '''Function to save the nn at the end of the training'''
      path = net_name + '.pth'
      torch.save(self.q_net, path)
