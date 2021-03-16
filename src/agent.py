import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from env import Env2048
from net import ValueNet, Transition, ReplayMemory


class Agent():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ValueNet(input_size=16*16*4).to(self.device)
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha, momentum=self.momentum)
        self.batch_size = 128
        self.memory = ReplayMemory(10000)

    def run(self):
        raise NotImplementedError

    def make_one_hot(self, s, a):
        # 4x4 grid, 16 powers of 2 (one-hot)
        num_rows, num_cols = s.shape
        encoding = np.zeros([num_rows, num_cols, 16, 4])
        for row in range(num_rows):
            for col in range(num_cols):
                tile_num = s[row, col]
                if tile_num != 0:
                    power = int(math.log2(tile_num))
                else:
                    power = 0
                encoding[row, col, power, a] = 1
        
        return torch.from_numpy(encoding.reshape(-1,1).flatten()).float().to(self.device)
    
    def policy(self, s):
        with torch.no_grad():
            # Choose A according to the policy (epsilon-greedy)
            actions = [self.model(self.make_one_hot(s, a)) for a in range(0,4)]
            if np.random.rand() < 1 - self.epsilon:
                action = np.argmax(actions)
            else:
                action = np.random.randint(0,4)
        
        return action

class AgentMC(Agent):
    def __init__(self, alpha, epsilon, decay_rate=0.9, momentum=0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.momentum = momentum
        super(AgentMC, self).__init__()

    def make_trajectory(self, s, env):
        '''Given a state, env, and policy, returns a terminating trajectory'''
        trajectory = []
        rewards = []

        while env.can_move(s):
            action = self.policy(s)

            s_prime, reward = env.step(s, action)
            if np.not_equal(s, s_prime).any():
                trajectory.append((s_prime, action))
                rewards.append(reward)
            s = s_prime

        return trajectory, rewards

    def run(self, epoch):
        '''
        Solves our 2048 game for a given policy using Gradient Monte Carlo 

        Args:
        - epoch (int): number of episodes to train over
        Returns:
        - losses (list): losses over training (averaged per episode)
        '''
        losses = []
        lengths = []
        max_pieces = []
        learning_reward = []
        
        for cycle in tqdm(range(epoch)):
            episode_loss = []
            env = Env2048()
            state = env.state
            trajectory, rewards = self.make_trajectory(state, env)
            lengths.append(len(trajectory))
            learning_state = np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 2], [128, 128, 8, 4]])

            # Create G
            G = np.zeros(len(trajectory))
            G[-1] = rewards[-1]
            for t in range(len(trajectory) - 2, -1, -1):
                G[t] = 0.6 * G[t + 1] + rewards[t]

            # Make one hots of trajectory
            one_hots = torch.cat([self.make_one_hot(s, a) for (s,a) in trajectory]).reshape([len(trajectory), 1024])
            value_estimate = self.model(one_hots)
            est_return = torch.FloatTensor([G[t] for t in range(len(trajectory))]).to(self.device)

            # Calculate loss, propogate backward
            loss = self.criterion(value_estimate, est_return)
            episode_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            
            losses.append(np.mean(episode_loss))
            max_pieces.append(np.max(state[-1][0]))
            
            if cycle % 10 == 0:
                with torch.no_grad():
                    learning_action = self.policy(learning_state)
                    reward = self.model(self.make_one_hot(learning_state, learning_action))
                    learning_reward.append(reward)
                    
            if self.epsilon > 0.01:
                self.epsilon *= self.decay_rate
            
        return max_pieces, learning_reward, lengths

class AgentTD(Agent):
    def __init__(self, alpha, epsilon, momentum=0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.momentum=momentum
        super(AgentTD, self).__init__()

    def run(self, epoch):
        '''
        Solves our 2048 game for a given policy using Semi-Gradient TD(0)

        Args:
        - epoch (int): number of episodes to train over
        Returns:
        - w (numpy.ndarray): weight vector for estimated value function
        '''
        max_pieces = []

        for _ in tqdm(range(epoch)):
            episode_loss = []
            # Initialize S
            env = Env2048()
            state = env.state
            
            # Choose A according to the policy (epsilon-greedy)
            action = self.policy(state)
                
            while env.can_move(state):
                # Take action A, observe R, S'
                state_prime, reward = env.step(state, action)

                # Get next action
                action_prime = self.policy(state)

                # Update w, S
                one_hot_s = self.make_one_hot(state, action)
                one_hot_s_prime = self.make_one_hot(state_prime, action_prime)
                value_estimate_s = self.model(one_hot_s)
                value_estimate_s_prime = self.model(one_hot_s_prime)
                est_return = reward + value_estimate_s_prime
                est_return = torch.FloatTensor([est_return])

                loss = self.criterion(value_estimate_s, est_return)
                episode_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

                state = state_prime
                action = action_prime

            losses.append(np.mean(episode_loss))
            max_pieces.append(np.max(state))

        return max_pieces

    
class AgentSARSA(Agent):
    def __init__(self, alpha, epsilon, decay_rate=0.9, momentum=0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.momentum = momentum
        super(AgentSARSA, self).__init__()
    
    def run(self, epoch):
        '''
        Solves our 2048 game for a given policy using Semi-Gradient SARSA

        Args:
        - epoch (int): number of episodes to train over
        Returns:
        - losses (list): losses over training (averaged per episode)
        '''
        max_pieces = []
        learning_reward = []
        env = Env2048()
        learning_state = np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 2], [128, 128, 8, 4]])
        
        for cycle in tqdm(range(epoch)):
            state = env.state.copy()
            action = self.policy(state)
            
            while env.can_move(state):
                # Take action A, ovserve R, S'
                state_prime, reward = env.step(state, action)
                
                # Select next action epsilon-greedily
                action_prime = self.policy(state_prime)
                
                # Add information to memory
                one_hot_s = self.make_one_hot(state, action)
                one_hot_s_prime = self.make_one_hot(state_prime, action_prime)
                reward = torch.FloatTensor([reward]).to(self.device)
                self.memory.push(one_hot_s, action, one_hot_s_prime, reward)
                    
                if len(self.memory) >= self.batch_size:
                    # Get batch info
                    transitions = self.memory.sample(self.batch_size)
                    batch = Transition(*zip(*transitions))
                    
                    state_batch = torch.cat(batch.state).reshape([self.batch_size, 1024]).to(self.device)
                    reward_batch = torch.cat(batch.reward).to(self.device)
                    next_state_batch = torch.cat(batch.next_state).reshape([self.batch_size, 1024]).to(self.device)
                    
                    state_action_values = self.model(state_batch)
                    expected_state_action_values = 0.8 * reward_batch + self.model(next_state_batch)
                    
                    # If S' terminal, update different
                    if not env.can_move(state_prime):
                        value_estimate = self.model(one_hot_s)
                        est_return = torch.FloatTensor([reward]).to(self.device)

                        loss = self.criterion(value_estimate, est_return)
                        loss.backward()
                        self.optimizer.step()
                        break

                    # Compute loss, perform gradient descent step
                    loss = self.criterion(state_action_values, expected_state_action_values)
                    loss.backward()
                    self.optimizer.step()
                    state = state_prime
                    action = action_prime
            
            if cycle % 10 == 0:
                learning_action = self.policy(learning_state)
                reward = self.model(self.make_one_hot(learning_state, learning_action))
                learning_reward.append(reward)
                #loss = self.criterion(reward, torch.FloatTensor([256]))
                #loss.backward()
                #self.optimizer.step()
                    
            if self.epsilon > 0.01:
                self.epsilon *= self.decay_rate
            
            max_pieces.append(np.max(state))
                
        return max_pieces, learning_reward