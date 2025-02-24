import gym
import numpy as np
import pandas as pd
import torch
from gym import spaces
from sklearn.preprocessing import MinMaxScaler

class NIDSEnv(gym.Env):
    def __init__(self, data_path):
        super(NIDSEnv, self).__init__()

        # Load Dataset
        self.df = pd.read_csv(data_path)

        self.df.drop(["attack_cat","stcpb", "dtcpd", "swin", "dwin", "tcprtt", "synack", "ackdat", "ct_flw_http_mthd", "smean", "dmean", "sloss", "dloss"], axis=1, inplace=True)
        categorical_cols = ["proto", "service", "state"]
        self.df = pd.get_dummies(self.df, columns=categorical_cols)

        # Extract features and label
        self.features = self.df.iloc[:, :-1].values
        self.labels = self.df.iloc[:, -1].values

        # Normalize features
        self.scaler = MinMaxScaler()
        self.features = self.scaler.fit_transform(self.features)

        self.current_index = 0

        # Define action spaces (2 actions: benign, malicious), removed Suspicious action
        self.action_space = spaces.Discrete(2)

        # Define state space (feature vector)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.features.shape[1],), dtype=np.float32)

    def reset(self):
        """Reset environment at the beginning of an episode"""
        self.current_index = 0
        return self.features[self.current_index]
    
    def step(self, action):
        """Take an action and return the next state, reward and done flag"""

        # Get correct label
        actual_label = self.labels[self.current_index]

        # Reward system
        if action == actual_label:
            reward = 1
        else:
            reward = -1

        # Move to the next sample
        self.current_index += 1

        # Check if the episode is done
        done = self.current_index >= len(self.features)

        # Get next state
        if not done:
            next_state = self.features[self.current_index]
        else:
            next_state = np.zeros(self.features.shape[1])

        return next_state, reward, done, {}
    
    def render(self, mode="human"):
        """Later"""
        pass