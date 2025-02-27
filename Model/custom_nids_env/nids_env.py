import gym
import numpy as np
import pandas as pd
import torch
from gym import spaces
from sklearn.preprocessing import MinMaxScaler

class NIDSEnv(gym.Env):
    def __init__(self, data_path="UNSW_NB15_training-set.csv"):
        super(NIDSEnv, self).__init__()

        # Load Dataset
        self.df = pd.read_csv(data_path)

        # Feature selection algorithm needed for live traffic later on
        self.df.drop(["attack_cat","stcpb", "dtcpb", "swin", "dwin", "tcprtt", "synack", "ackdat", "ct_flw_http_mthd", "smean", "dmean", "sloss", "dloss"], axis=1, inplace=True)
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
        return self.features[self.current_index].astype(np.float32), {}

    def step(self, action):
        """Take an action and return the next state, reward, done flag, and info"""

        # Reward system (modify as per your logic)
        actual_label = self.labels[self.current_index]
        reward = 1 if action == actual_label else -1

        # Move to the next sample
        self.current_index += 1

        # Ensure `done` is a Python bool
        done = bool(self.current_index >= len(self.features))

        print(f"done type: {type(done)}")

        # Get next state or return zeros if done
        if not done:
            next_state = self.features[self.current_index].astype(np.float32)
        else:
            next_state = np.zeros(self.features.shape[1], dtype=np.float32)

        return next_state, reward, done, {}  # Must return exactly 4 values

    def render(self, mode="human"):
        """Later"""
        pass