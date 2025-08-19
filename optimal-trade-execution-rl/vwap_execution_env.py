import numpy as np
import gymnasium as gym
from gymnasium import spaces

class VWAPExecutionEnv(gym.Env):
    def __init__(self, prices, volumes, total_qty=10000):
        self.prices = prices
        self.volumes = volumes
        self.total_qty = total_qty
        self.total_steps = len(prices)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)


    def reset(self):
        self.current_step = 0
        self.inventory_left = self.total_qty
        self.executed = []
        obs = self._get_obs()
        return obs, {}


    def _get_obs(self):
        return np.array([
            self.inventory_left,
            self.prices[self.current_step]
        ], dtype=np.float32)


    def step(self, action):
        action = np.clip(action[0], 0.0, 1.0)
        qty = action * self.inventory_left
        price = self.prices[self.current_step]
        cost = qty * price

        self.inventory_left -= qty
        self.executed.append((qty, price))
        self.current_step += 1
        
        done = self.current_step >= len(self.prices) - 1
        reward = -cost  # We want to minimize cost

        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        info = {
            'inventory_left': self.inventory_left,
            'step': self.current_step,
            'executed': self.executed
        }

        return obs, reward, done, False, info

