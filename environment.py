from enum import Enum
from typing import Optional

import gymnasium as gym
import numpy as np
import torchrl

import torch
import torch.nn.functional as F
import torchrl.envs

from thud_game import ThudGame

class Player(Enum):
    DWARF = 0
    TROLL = 1

class ThudEnv(gym.Env):
    def __init__(self, move_limit=400, no_capture_limit=200, h: Optional[int] = None, w: Optional[int] = None):
        self._game: Optional[ThudGame] = None
        self._move_limit = move_limit
        self._no_capture_limit = no_capture_limit
        self._player_to_move = Player.DWARF
        self._pos_selected: Optional[tuple[int, int]] = None

        if h is None:
            h = 15
        self._h = h
        if w is None:
            w = 15
        self._w = w

        self.observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=0, high=1, shape=(12, self._h, self._w), dtype=float),
        })
        self.action_space = gym.spaces.Dict({"action": gym.spaces.Box(low=0, high=224, shape=(1,), dtype=int)})

    def _get_obs(self):
        import copy
        obs = self._game.observations(copy.deepcopy(self._pos_selected), self._h, self._w)
        return {
            "obs": obs,
        }
    
    def _get_info(self):
        return {
            # "player_to_move": self._player_to_move.value,
            # "pos_selected": (-1, -1) if self._pos_selected is None else self._pos_selected,
            # "legal_units": self._game.legal_units(),
            # "legal_moves": self._game.get_legal_moves(self._pos_selected),
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._game = ThudGame(self._move_limit, self._no_capture_limit)
        self._player_to_move = Player.DWARF
        self._pos_selected = None

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        player = self._player_to_move
        if self._pos_selected is None:
            self._pos_selected = (action // self._w, action % self._w)
        else:
            to_pos = (action // self._w, action % self._w)
            self._game.make_move((int(self._pos_selected[0]), int(self._pos_selected[1])), to_pos)
            self._player_to_move = Player((self._player_to_move.value + 1) % 2)
            self._pos_selected = None
        
        terminated = self._game.is_game_over()
        truncated = False
        reward = 0
        if terminated:
            dwarf_count = np.sum(self._game.board == 1)
            troll_count = np.sum(self._game.board == 2)
            if player == Player.DWARF:
                reward = dwarf_count - 4 * troll_count
            else:
                reward = 4 * troll_count - dwarf_count

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

gym.register(
    id="gymnasium_env/ThudGame-v0",
    entry_point=ThudEnv,
)

def apply_masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_size, rows, cols = logits.shape
    policy = F.softmax(logits.view(batch_size, -1), dim=1)
    policy *= mask.view(batch_size, -1)
    policy = F.normalize(policy, dim=1, p=1)
    return policy.view(batch_size, rows, cols)

def find_max_coords(matrix: np.ndarray) -> tuple[int, int]:
    max_value = torch.max(matrix)  # Находим максимальное значение
    indices = torch.argwhere(matrix == max_value)  # Находим все координаты с этим значением
    return tuple(indices[0])  # Берем первую найденную координату

def find_max_coords_np(matrix: np.ndarray) -> tuple[int, int]:
    max_value = np.max(matrix)  # Находим максимальное значение
    indices = np.argwhere(matrix == max_value)  # Находим все координаты с этим значением
    return tuple(indices[0])  # Берем первую найденную координату


def rollout_policy(obs):
    logits = torch.randn(15, 15)
    units_legal = obs[5]
    move_legal = obs[6]
    if move_legal.sum().item() == 0:
        logits = apply_masked_softmax(logits[None,], units_legal[None,])
    else:
        logits = apply_masked_softmax(logits[None,], move_legal[None,])
    pos = find_max_coords(logits[0])
    return {
        "action": (15 * pos[0] + pos[1]).item()
    }

def rollout_policy_parallel(obs):
    logits = torch.randn(obs.size(0), 15, 15)
    units_legal = obs[:, 5]
    move_legal = obs[:, 6]
    if move_legal.sum().item() == 0:
        logits = apply_masked_softmax(logits, units_legal)
    else:
        logits = apply_masked_softmax(logits, move_legal)

    batched_poses = []
    for i in range(obs.size(0)):
        pos = find_max_coords(logits[i])
        batched_poses.append(15 * pos[0] + pos[1])
    return {
        "action": torch.stack(batched_poses)[..., None]
    }

if __name__ == "__main__":
    env = torchrl.envs.GymEnv("gymnasium_env/ThudGame-v0")#, num_envs=1)
    env.reset()
    env.rollout(max_steps=100, policy=rollout_policy)
    td = env.rand_step()
    # print(td)

    # parallel_env = torchrl.envs.ParallelEnv(2, lambda env=env: env)
    # parallel_env.reset()
    # parallel_env.rollout(max_steps=300, policy=rollout_policy_parallel)
