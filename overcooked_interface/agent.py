from abc import ABC, abstractmethod
import numpy as np

from overcooked_interface import ActionType, ObservationType
from overcooked_interface import OvercookedActionDim


class Agent(ABC):
    """Agent interface"""

    @abstractmethod
    def select_action(self, input: ObservationType) -> ActionType:
        raise NotImplementedError


class RandomAgent(Agent):
    """Random agent used for debugging"""

    def __init__(self, seed: int):
        np.random.seed(seed)

    def select_action(self, input: ObservationType) -> ActionType:
        return np.random.randint(0, OvercookedActionDim - 1)
