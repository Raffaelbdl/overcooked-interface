from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

import cv2
import numpy as np
import pettingzoo
import pygame

from overcooked_interface import ActionMap
from overcooked_interface import ActionType, ObservationType
from overcooked_interface.agent import Agent


class Game(ABC):
    """Game interface"""

    @abstractmethod
    def play_game(self, agents: Agent) -> bool:
        raise NotImplementedError


class PygameModule:
    """Handle human inputs and visual display"""

    def __init__(
        self, screen: pygame.display, action_map: dict[int:int] | None, fps: int
    ) -> None:
        self.screen = screen

        self.action_map = ActionMap if action_map is None else action_map

        self.human_action_queue = 4  # do nothing

        self.fps = fps
        self.frame_timer = 0.0
        self.frame_duration = 1 / self.fps

        self._prev_time = 0.0

    def collect_human(self) -> None:
        """Queue human action"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key in self.action_map.keys():
                    self.human_action_queue = self.action_map[event.key]

    @property
    def delta_time(self):
        next_time = time.time()
        delta = next_time - self._prev_time
        self._prev_time = next_time
        return delta

    def update_display(self, frame: np.ndarray):
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)
        frame = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame, (0, 0))
        pygame.display.update()

    def reset_human_action_queue(self):
        self.human_action_queue = 4


@dataclass
class HumanAgentAction:
    human_action: ActionType
    agent_action: ActionType


class PettingZooModule:
    """Handle game core environment"""

    def __init__(
        self, seed: int, parallel_env: pettingzoo.ParallelEnv, agent_skip_steps: int
    ) -> None:
        self.env = parallel_env
        self.env.reset(seed=seed)
        self.agent_names = parallel_env.agents

        self.agent_action_queue = 4  # do nothing
        self.agent_observation: ObservationType = None

        self.agent_count = 0
        self.agent_skip_steps = agent_skip_steps

    def interact_with_env(self, joint_action: HumanAgentAction) -> tuple[bool, float]:
        joint_action = {
            self.agent_names[0]: joint_action.human_action,
            self.agent_names[1]: joint_action.agent_action,
        }

        s, r, d, t, i = self.env.step(joint_action)
        self.agent_observation = self.make_agent_observation(s)

        termination = any(d.values()) or any(t.values())
        return not termination, r

    def make_agent_observation(self, s: dict[str, ObservationType]) -> ObservationType:
        return s[self.agent_names[1]]

    def collect_agent(self, agent: Agent) -> None:
        if self.agent_count % self.agent_skip_steps == 0:
            self.agent_action_queue = agent.select_action(self.agent_observation)

    def incr_agent_count(self):
        self.agent_count += 1

    def reset_agent_action_queue(self):
        self.agent_action_queue = 4


class PygamePettingZooGame(Game):
    def __init__(
        self,
        seed: int,
        parallel_env: pettingzoo.ParallelEnv,
        agent_skip_steps: int,
        screen: pygame.surface,
        action_map: dict[int:int] | None,
        fps: int,
    ) -> None:
        super().__init__()
        self.pettingzoo = PettingZooModule(seed, parallel_env, agent_skip_steps)
        self.pygame = PygameModule(screen, action_map, fps)

    def play_game(self, agent: Agent) -> bool:
        total_score = {"human": 0.0, "agent": 0.0}

        s, _ = self.pettingzoo.env.reset()
        self.pettingzoo.make_agent_observation(s)

        self.pygame._prev_time = time.time()

        play_active = True
        while play_active:
            delta_time = self.pygame.delta_time
            self.pygame.frame_timer += delta_time

            self.pygame.collect_human()
            if self.pygame.frame_timer > self.pygame.frame_duration:
                play_active, score = self.pettingzoo.interact_with_env(
                    self.get_human_agent_joint_action()
                )
                self.pygame.update_display(self.pettingzoo.env.render())

                self.reset_human_agent_action_queue()

                for i, v in enumerate(score.values()):
                    if i == 0:
                        total_score["human"] += v
                    else:
                        total_score["agent"] += v

                self.pettingzoo.incr_agent_count()
                self.pettingzoo.collect_agent(agent)

                self.pygame.frame_timer = 0.0

        return total_score

    def get_human_agent_joint_action(self):
        return HumanAgentAction(
            self.pygame.human_action_queue, self.pettingzoo.agent_action_queue
        )

    def reset_human_agent_action_queue(self):
        self.pygame.reset_human_action_queue()
        self.pettingzoo.reset_agent_action_queue()
