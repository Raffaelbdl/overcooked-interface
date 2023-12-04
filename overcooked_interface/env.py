import copy

import cv2
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
import pygame

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
)
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

policy_mapping_dict = {
    "all_scenario": {
        "description": "overcook all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

env_config = {
    # "map_name": "asymmetric_advantages",
    # "map_name": "cramped_room",
    "map_name": "coordination_ring",
    "horizon": 400,
    "featurize_fn": "lossless_state_encoding_mdp",  # featurize_state_mdp
}


class OvercookedEnvPettingZoo(ParallelEnv):
    metadata = {"name": "overcooked"}

    def __init__(
        self, timesteps: int = 400, map_name: str = None, flatten_obs: bool = False
    ):
        self.max_cycles = env_config["horizon"] = timesteps
        if map_name is not None:
            env_config["map_name"] = map_name

        layout_mdp = OvercookedGridworld.from_layout_name(env_config["map_name"])
        base_env = OvercookedEnv.from_mdp(
            layout_mdp, horizon=env_config["horizon"], info_level=0
        )

        self.featurize_fn = {
            "lossless_state_encoding_mdp": base_env.lossless_state_encoding_mdp,
            "featurize_state_mdp": base_env.featurize_state_mdp,
        }[env_config["featurize_fn"]]

        self.agents = [0, 1]
        self.possible_agents = [0, 1]
        self.base_env = base_env

        self.flatten_obs = flatten_obs
        self._observation_space = self._setup_observation_space()
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.agents
        }
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}

        self.visualizer = StateVisualizer()

        self.reset()

    def observation_space(self, agent: str):
        return self._observation_space

    def action_space(self, agent):
        return Discrete(len(Action.ALL_ACTIONS))

    def step(self, joint_action):
        joint_action = [
            Action.ALL_ACTIONS[joint_action[agent]] for agent in joint_action
        ]
        obs, reward, done, info = self.base_env.step(joint_action)
        # https://gymnasium.farama.org/content/basic_usage/
        # we have no early termination condition in this env, and the environment only terminates when the time horizon is reached
        # therefore the terminated is always False, and we set truncated to done
        terminated = False
        truncated = done

        def create_dict(value):
            """
            Each agent should have the same reward, terminated, truncated, info
            """
            return {agent: value for agent in self.agents}

        def create_obs_dict(obs):
            """
            Observation is potentially different for each agent
            """
            return {
                agent: self.observe(self.featurize_fn(obs)[i])
                for i, agent in enumerate(self.agents)
            }

        def create_r_shaped_dict(reward, info):
            return {
                agent: reward + info["shaped_r_by_agent"][i]
                for i, agent in enumerate(self.agents)
            }

        obs = create_obs_dict(obs)
        # reward = create_dict(reward)
        reward = create_r_shaped_dict(reward, info)
        terminated = create_dict(terminated)
        truncated = create_dict(truncated)
        info = create_dict(info)
        if done:
            self.agents = []
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the embedded OvercookedEnv envrionment to the starting state
        """
        self.base_env.reset()
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        # when an environment terminates/truncates, PettingZoo wants all agents removed, so during reset we re-add them
        self.agents = self.possible_agents[:]
        # return the obsevations as dict
        obs_dict = {
            agent: self.observe(self.featurize_fn(dummy_state)[0])
            for agent in self.agents
        }
        return obs_dict, None

    def render(self):
        rewards_dict = {}  # dictionary of details you want rendered in the UI
        for key, value in self.base_env.game_stats.items():
            if key in [
                "cumulative_shaped_rewards_by_agent",
                "cumulative_sparse_rewards_by_agent",
            ]:
                rewards_dict[key] = value

        image = self.visualizer.render_state(
            state=self.base_env.state,
            grid=self.base_env.mdp.terrain_mtx,
            hud_data=StateVisualizer.default_hud_data(
                self.base_env.state, **rewards_dict
            ),
        )

        buffer = pygame.surfarray.array3d(image)
        image = copy.deepcopy(buffer)
        image = np.flip(np.rot90(image, 3), 1)
        image = cv2.resize(image, (2 * 528, 2 * 464))
        return image

    def _setup_observation_space(self):
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        if not self.flatten_obs:
            high = np.ones(obs_shape) * float("inf")
            low = np.zeros(obs_shape)
            return Box(low, high, dtype=np.float32)
        obs_shape = (np.prod(np.array(obs_shape)),)
        high = np.ones(obs_shape) * float("inf")
        low = np.zeros(obs_shape)
        return Box(low, high, dtype=np.float32)

    def observe(self, observation):
        if self.flatten_obs:
            return np.reshape(observation, (-1,))
        return observation
