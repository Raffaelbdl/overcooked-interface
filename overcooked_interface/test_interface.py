import pygame

from overcooked_interface.agent import RandomAgent
from overcooked_interface.env import OvercookedEnvPettingZoo
from overcooked_interface.game import PygamePettingZooGame
from overcooked_interface.pygame_utils import minimal_setup_pygame


def main():
    screen = minimal_setup_pygame(1056, 928)
    agent = RandomAgent(0)
    env = OvercookedEnvPettingZoo(400, "cramped_room")
    game = PygamePettingZooGame(0, env, 4, screen, None, 6)

    game.play_game(agent)


if __name__ == "__main__":
    main()
