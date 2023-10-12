import pygame


def minimal_setup_pygame(width: int, height: int) -> pygame.Surface:
    pygame.init()
    pygame.display.set_caption("Pygame")
    return pygame.display.set_mode((width, height))
