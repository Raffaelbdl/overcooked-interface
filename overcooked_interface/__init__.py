import numpy as np
import pygame

OvercookedActionDim = 6
ObservationType = np.ndarray
ActionType = int

ActionMap = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_RIGHT: 2,
    pygame.K_LEFT: 3,
    pygame.K_SPACE: 5,
}
