from typing import Callable, Tuple
import game
import numpy as np

"""Type alias for m by n location/index values"""
Location = Tuple[int, int]

"""Type alias for movement tuple"""
GameMove = Tuple[Callable, Location]


def minimax_search(state: np.ndarray) -> GameMove:
    """Perform a minimax search for the given state and return the optimal move.

    Implemented according to textbook pseudo code on p. 150

    Args:
        state (np.ndarray): Current state

    Returns:
        Tuple[Callable, game.Location]: Return the optimal move. Tuple of Callable and Location
    """
    player = state[0]
    value, move = max_value(state)

    return move


def max_value(state: np.ndarray) -> Tuple[int, GameMove]:
    """Look for the move generating the maximum value.

    Args:
        state (np.ndarray): Current state

    Returns:
        Tuple[int, Tuple]: Tuple of value and move
    """
    move = None
    if game.is_terminal(state):
        return game.utility(state), move

    v = -20
    for act in game.actions(state):
        v2, act2 = min_value(game.result(state, act))

        if v2 > v:
            v = v2
            move = act

    return v, move


def min_value(state: np.ndarray) -> Tuple[int, GameMove]:
    """Look for the move generating minimum value.

    Args:
        state (np.ndarray): Current state

    Returns:
        Tuple[int, Tuple]: Tuple of value and move
    """
    move = None
    v = 20
    for act in game.actions(state):
        v2, act2 = max_value(game.result(state, act))

        if v2 < v:
            v = v2
            move = act

    return v, move
