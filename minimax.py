from typing import Callable, Dict, List, Tuple
import game
import numpy as np

"""Type alias for m by n location/index values"""
Location = Tuple[int, int]

"""Type alias for movement tuple"""
GameMove = Tuple[Callable, Location]

"""Type alias for list of legal actions"""
MoveList = List[Tuple[Callable, Location]]

"""Type alias for game state tuple"""
GameState = Tuple[int, np.ndarray]


def create_policytable() -> Dict[GameState, Tuple[int, MoveList]]:
    """Create policytable by iterating through game state space.

    Returns:
        Dict[GameState, Tuple[int, MoveList]]: Map of state to utility and list of moves
    """
    space = game.create_statespace()
    pol_tab = {}

    for state in space:
        move_list = []

        # Convert 1x10 vector into game state tuple
        state = (state[0], np.asarray(state[1:]).reshape(3, 3))

        next_move = minimax_search(state)
        s = state
        while not game.is_terminal(s):
            move_list.append(next_move)

            next_move = minimax_search(s)
            s = game.result(s, next_move)

        u = game.utility(s)

        pol_tab[tuple(game.to_vector(state))] = (u, move_list)

    return pol_tab


def minimax_search(state: np.ndarray) -> GameMove:
    """Perform a minimax search for the given state and return the optimal move.

    Implemented according to textbook pseudo code on p. 150

    Args:
        state (np.ndarray): Current state

    Returns:
        GameMove: Return the optimal move. Tuple of Callable and Location
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
