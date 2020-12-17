from typing import Callable, Tuple
import game
import numpy as np

policy_states = []
policy_table = []


def gen_states() -> set:
    """Generate a set of all possible states.

    Returns:
        set: Set of potential game states
    """
    g = game.new_game()
    _ = minimax_search(g, True)
    return policy_states


def gen_table(ps: set) -> dict:
    """Generate dict matching optimal move for every state.

    Args:
        ps (set): Set of policy states

    Returns:
        dict: Initial State -> Next State mapping
    """
    for s in ps:
        m = minimax_search(s)
        if m:
            policy_table.append((game.to_vector(s), game.to_vector(m)))
    return policy_table


def minimax_search(
    state: np.ndarray, fill_states: bool = False
) -> Tuple[Callable, game.Location]:
    """Perform a minimax search for the given state and return the optimal move.

    Implemented according to textbook pseudo code on p. 150

    Args:
        state (np.ndarray): Current state
        fill_states (bool, optional): Whether to fill the policy states. Defaults to False.

    Returns:
        Tuple[Callable, game.Location]: Return the optimal move. Tuple of Callable and Location
    """
    player = state[0]
    value, move = max_value(state, fill_states)

    return move


def max_value(state: np.ndarray, fill_states: bool = False) -> Tuple[int, Tuple]:
    """Look for the move generating the maximum value.

    Args:
        state (np.ndarray): Current state
        fill_states (bool, optional): Whether to fill the policy states. Defaults to False.

    Returns:
        Tuple[int, Tuple]: Tuple of value and move
    """
    move = None
    if game.is_terminal(state):
        return game.utility(state), move

    v = -20
    for act in game.actions(state):
        if fill_states and act:
            policy_states.append(game.result(state, act))

        v2, act2 = min_value(game.result(state, act), fill_states)

        if fill_states and act2:
            policy_states.append(game.result(state, act2))

        if v2 > v:
            v = v2
            move = act

    return v, move


def min_value(state: np.ndarray, fill_states: bool = False) -> Tuple[int, Tuple]:
    """Look for the move generating minimum value.

    Args:
        state (np.ndarray): Current state
        fill_states (bool, optional): Whether to fill the policy states. Defaults to False.

    Returns:
        Tuple[int, Tuple]: Tuple of value and move
    """
    move = None
    if game.is_terminal(state):
        return game.utility(state), move

    v = 20
    for act in game.actions(state):
        if fill_states and act:
            policy_states.append(game.result(state, act))

        v2, act2 = max_value(game.result(state, act), fill_states)

        if fill_states and act2:
            policy_states.append(game.result(state, act2))

        if v2 < v:
            v = v2
            move = act

    return v, move
