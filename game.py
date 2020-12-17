import numpy as np
from typing import Callable, List, NoReturn, Tuple

"""Type alias for m by n location/index values"""
Location = Tuple[int, int]

"""Type alias for list of legal actions"""
ActionList = List[Tuple[Callable, Location]]


def actions_check_advance(state: np.ndarray, src: Location) -> bool:
    """Check if advance is a legal move.

    Args:
        state (np.ndarray): Current state
        src (Location): m by n index values for pawn

    Returns:
        bool: True if advance is a legal move
    """
    direction = state[1][src[0], src[1]]
    dst = (src[0] + direction, src[1])

    if not is_valid_index(dst):
        return False

    if state[1][dst[0], dst[1]]:
        return False

    return True


def actions_check_capture(state: np.ndarray, src: Location, target: int) -> bool:
    """Check if capture in the target direction is a legal move.

    Args:
        state (np.ndarray): Current state
        src (Location): m by n index values for pawn
        target (int): -1 or 1 for left or right capture

    Returns:
        bool: True if capture is a legal move in target direction.
    """
    direction = state[1][src[0], src[1]]
    dst = (src[0] + direction, src[1] + target)

    if not is_valid_index(dst):
        return False

    src_id = state[1][src[0], src[1]]
    dst_id = state[1][dst[0], dst[1]]
    if src_id * dst_id == -1:
        return True

    return False


def actions(state: np.ndarray) -> ActionList:
    """Generate a list of actions for the current player.

    Iterate through the current player's pawns. For each,
    query possible moves and add to list of legal moves if
    valid.

    Args:
        state (np.ndarray): Current state

    Returns:
        ActionList: List of valid moves of the form (Callable, Location)
    """
    action_list = []
    pawn_id = to_move(state)

    for m in range(3):
        for n in range(3):
            if state[1][m][n] == pawn_id:
                adv = actions_check_advance(state, (m, n))
                cleft = actions_check_capture(state, (m, n), -1)
                cright = actions_check_capture(state, (m, n), 1)

                if adv:
                    action_list.append((advance, (m, n)))

                if cleft:
                    action_list.append((capture_left, (m, n)))

                if cright:
                    action_list.append((capture_right, (m, n)))

    return action_list


def advance(state: np.ndarray, src: Location) -> np.ndarray:
    """Advance a pawn at the src location.

    Args:
        state (np.ndarray): Current state
        src (Location): Location of pawn

    Returns:
        np.ndarray: New board state
    """
    direction = state[1][src[0], src[1]]
    dst = (src[0] + direction, src[1])

    return finish_action(state, dst, src)


def capture_left(state: np.ndarray, src: Location) -> np.ndarray:
    """Capture the pawn at the left diagonal to the src location.

    Args:
        state (np.ndarray): Current state
        src (Location): Location of pawn

    Returns:
        np.ndarray: New board state
    """
    direction = state[1][src[0], src[1]]
    dst = (src[0] + direction, src[1] - 1)

    return finish_action(state, dst, src)


def capture_right(state: np.ndarray, src: Location) -> np.ndarray:
    """Capture the pawn at the right diagonal to the src location.

    Args:
        state (np.ndarray): Current state
        src (Location): Location of pawn

    Returns:
        np.ndarray: New board state
    """
    direction = state[1][src[0], src[1]]
    dst = (src[0] + direction, src[1] + 1)

    return finish_action(state, dst, src)


def finish_action(state: np.ndarray, dst: Location, src: Location) -> np.ndarray:
    """Complete an action by moving the pawn from src to dst and backfilling with 0.

    Args:
        state (np.ndarray): Current state
        dst (Location): Ending location
        src (Location): Starting location

    Returns:
        np.ndarray: New board state
    """
    val = state[1][src[0], src[1]]

    new_board = np.copy(state[1])
    new_board[src[0], src[1]] = 0
    new_board[dst[0], dst[1]] = val

    return new_board


def is_terminal(state: np.ndarray) -> bool:
    """Check for terminal game state.

    Args:
        state (np.ndarray): Current state

    Returns:
        bool: Return true if game over
    """
    if utility(state):
        return True

    return False


def is_valid_index(idx: Location) -> bool:
    """Validate index is within bounds of board.

    Args:
        idx (Location): m by n index of pawn

    Returns:
        bool: Return true if location is valid
    """
    for i in idx:
        if i not in [0, 1, 2]:
            return False
    return True


def new_game() -> np.ndarray:
    """Generate new board and place pawns.

    Board is flipped upside down from writeup to allow
    -1 or 1 to encode direction of travel.

    Returns:
        np.ndarray: Initial board state
    """
    board = np.zeros((3, 3), dtype=int)
    board[0, :] += 1
    board[2, :] -= 1

    return (-1, board)


def print_state(state: np.ndarray) -> NoReturn:
    """Print board state in more legible format.

    Args:
        state (np.ndarray): Current state

    Returns:
        NoReturn:
    """
    pd = {-1: "Min", 1: "Max"}
    print(f"Current Player: {pd[state[0]]}\n{state[1]}")


def result(state: np.ndarray, act: Tuple[Callable, Location]) -> Tuple[int, np.ndarray]:
    """Apply an action to the current board and return the resulting state.

    Args:
        state (np.ndarray): Current state
        act (Tuple[Callable, Location]): Tuple of action function and pawn location

    Returns:
        Tuple[int, np.ndarray]: New state after applying action
    """
    new_turn = state[0] * -1
    new_board = act[0](state, act[1])
    return (new_turn, new_board)


def to_move(state: np.ndarray) -> int:
    """Return the current player.

    Args:
        state (np.ndarray): Current state

    Returns:
        int: Int representing the current player
    """
    return state[0]


def to_vector(state: np.ndarray) -> np.ndarray:
    """Convert tuple state to vector format.

    Args:
        state (np.ndarray): Current state

    Returns:
        np.ndarray: Vector representation of state
    """
    return np.concatenate((np.array(state[0]), state[1]), axis=None)


def utility(state: np.ndarray) -> int:
    """Calculate state utility.

    Args:
        state (np.ndarray): Current state

    Returns:
        int: Utility is element of [-1, 0, 1]
    """
    # Min in Top Row
    if -1 in state[1][0, :]:
        return -1

    # Max in Bottom Row
    elif 1 in state[1][2, :]:
        return 1

    # No Moves for Current Player
    # action list empty == False
    elif not actions(state):
        return -1 * state[0]

    else:
        return 0
