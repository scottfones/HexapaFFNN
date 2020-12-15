import numpy as np


def actions_check_advance(state, src):
    direction = state[1][src[0], src[1]]
    dst = (src[0] + direction, src[1])

    if not is_valid_index(dst):
        return False

    if state[1][dst[0], dst[1]]:
        return False

    return True


def actions_check_capture(state, src, target):
    direction = state[1][src[0], src[1]]
    dst = (src[0] + direction, src[1] + target)

    if not is_valid_index(dst):
        return False

    src_id = state[1][src[0], src[1]]
    dst_id = state[1][dst[0], dst[1]]
    if src_id * dst_id == -1:
        return True

    return False


def actions(state):
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


def advance(state, src):
    direction = state[1][src[0], src[1]]
    dst = (src[0] + direction, src[1])

    return finish_action(state, dst, src)


def capture_left(state, src):
    direction = state[1][src[0], src[1]]
    dst = (src[0] + direction, src[1] - 1)

    return finish_action(state, dst, src)


def capture_right(state, src):
    direction = state[1][src[0], src[1]]
    dst = (src[0] + direction, src[1] + 1)

    return finish_action(state, dst, src)


def finish_action(state, dst, src):
    val = state[1][src[0], src[1]]

    new_board = np.copy(state[1])
    new_board[src[0], src[1]] = 0
    new_board[dst[0], dst[1]] = val

    return new_board


def is_terminal(state):
    if utility(state):
        return True

    return False


def is_valid_index(idx):
    for i in idx:
        if i not in [0, 1, 2]:
            return False
    return True


def new_game():
    board = np.zeros((3, 3), dtype=int)
    board[0, :] += 1
    board[2, :] -= 1

    return (-1, board)


def print_state(state):
    pd = {-1: "Min", 1: "Max"}
    print(f"Current Player: {pd[state[0]]}\n{state[1]}")


def result(state, act):
    new_turn = state[0] * -1
    new_board = act[0](state, act[1])
    return (new_turn, new_board)


def to_move(state):
    return state[0]


def to_vector(state):
    return np.concatenate((np.array(state[0]), state[1]), axis=None)


def utility(state):
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
