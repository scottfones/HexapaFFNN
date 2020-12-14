import game


def minimax_search(state):
    player = state[0]
    value, move = max_value(state)

    return move


def max_value(state):
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


def min_value(state):
    move = None
    if game.is_terminal(state):
        return game.utility(state), move

    v = 20
    for act in game.actions(state):
        v2, act2 = max_value(game.result(state, act))

        if v2 < v:
            v = v2
            move = act

    return v, move
