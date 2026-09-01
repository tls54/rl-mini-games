import random
from functools import lru_cache

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def _winner(board):
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return None


def _legal(board):
    return [i for i, v in enumerate(board) if v == 0]


def _current_player(board):
    return 1 if sum(board) == 0 else -1


@lru_cache(maxsize=None)
def _negamax(board):
    """Best achievable outcome (1 win / 0 draw / -1 loss) for the player to move,
    assuming optimal play from both sides. `board` is a tuple, no terminal winner yet."""
    legal = _legal(board)
    if not legal:
        return 0

    player = _current_player(board)
    best = -2
    for action in legal:
        next_board = list(board)
        next_board[action] = player
        next_board = tuple(next_board)

        if _winner(next_board) == player:
            value = 1
        elif not _legal(next_board):
            value = 0
        else:
            value = -_negamax(next_board)

        best = max(best, value)

    return best


class MinimaxAgent:
    """Plays perfect TicTacToe. epsilon is accepted for interface compatibility
    but ignored - this agent never explores."""

    def choose_action(self, state, legal_actions, current_player=None, epsilon=0.0):
        board = tuple(int(x) for x in state)
        player = _current_player(board)

        best_value = -2
        best_actions = []
        for action in legal_actions:
            next_board = list(board)
            next_board[action] = player
            next_board = tuple(next_board)

            if _winner(next_board) == player:
                value = 1
            elif not _legal(next_board):
                value = 0
            else:
                value = -_negamax(next_board)

            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)
