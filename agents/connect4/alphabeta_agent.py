import math
import random
import time

ROWS, COLS = 6, 7
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
# center-out order: center columns are stronger opening moves and prune far
# more of the tree when searched first, so ordering matters more than raw
# column index for alpha-beta's pruning efficiency
COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]

WIN_SCORE = 1_000_000


def _all_windows():
    windows = []
    for r in range(ROWS):
        for c in range(COLS - 3):
            windows.append([(r, c + i) for i in range(4)])
    for c in range(COLS):
        for r in range(ROWS - 3):
            windows.append([(r + i, c) for i in range(4)])
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            windows.append([(r + i, c + i) for i in range(4)])
            windows.append([(r + 3 - i, c + i) for i in range(4)])
    return windows


WINDOWS = _all_windows()


def _legal_columns(board):
    return [c for c in COLUMN_ORDER if board[0][c] == 0]


def _drop(board, col, player):
    for row in range(ROWS - 1, -1, -1):
        if board[row][col] == 0:
            board[row][col] = player
            return row
    return None


def _undo(board, col, row):
    board[row][col] = 0


def _check_win_from(board, row, col, player):
    for dr, dc in DIRECTIONS:
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player:
            count += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player:
            count += 1
            r -= dr
            c -= dc
        if count >= 4:
            return True
    return False


def _score_window(cells, player):
    opp = -player
    p_count = cells.count(player)
    o_count = cells.count(opp)

    if p_count and o_count:
        return 0
    if p_count == 3:
        return 50
    if p_count == 2:
        return 10
    if p_count == 1:
        return 1
    if o_count == 3:
        return -50
    if o_count == 2:
        return -10
    if o_count == 1:
        return -1
    return 0


def _evaluate(board, player):
    score = 0
    for window in WINDOWS:
        cells = [board[r][c] for r, c in window]
        score += _score_window(cells, player)

    center_count = sum(1 for r in range(ROWS) if board[r][3] == player)
    score += center_count * 3
    return score


def _infer_current_player(board):
    ones = sum(row.count(1) for row in board)
    neg_ones = sum(row.count(-1) for row in board)
    return 1 if ones == neg_ones else -1


class AlphaBetaAgent:
    """Depth-limited negamax with alpha-beta pruning, center-out move ordering,
    and a per-move transposition table. With depth=None (the default), searches
    iteratively deeper until time_limit runs out, always returning the best
    move found by the deepest fully-completed depth - this adapts search
    strength to however much CPU time is available rather than committing to
    a fixed depth that might be too slow (mid-game) or too shallow (endgame).

    Classical alpha-beta search doesn't parallelize onto a GPU the way batched
    neural-net inference does - it's inherently sequential, branch-heavy tree
    traversal, not a matrix operation - so this stays CPU-only by design;
    the efficiency work here is algorithmic (pruning, ordering, memoization,
    mutating the board in place instead of copying at every node) rather than
    hardware-accelerated.
    """

    def __init__(self, depth=None, time_limit=2.0):
        self.depth = depth
        self.time_limit = time_limit
        self.transposition_table = {}

    def choose_action(self, state, legal_actions, current_player=None, epsilon=0.0):
        legal_actions = list(legal_actions)

        if random.random() < epsilon:
            return random.choice(legal_actions)

        if current_player is None:
            current_player = _infer_current_player(state)

        board = [list(int(v) for v in row) for row in state]
        self.transposition_table = {}

        if self.depth is not None:
            _, best_action = self._negamax(board, self.depth, -math.inf, math.inf, current_player, None)
            return best_action

        deadline = time.time() + self.time_limit
        best_action = legal_actions[0]
        depth = 1
        while True:
            try:
                _, action = self._negamax(board, depth, -math.inf, math.inf, current_player, deadline)
            except TimeoutError:
                break
            best_action = action
            depth += 1
        return best_action

    def _negamax(self, board, depth, alpha, beta, player, deadline):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError

        legal = _legal_columns(board)
        if not legal:
            return 0, None

        key = (tuple(tuple(row) for row in board), depth, player)
        cached = self.transposition_table.get(key)
        if cached is not None:
            return cached, None

        if depth == 0:
            value = _evaluate(board, player)
            return value, None

        best_value = -math.inf
        best_action = legal[0]

        for col in legal:
            row = _drop(board, col, player)

            if _check_win_from(board, row, col, player):
                value = WIN_SCORE + depth
            else:
                child_value, _ = self._negamax(board, depth - 1, -beta, -alpha, -player, deadline)
                value = -child_value

            _undo(board, col, row)

            if value > best_value:
                best_value = value
                best_action = col

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        self.transposition_table[key] = best_value
        return best_value, best_action