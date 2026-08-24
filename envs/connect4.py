import numpy as np


def get_diag(action, y_axis, board):
    k = np.arange(-3, 4)
    diag_rows1 = y_axis + k
    diag_cols1 = action + k

    valid1 = (diag_rows1 >= 0) & (diag_rows1 < board.shape[0]) & (diag_cols1 >= 0) & (diag_cols1 < board.shape[1])
    safe_rows1 = np.clip(diag_rows1, 0, board.shape[0]-1)
    safe_cols1 = np.clip(diag_cols1, 0, board.shape[1]-1)
    diag_line1 = board[safe_rows1, safe_cols1]
    diag_line1 = np.where(valid1, diag_line1, 0)


    diag_rows2 = y_axis + k
    diag_cols2 = action - k

    valid2 = (diag_rows2 >= 0) & (diag_rows2 < board.shape[0]) & (diag_cols2 >= 0) & (diag_cols2 < board.shape[1])
    safe_rows2 = np.clip(diag_rows2, 0, board.shape[0]-1)
    safe_cols2 = np.clip(diag_cols2, 0, board.shape[1]-1)
    diag_line2 = board[safe_rows2, safe_cols2]
    diag_line2 = np.where(valid2, diag_line2, 0)

    return diag_line1, diag_line2 



def check_win(board, action, y_axis):
    win_check_array = np.ones(4)

    col_convolve = np.convolve(board[:, action], win_check_array)
    if 4 in np.abs(col_convolve):
        return True

    row_convolve = np.convolve(board[y_axis, :], win_check_array)
    if 4 in np.abs(row_convolve):
        return True
    
    diag_arrays = get_diag(action, y_axis, board)

    diag1_convolve = np.convolve(diag_arrays[0], win_check_array)
    diag2_convolve = np.convolve(diag_arrays[1], win_check_array)

    if 4 in np.abs(diag1_convolve) or 4 in np.abs(diag2_convolve):
        return True

    else: return False

class Connect4:
    def __init__(self):
        self.board = np.zeros(shape=(6,7), dtype=int)
        self.player = int(1)
        self.game_over = False

    def reset(self):
        self.board = np.zeros(shape=(6,7), dtype=int)
        self.player = int(1)
        self.game_over = False
        return self.board.copy()

    def legal_actions(self):
        mask = self.board[0] == 0
        return np.where(mask)[0]
    
    def step(self, action):
        reward = 0
        info = {}

        if action in self.legal_actions():
            for i in range(self.board.shape[0]): # gravity loop
                y_axis = self.board.shape[0] - i - 1
                if self.board[y_axis, action] == 0:
                    self.board[y_axis, action] = self.player
                    break

        else: raise ValueError(f"illegal action {action}")

        win = check_win(self.board, action, y_axis)

        if win:
            reward = 1
            self.game_over = True

        if len(self.legal_actions()) < 1:
            self.game_over = True

        self.player *= -1

        return self.board.copy(), reward, self.game_over, info

    def render(self):
        symbols = {1: 'x', -1: 'o', 0: '.'}
        for row in self.board:
            print(' '.join(symbols[v] for v in row))
        print(' '.join(str(c) for c in range(self.board.shape[1])))


    @property
    def current_player(self):
        return self.player