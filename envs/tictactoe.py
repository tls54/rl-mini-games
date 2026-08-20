# TicTacToe env
import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.player = int(1)
        self.game_over = False

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.game_over = False
        return self.board.copy()

    def legal_actions(self):
        mask = self.board == 0
        return np.where(mask)[0]

    def step(self, action):
        reward = 0
        info = {}
        if action in self.legal_actions():
            self.board[action] = self.player

            winning_indexes =[
                (0,1,2), (3,4,5), (6,7,8),
                (0,3,6), (1,4,7), (2,5,8),
                (0,4,8), (2,4,6)
                ]
            
            sums = [self.board[list(idx)].sum() for idx in winning_indexes]
            
            if 3 in sums or -3 in sums:
                self.game_over = True
                reward = 1
            
            if len(self.legal_actions()) < 1:
                self.game_over = True

        else: raise ValueError(f"illegal action {action}")

        self.player *= -1

        return self.board.copy(), reward, self.game_over, info

    def render(self):
        pretty_board = []
        for i, v in enumerate(self.board):
            if v == 1:
                pretty_board.append('x')
            elif v == -1:
                pretty_board.append('o')
            else:
                pretty_board.append(str(i))

        pretty_board = np.array(pretty_board)

        print(pretty_board.reshape(3,3))
    
    @property
    def current_player(self):
        return self.player