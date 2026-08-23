import numpy as np

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
            for i in range(self.board.shape[0]):
                y_axis = self.board.shape[0] - i - 1
                if self.board[y_axis, action] == 0:
                    self.board[y_axis, action] = self.player
                    break
        
        # check for winner logic here

        if len(self.legal_actions()) < 1:
            self.game_over = True

        else: raise ValueError(f"illegal action {action}")

