import random

import numpy as np
import torch

from agents.connect4.encoding import encode_board_state


class PPOPlayer:
    """Wraps a trained PiTheta actor for play/eval, matching the same
    choose_action(state, legal_actions, current_player, epsilon) interface
    used by RandomAgent/QAgent/MinimaxAgent. Always plays greedily (argmax) -
    epsilon is only used to occasionally fall back to a random legal move.
    """

    def __init__(self, actor):
        self.actor = actor
        self.actor.eval()

    def choose_action(self, state, legal_actions, current_player=None, epsilon=0.0):
        legal_actions = list(legal_actions)

        if random.random() < epsilon:
            return random.choice(legal_actions)

        player_state, opponent_state = encode_board_state(state, current_player)
        stacked = np.stack([player_state, opponent_state], axis=0)
        x = torch.from_numpy(stacked).float().unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(x)

        return int(torch.argmax(logits[0]).item())