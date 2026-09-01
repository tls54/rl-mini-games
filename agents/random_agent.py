import random


class RandomAgent:
    def choose_action(self, state, legal_actions, current_player=None, epsilon=0.0):
        return random.choice(list(legal_actions))
