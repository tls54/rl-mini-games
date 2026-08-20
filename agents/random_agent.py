import random


class RandomAgent:
    def choose_action(self, state, legal_actions, epsilon=0.0):
        return random.choice(list(legal_actions))
