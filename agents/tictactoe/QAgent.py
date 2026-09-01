import pickle
import random
from utils.symmetry import canonical, invert

class QAgent:
    def __init__(self, q_table=None, alpha=0.1, gamma=0.9):
        self.q_table = q_table if q_table else {}
        self.alpha = alpha
        self.gamma = gamma

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)


    def choose_action(self, state, legal_actions, current_player=None, epsilon=0.0):
        if random.random() < epsilon:
            return random.choice(legal_actions)

        canon_state, transform = canonical(state)
        inv = invert(transform)                    # shape (9,)
        key = tuple(int(x) for x in canon_state)

        action_values = self.q_table.get(key, {})
        values = {a: action_values.get(int(inv[a]), 0.0) for a in legal_actions}

        best = max(values.values())
        choices = [a for a, v in values.items() if v == best]

        return random.choice(choices)
        



    def update(self, state, action, next_legal_action, reward, next_state, done):
        canon_state, transform = canonical(state)
        inv = invert(transform)
        key = tuple(int(x) for x in canon_state)

        current_estimate = self.q_table.get(key, {}).get(int(inv[action]), 0.0)

        if not done: 
            next_canon_state, next_transform = canonical(next_state)
            next_key = tuple(int(x) for x in next_canon_state)
            next_inv = invert(next_transform)
            next_action_values = self.q_table.get(next_key, {})
            values = {a: next_action_values.get(int(next_inv[a]), 0.0) for a in next_legal_action}

            best_next = max(values.values())

            target = reward + self.gamma * best_next

        
        else:
            target = reward

        new_estimate = current_estimate + self.alpha * (target - current_estimate)

        self.q_table.setdefault(key, {})[int(inv[action])] = new_estimate


# Dummy table, just to see the shape:
# - keys are states, as hashable tuples of the flat 9-cell board (1=x, -1=o, 0=empty)
# - values are dicts of {action: q_value} for every action tried from that state so far
#   (an action not present yet just hasn't been explored from this state)
dummy_q_table = {
    (0, 0, 0, 0, 0, 0, 0, 0, 0): {
        0: 0.1,
        4: 0.6,   # center looks best so far from an empty board
        8: 0.05,
    },
    (1, 0, 0, 0, -1, 0, 0, 0, 0): {
        1: -0.2,
        2: 0.3,
        7: 0.15,
    },
}


