import random

class QAgent:
    def __init__(self, q_table=None):
        self.q_table = q_table if q_table else {}

    def save(self, path): pass
    
    def load(self, path): pass


    def choose_action(self, state, legal_actions, epsilon=0.0):
        if random.random() < epsilon:
            return random.choice(legal_actions)

        key = tuple(int(x) for x in state)
        action_values = self.q_table.get(key, {})
        values = {a: action_values.get(a, 0.0) for a in legal_actions}
        best = max(values.values())
        choices = [a for a, v in values.items() if v == best]
        
        return random.choice(choices)
        



    def update(self, state, action, next_legal_action, reward, next_state, done):
        pass


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

state = (0, 0, 0, 0, 0, 0, 0, 0, 0)

key = tuple(int(x) for x in state)
action_values = dummy_q_table.get(key, {})


