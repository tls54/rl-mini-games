import numpy as np


# Shape key used this file:
#   (3, 3)  - a board/permutation laid out as a grid
#   (9,)    - the same thing flattened to match env board state


def build_permutations() -> tuple[np.ndarray, ...]:
    """Returns the 8 index permutations (each shape (3, 3)) of the dihedral
    group for a 3x3 grid: identity, 3 rotations, 4 reflections."""
    grid = np.arange(9).reshape(3, 3)
    return (
        grid,
        np.rot90(grid),
        np.rot90(np.rot90(grid)),
        np.rot90(np.rot90(np.rot90(grid))),
        np.flip(grid, axis=1),
        np.flip(grid, axis=0),
        grid.T,
        np.rot90(np.rot90(grid)).T
        )


PERMUTATIONS: tuple[np.ndarray, ...] = build_permutations()

PERMUTATIONS_FLAT = np.array([p.flatten() for p in PERMUTATIONS])  # shape (8, 9), computed once

def get_rotations(state):
    return state.flatten()[PERMUTATIONS_FLAT]  # shape (8, 9), all 8 in one shot



def canonical(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """state: shape (9,).

    Returns (canonical_state, transformation):
      canonical_state: shape (9,), the lexicographically smallest of the 8 symmetric variants.
      transformation:  shape (3, 3), the permutation (one of PERMUTATIONS) that produced it.
    """
    states = get_rotations(state)
    permutation_index = min(range(len(states)), key=lambda i: tuple(states[i]))
    canonical_state = states[permutation_index]
    canonical_transformation = PERMUTATIONS[permutation_index]

    return canonical_state, canonical_transformation


def invert(perm: np.ndarray) -> np.ndarray:
    """perm: shape (3, 3) or (9,). Returns the inverse permutation, shape (9,)."""
    return np.argsort(perm.flatten())