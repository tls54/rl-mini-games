def encode_board_state(board, player):
    """Split a raw Connect4 board into (mover's pieces, opponent's pieces) channels."""
    player_state = (board == player).astype(int)
    opponent_state = (board == -player).astype(int)
    return player_state, opponent_state