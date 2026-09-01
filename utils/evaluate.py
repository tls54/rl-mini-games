def evaluate(agent, opponent, env, n_games=200):
    """Play agent vs opponent with no exploration, alternating who moves first.

    Returns win/draw/loss rates from `agent`'s perspective.
    """
    wins = draws = losses = 0

    for i in range(n_games):
        state = env.reset()
        done = False
        # alternate who plays player 1 so first-move advantage doesn't bias the result
        agent_is_player_1 = i % 2 == 0
        agents = {1: agent, -1: opponent} if agent_is_player_1 else {1: opponent, -1: agent}

        while not done:
            mover = env.current_player
            action = agents[mover].choose_action(state, env.legal_actions(), current_player=mover, epsilon=0.0)
            state, reward, done, info = env.step(action)

        if reward == 0:
            draws += 1
        else:
            winner_was_agent = (mover == 1 and agent_is_player_1) or (mover == -1 and not agent_is_player_1)
            if winner_was_agent:
                wins += 1
            else:
                losses += 1

    return {
        "win_rate": wins / n_games,
        "draw_rate": draws / n_games,
        "loss_rate": losses / n_games,
    }
