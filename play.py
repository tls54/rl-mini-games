from envs.tictactoe import TicTacToe
from envs.connect4 import Connect4
from agents.random_agent import RandomAgent
from agents.tictactoe.QAgent import QAgent
from agents.tictactoe.minimax_agent import MinimaxAgent
from agents.connect4.ppo_eval_agent import PPOPlayer
from utils.checkpoints import list_tictactoe_checkpoints, list_connect4_checkpoints, load_connect4_actor

SYMBOLS = {1: "x", -1: "o"}


def prompt_choice(title, options):
    """options: list of (label, value). Returns the chosen value."""
    print(title)
    for i, (label, _) in enumerate(options, start=1):
        print(f"  [{i}] {label}")
    while True:
        raw = input(f"> ")
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][1]
        except ValueError:
            pass
        print(f"Enter a number from 1 to {len(options)}.")


def select_game():
    return prompt_choice("Select game:", [
        ("Tic-Tac-Toe", "tictactoe"),
        ("Connect 4", "connect4"),
    ])


def make_env(game):
    return TicTacToe() if game == "tictactoe" else Connect4()


def select_opponent(game, label):
    if game == "tictactoe":
        options = [("Random", "random"), ("Minimax", "minimax")]
        checkpoints = list_tictactoe_checkpoints()
        options += [(str(p.relative_to("checkpoints/tictactoe")), p / "q_table.pkl") for p in checkpoints]
    else:
        options = [("Random", "random")]
        checkpoints = list_connect4_checkpoints()
        options += [(str(p.relative_to("checkpoints/connect4/foundations")), p) for p in checkpoints]

    choice = prompt_choice(f"Select {label}:", options)
    return load_agent(game, choice)


def load_agent(game, choice):
    if choice == "random" or choice is None:
        return RandomAgent()
    if choice == "minimax":
        return MinimaxAgent()

    if game == "tictactoe":
        agent = QAgent()
        agent.load(choice)
        return agent

    actor = load_connect4_actor(choice)
    return PPOPlayer(actor)


def get_human_action(env):
    symbol = SYMBOLS[env.current_player]
    while True:
        raw = input(f"Your move ({symbol}) {env.legal_actions().tolist()}: ")
        try:
            action = int(raw)
        except ValueError:
            print("Enter a number.")
            continue
        try:
            return env.step(action)
        except ValueError as e:
            print(e)


def play_human_vs_agent(env, agent):
    state = env.reset()
    env.render()
    done = False

    # human is player 1, agent is player -1
    while not done:
        if env.current_player == 1:
            state, reward, done, info = get_human_action(env)
        else:
            action = agent.choose_action(state, env.legal_actions(), current_player=env.current_player)
            state, reward, done, info = env.step(action)
        env.render()
        print()

    if reward == 1:
        winner = "Human" if env.current_player == -1 else "Agent"
        print(f"{winner} wins!")
    else:
        print("Draw!")


def play_agent_vs_agent(env, agent_a, agent_b):
    state = env.reset()
    env.render()
    done = False
    agents = {1: agent_a, -1: agent_b}

    while not done:
        player = env.current_player
        action = agents[player].choose_action(state, env.legal_actions(), current_player=player)
        state, reward, done, info = env.step(action)
        env.render()

    if reward == 1:
        print(f"Player {-env.current_player} wins!")
    else:
        print("Draw!")


def main():
    game = select_game()
    env = make_env(game)

    mode = prompt_choice("Select mode:", [
        ("Human vs Agent", "human"),
        ("Agent vs Agent", "agent"),
    ])

    if mode == "human":
        opponent = select_opponent(game, "opponent (plays second)")
        play_human_vs_agent(env, opponent)
    else:
        agent_a = select_opponent(game, "player 1")
        agent_b = select_opponent(game, "player 2")
        play_agent_vs_agent(env, agent_a, agent_b)


if __name__ == "__main__":
    main()