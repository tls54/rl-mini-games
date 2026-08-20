import argparse

from envs.tictactoe import TicTacToe
from agents.random_agent import RandomAgent
from agents.QAgent import QAgent

SYMBOLS = {1: "x", -1: "o"}


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
            action = agent.choose_action(state, env.legal_actions())
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
        agent = agents[env.current_player]
        action = agent.choose_action(state, env.legal_actions())
        state, reward, done, info = env.step(action)
        env.render()

    if reward == 1:
        print(f"Player {-env.current_player} wins!")
    else:
        print("Draw!")


def load_agent(checkpoint):
    if checkpoint is None:
        return RandomAgent()
    agent = QAgent()
    agent.load(checkpoint)
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human", "agent"], default="human")
    parser.add_argument("--checkpoint", default=None, help="path to a trained QAgent q_table.pkl; omit for a random agent")
    args = parser.parse_args()

    env = TicTacToe()
    agent = load_agent(args.checkpoint)

    if args.mode == "human":
        play_human_vs_agent(env, agent)
    else:
        play_agent_vs_agent(env, agent, load_agent(args.checkpoint))


if __name__ == "__main__":
    main()
