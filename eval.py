import argparse

from envs.tictactoe import TicTacToe
from play import load_agent
from utils.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", default=None, help="path to agent A's q_table.pkl; omit for a random agent")
    parser.add_argument("--checkpoint-b", default=None, help="path to agent B's q_table.pkl; omit for a random agent")
    parser.add_argument("--games", type=int, default=500)
    args = parser.parse_args()

    env = TicTacToe()
    agent_a = load_agent(args.checkpoint_a)
    agent_b = load_agent(args.checkpoint_b)

    metrics = evaluate(agent_a, agent_b, env, n_games=args.games)

    print(f"A: {args.checkpoint_a or 'random'}")
    print(f"B: {args.checkpoint_b or 'random'}")
    print(f"over {args.games} games (sides alternated):")
    print(f"  A win rate:  {metrics['win_rate']:.3f}")
    print(f"  draw rate:   {metrics['draw_rate']:.3f}")
    print(f"  A loss rate: {metrics['loss_rate']:.3f}")


if __name__ == "__main__":
    main()
