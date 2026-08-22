import argparse
import json
import time
from pathlib import Path

from agents.QAgent import QAgent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from envs.tictactoe import TicTacToe
from utils.evaluate import evaluate


def run_episode(env, agent, epsilon, loss_reward, draw_reward):
    done = False
    env.reset()
    pending = {1: None, -1: None}

    while not done:
        player = env.current_player
        state = env.board.copy()
        action = agent.choose_action(state, env.legal_actions(), epsilon)

        if pending[player] is not None:
            prev_state, prev_action = pending[player]
            agent.update(prev_state, prev_action, env.legal_actions(), reward=0, next_state=state, done=False)

        next_state, reward, done, info = env.step(action)
        pending[player] = (state, action)

        if done:
            agent.update(state, action, env.legal_actions(), reward=reward, next_state=next_state, done=True)
            pending[player] = None

            remaining_reward = loss_reward if reward == 1 else draw_reward
            remaining_state, remaining_action = pending[-player]
            agent.update(remaining_state, remaining_action, env.legal_actions(), remaining_reward, next_state=state, done=True)


def run_episode_vs_opponent(env, agent, opponent, agent_player, epsilon, loss_reward, draw_reward):
    """Like run_episode, but only `agent` learns - `opponent` (e.g. MinimaxAgent)
    is fixed and never gets update() called on it. Only one pending transition
    is needed since only one side is ever waiting to be resolved."""
    done = False
    env.reset()
    pending = None

    while not done:
        player = env.current_player
        state = env.board.copy()

        if player == agent_player:
            action = agent.choose_action(state, env.legal_actions(), epsilon)
        else:
            action = opponent.choose_action(state, env.legal_actions(), epsilon=0.0)

        if player == agent_player and pending is not None:
            prev_state, prev_action = pending
            agent.update(prev_state, prev_action, env.legal_actions(), reward=0, next_state=state, done=False)

        next_state, reward, done, info = env.step(action)

        if player == agent_player:
            pending = (state, action)

        if done:
            if player == agent_player:
                # agent made the game-ending move - resolve immediately with the real reward
                agent.update(state, action, env.legal_actions(), reward=reward, next_state=next_state, done=True)
            elif pending is not None:
                # opponent ended the game - agent's last move gets the opposite outcome
                remaining_reward = loss_reward if reward == 1 else draw_reward
                remaining_state, remaining_action = pending
                agent.update(remaining_state, remaining_action, env.legal_actions(), remaining_reward, next_state=state, done=True)


def train(episodes, epsilon, alpha, gamma, loss_reward, draw_reward, eval_every, eval_games, run_name, opponent="self"):
    env = TicTacToe()
    agent = QAgent(alpha=alpha, gamma=gamma)
    baseline = RandomAgent()
    fixed_opponent = MinimaxAgent() if opponent == "minimax" else None

    eval_history = []

    for episode in range(1, episodes + 1):
        if fixed_opponent is None:
            run_episode(env, agent, epsilon, loss_reward, draw_reward)
        else:
            agent_player = 1 if episode % 2 == 1 else -1
            run_episode_vs_opponent(env, agent, fixed_opponent, agent_player, epsilon, loss_reward, draw_reward)

        if eval_every and episode % eval_every == 0:
            metrics = evaluate(agent, baseline, env, n_games=eval_games)
            metrics["episode"] = episode
            eval_history.append(metrics)
            print(
                f"episode {episode}/{episodes} "
                f"vs random -> win {metrics['win_rate']:.2f} "
                f"draw {metrics['draw_rate']:.2f} "
                f"loss {metrics['loss_rate']:.2f}"
            )

    run_dir = Path("checkpoints") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    agent.save(run_dir / "q_table.pkl")

    config = {
        "episodes": episodes,
        "epsilon": epsilon,
        "alpha": alpha,
        "gamma": gamma,
        "loss_reward": loss_reward,
        "draw_reward": draw_reward,
        "eval_every": eval_every,
        "eval_games": eval_games,
        "opponent": opponent,
        "eval_history": eval_history,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"saved checkpoint to {run_dir}")
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--loss-reward", type=float, default=-1.0)
    parser.add_argument("--draw-reward", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--opponent", choices=["self", "minimax"], default="self")
    args = parser.parse_args()

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")

    train(
        episodes=args.episodes,
        epsilon=args.epsilon,
        alpha=args.alpha,
        gamma=args.gamma,
        loss_reward=args.loss_reward,
        draw_reward=args.draw_reward,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        run_name=run_name,
        opponent=args.opponent,
    )


if __name__ == "__main__":
    main()
