from play import select_game, make_env, select_opponent, prompt_choice
from utils.evaluate import evaluate


def prompt_int(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("Enter a whole number, using default.")
        return default


def main():
    game = select_game()
    env = make_env(game)

    agent_a = select_opponent(game, "agent A")
    agent_b = select_opponent(game, "agent B")

    games = prompt_int("Number of games", 200)

    metrics = evaluate(agent_a, agent_b, env, n_games=games)

    print(f"\nover {games} games (sides alternated):")
    print(f"  A win rate:  {metrics['win_rate']:.3f}")
    print(f"  draw rate:   {metrics['draw_rate']:.3f}")
    print(f"  A loss rate: {metrics['loss_rate']:.3f}")


if __name__ == "__main__":
    main()