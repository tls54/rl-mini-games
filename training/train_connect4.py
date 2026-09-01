import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from torch.optim import Adam

from agents.connect4.rollout import RolloutCollector
from agents.connect4.ppo_agent import PPOAgent, ActorParams, CriticParams
from agents.connect4.ppo_networks import CriticNet, PiTheta
from agents.connect4.gae import calc_gae

FOUNDATIONS_DIR = Path("checkpoints/connect4/foundations")


def foundation_dir(foundation_name):
    return FOUNDATIONS_DIR / foundation_name


def latest_checkpoint_dir(foundation_name):
    checkpoints_root = foundation_dir(foundation_name) / "checkpoints"
    if not checkpoints_root.exists():
        return None

    steps = []
    for p in checkpoints_root.iterdir():
        if p.is_dir() and p.name.startswith("step_"):
            try:
                steps.append((int(p.name.split("_")[1]), p))
            except ValueError:
                continue

    if not steps:
        return None
    return max(steps, key=lambda pair: pair[0])[1]


def parse_int_list(s):
    return tuple(int(x) for x in s.split(","))


def load_foundation_config(foundation_name):
    path = foundation_dir(foundation_name) / "foundation_config.json"
    with open(path) as f:
        return json.load(f)


def save_foundation_config(foundation_name, args):
    fdir = foundation_dir(foundation_name)
    fdir.mkdir(parents=True, exist_ok=True)

    config = {
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "gamma": args.gamma,
        "lambda_value": args.lam,
        "eps": args.eps,
        "opponent": args.opponent,
        "actor_conv_channels": list(args.actor_conv_channels),
        "actor_fc_hidden": list(args.actor_fc_hidden),
        "critic_conv_channels": list(args.critic_conv_channels),
        "critic_fc_hidden": list(args.critic_fc_hidden),
    }
    with open(fdir / "foundation_config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_checkpoint(foundation_name, step, actor_params, critic_params):
    ckpt_dir = foundation_dir(foundation_name) / "checkpoints" / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(actor_params.actor.state_dict(), ckpt_dir / "actor.pt")
    torch.save(critic_params.critic.state_dict(), ckpt_dir / "critic.pt")
    torch.save(actor_params.optimizer.state_dict(), ckpt_dir / "actor_optimizer.pt")
    torch.save(critic_params.optimizer.state_dict(), ckpt_dir / "critic_optimizer.pt")

    config = {"step": step}
    with open(ckpt_dir / "checkpoint_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"saved checkpoint to {ckpt_dir}")


def load_checkpoint(ckpt_dir, actor_params, critic_params):
    actor_params.actor.load_state_dict(torch.load(ckpt_dir / "actor.pt"))
    critic_params.critic.load_state_dict(torch.load(ckpt_dir / "critic.pt"))
    actor_params.optimizer.load_state_dict(torch.load(ckpt_dir / "actor_optimizer.pt"))
    critic_params.optimizer.load_state_dict(torch.load(ckpt_dir / "critic_optimizer.pt"))

    with open(ckpt_dir / "checkpoint_config.json") as f:
        config = json.load(f)
    return config


def train(args):
    ckpt_dir = latest_checkpoint_dir(args.foundation_name) if args.resume else None

    if ckpt_dir is not None:
        # architecture must match the saved weights - read it from the foundation's
        # own config rather than trusting whatever CLI flags were passed this run
        fconfig = load_foundation_config(args.foundation_name)
        actor_conv_channels = tuple(fconfig["actor_conv_channels"])
        actor_fc_hidden = tuple(fconfig["actor_fc_hidden"])
        critic_conv_channels = tuple(fconfig["critic_conv_channels"])
        critic_fc_hidden = tuple(fconfig["critic_fc_hidden"])
    else:
        if args.resume:
            print(f"--resume set but no checkpoints found for '{args.foundation_name}', starting fresh")
        actor_conv_channels = args.actor_conv_channels
        actor_fc_hidden = args.actor_fc_hidden
        critic_conv_channels = args.critic_conv_channels
        critic_fc_hidden = args.critic_fc_hidden
        save_foundation_config(args.foundation_name, args)

    actor_params = ActorParams(
        actor=PiTheta(conv_channels=actor_conv_channels, fc_hidden_sizes=actor_fc_hidden),
        optimizer_cls=Adam,
        learning_rate=args.actor_lr,
    )
    critic_params = CriticParams(
        critic=CriticNet(conv_channels=critic_conv_channels, fc_hidden_sizes=critic_fc_hidden),
        optimizer_cls=Adam,
        learning_rate=args.critic_lr,
    )
    agent = PPOAgent(actor=actor_params, critic=critic_params)
    collector = RolloutCollector(num_envs=args.num_envs)

    total_steps_done = 0

    if ckpt_dir is not None:
        config = load_checkpoint(ckpt_dir, actor_params, critic_params)
        total_steps_done = config["step"]
        print(f"resumed from {ckpt_dir} (step {total_steps_done})")

    total_loops = args.total_steps // args.num_steps

    for loop in range(1, total_loops + 1):
        loop_start = time.time()

        states, actions, log_probs, rewards, dones, values, bootstrap_values = collector.collect(
            actor=actor_params.actor, critic=critic_params.critic, num_steps=args.num_steps
        )

        num_envs = len(collector.envs)
        steps_per_env = len(states) // num_envs
        rewards_arr = np.array(rewards).reshape(steps_per_env, num_envs)
        values_arr = torch.stack(values).squeeze().numpy().reshape(steps_per_env, num_envs)
        dones_arr = np.array(dones).reshape(steps_per_env, num_envs)

        advantages, returns = calc_gae(
            rewards=rewards_arr,
            values=values_arr,
            dones=dones_arr,
            bootstrap_values=bootstrap_values,
            num_envs=num_envs,
            gamma=args.gamma,
            lambda_value=args.lam,
        )

        agent.update(
            states=states,
            actions=actions,
            old_log_probs=log_probs,
            advantages=advantages,
            returns=returns,
            epochs=args.epochs,
            minibatch_size=args.minibatch_size,
            eps=args.eps,
        )

        total_steps_done += len(states)
        wins = sum(1 for r in rewards if r == 1)
        losses = sum(1 for r in rewards if r == -1)
        elapsed = time.time() - loop_start
        print(
            f"loop {loop}/{total_loops} (total steps {total_steps_done}): "
            f"{len(states)} collected, {wins}W/{losses}L, {elapsed:.1f}s"
        )

        if args.checkpoint_every and loop % args.checkpoint_every == 0:
            save_checkpoint(args.foundation_name, total_steps_done, actor_params, critic_params)

    save_checkpoint(args.foundation_name, total_steps_done, actor_params, critic_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--opponent", choices=["self"], default="self")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--actor-conv-channels", type=parse_int_list, default=(2, 6, 12, 24))
    parser.add_argument("--actor-fc-hidden", type=parse_int_list, default=(128, 64))
    parser.add_argument("--critic-conv-channels", type=parse_int_list, default=(2, 6, 12, 24))
    parser.add_argument("--critic-fc-hidden", type=parse_int_list, default=(128, 64, 16))
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()