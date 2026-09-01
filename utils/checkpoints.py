import json
from pathlib import Path

import torch


def list_tictactoe_checkpoints():
    root = Path("checkpoints/tictactoe")
    if not root.exists():
        return []
    return sorted(p.parent for p in root.glob("*/*/q_table.pkl"))


def list_connect4_checkpoints():
    root = Path("checkpoints/connect4/foundations")
    if not root.exists():
        return []

    results = []
    for foundation_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        checkpoints_root = foundation_dir / "checkpoints"
        if not checkpoints_root.exists():
            continue

        steps = []
        for step_dir in checkpoints_root.iterdir():
            if step_dir.is_dir() and step_dir.name.startswith("step_"):
                try:
                    steps.append((int(step_dir.name.split("_")[1]), step_dir))
                except ValueError:
                    continue

        results.extend(step_dir for _, step_dir in sorted(steps))

    return results


def load_connect4_actor(checkpoint_dir):
    """checkpoint_dir: a checkpoints/connect4/foundations/<name>/checkpoints/step_N dir."""
    from agents.connect4.ppo_networks import PiTheta

    foundation_dir = checkpoint_dir.parent.parent
    with open(foundation_dir / "foundation_config.json") as f:
        config = json.load(f)

    # older foundations predate configurable architecture and don't record it -
    # they were trained with PiTheta's original hardcoded defaults, so falling
    # back to its constructor defaults reproduces the right shape.
    kwargs = {}
    if "actor_conv_channels" in config:
        kwargs["conv_channels"] = tuple(config["actor_conv_channels"])
    if "actor_fc_hidden" in config:
        kwargs["fc_hidden_sizes"] = tuple(config["actor_fc_hidden"])

    actor = PiTheta(**kwargs)
    actor.load_state_dict(torch.load(checkpoint_dir / "actor.pt"))
    return actor