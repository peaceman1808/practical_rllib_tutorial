"""
This trainer is based off of PPO and points you in the direction of how to
customize it.

PPO is as a policy gradient method called Proximal Policy Optimization.
Policy gradient means you update the parameters of the policy directly instead of using
a Q table or something like that. Good for continuous actions.
"""

from ray.rllib.algorithms import ppo
from ray.train.rl import RLTrainer

# PPO default config builds on DEFAULT_CONFIG here
# https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/agents/trainer.py
config = ppo.PPOConfig

# Further update with config
config.training(
    {
    # The batch size collected for each worker
    rollout_fragment_length: 1000,
    # Can be "complete_episodes" or "truncate_episodes"
    "batch_mode": "complete_episodes",
    "simple_optimizer": True,
    "framework": "torch",
    })


class YourTrainer:

    trainer = RLTrainer(
        algorithm="PPO",
        config=config,
    )
    result = trainer.fit()
