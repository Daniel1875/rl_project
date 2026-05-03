"""
Training loop — Section 5 of Spooner et al. (2018).

Usage:
    python -m market_maker.train --parquet data/BIIB.parquet
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from market_maker import config
from market_maker.data.loader import load_parquet, train_val_test_split
from market_maker.agent.agent import MarketMakingAgent
from market_maker.agent.reward import (
    PnLReward, SymmetricDampedReward, AsymmetricDampedReward,
)


def make_reward(name: str, eta: float = config.DAMPING_ETA):
    return {"pnl": PnLReward,
            "symm": lambda: SymmetricDampedReward(eta),
            "asymm": lambda: AsymmetricDampedReward(eta)}[name]()


def train(parquet: str, n_episodes: int = config.TRAIN_EPISODES,
          reward: str = "asymm", eta: float = config.DAMPING_ETA,
          seed: int = 0) -> MarketMakingAgent:

    np.random.seed(seed)
    random.seed(seed)

    print(f"Loading {parquet} ...")
    days = load_parquet(parquet)
    train_days, val_days, _ = train_val_test_split(days)
    print(f"  {len(train_days)} train / {len(val_days)} val days")

    agent = MarketMakingAgent(reward_fn=make_reward(reward, eta))

    print(f"Training {n_episodes} episodes  (reward={reward}, eta={eta})\n")
    window = []
    for ep in range(n_episodes):
        day = random.choice(train_days)
        r   = agent.run_episode(day, learn=True)
        window.append(r.total_pnl)

        if (ep + 1) % 50 == 0:
            print(f"  ep {ep+1:5d}/{n_episodes}  "
                  f"PnL(50-ep avg): {np.mean(window[-50:]):+.4f}  "
                  f"eps={agent.learner.epsilon:.4f}")

    # Validation pass
    print("\nValidation:")
    nd = [agent.run_episode(d, learn=False).normalised_daily_pnl() for d in val_days]
    print(f"  ND-PnL  {np.mean(nd):+.4f} +/- {np.std(nd):.4f}")
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",   required=True)
    p.add_argument("--episodes",  type=int,   default=config.TRAIN_EPISODES)
    p.add_argument("--reward",    default="asymm", choices=["pnl","symm","asymm"])
    p.add_argument("--eta",       type=float, default=config.DAMPING_ETA)
    p.add_argument("--seed",      type=int,   default=0)
    args = p.parse_args()
    train(args.parquet, args.episodes, args.reward, args.eta, args.seed)


if __name__ == "__main__":
    main()
