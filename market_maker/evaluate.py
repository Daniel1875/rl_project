"""
Out-of-sample evaluation — Section 5 of Spooner et al. (2018).

Prints a table of ND-PnL and MAP for all agents on the test set,
matching the format of Table 6 in the paper.

Usage:
    python -m market_maker.evaluate --parquet data/BIIB.parquet
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from market_maker import config
from market_maker.data.loader import load_parquet, train_val_test_split
from market_maker.agent.agent import EpisodeResult
from market_maker.benchmarks.fixed_spread import FixedSpreadAgent, RandomAgent
from market_maker.benchmarks.mmmw import MMMMWAgent
from market_maker.train import train


def _stats(results: List[EpisodeResult]) -> Dict[str, float]:
    nd   = [r.normalised_daily_pnl() for r in results]
    maps = [r.mean_abs_position      for r in results]
    return dict(nd_mean=np.mean(nd), nd_std=np.std(nd),
                map_mean=np.mean(maps), map_std=np.std(maps))


def _row(label: str, s: Dict) -> None:
    print(f"  {label:<35s}  "
          f"{s['nd_mean']:+10.4f} +/- {s['nd_std']:8.4f}  "
          f"{s['map_mean']:10.1f} +/- {s['map_std']:8.1f}")


def evaluate(parquet: str, reward: str = "asymm",
             eta: float = config.DAMPING_ETA, seed: int = 0) -> None:
    days = load_parquet(parquet)
    _, _, test_days = train_val_test_split(days)
    print(f"\nTest set: {len(test_days)} days\n")

    # Train consolidated agent
    agent = train(parquet, reward=reward, eta=eta, seed=seed)

    # Collect test results
    rl_res = [agent.run_episode(d, learn=False) for d in test_days]

    # Benchmarks
    bench: Dict[str, List[EpisodeResult]] = {}
    for theta in [1, 2, 3, 4, 5]:
        fa = FixedSpreadAgent(theta)
        bench[f"Fixed theta={theta}"] = [fa.run_episode(d) for d in test_days]
    ra = RandomAgent()
    bench["Random"] = [ra.run_episode(d) for d in test_days]
    mw = MMMMWAgent()
    bench["MMMW (Abernethy & Kale)"] = [mw.run_episode(d) for d in test_days]

    # Print table
    print(f"\n  {'Strategy':<35s}  {'ND-PnL (mean +/- std)':>25s}  {'MAP (mean +/- std)':>22s}")
    print("  " + "-" * 90)
    for label, res in bench.items():
        _row(label, _stats(res))
    _row(f"Consolidated RL ({reward}, eta={eta})", _stats(rl_res))
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--reward",  default="asymm", choices=["pnl","symm","asymm"])
    p.add_argument("--eta",     type=float, default=config.DAMPING_ETA)
    p.add_argument("--seed",    type=int,   default=0)
    args = p.parse_args()
    evaluate(args.parquet, args.reward, args.eta, args.seed)


if __name__ == "__main__":
    main()
