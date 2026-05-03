"""
Abernethy & Kale (2013) MMMW benchmark — Section 5.1.

Multiplicative-weights meta-algorithm over K fixed-spread experts.
"""

from __future__ import annotations

import numpy as np

from market_maker import config
from market_maker.data.loader import DayData
from market_maker.data.simulator import LOBSimulator, StepResult
from market_maker.agent.reward import compute
from market_maker.agent.agent import EpisodeResult

EXPERT_THETAS = [1, 2, 3, 4, 5]
K = len(EXPERT_THETAS)


class MMMMWAgent:

    def __init__(self, eta_mw: float = 0.01):
        self.eta_mw = eta_mw
        self.sim    = LOBSimulator()

    def run_episode(self, day: DayData) -> EpisodeResult:
        self.sim.reset(day)
        res     = EpisodeResult()
        weights = np.ones(K, dtype=np.float64)
        selected = 0
        self._place(selected)

        while not self.sim.done:
            step: StepResult = self.sim.step()
            sc  = step.spread_scale
            mid = step.prev_mid_price

            # Counterfactual PnL for all experts
            cf = np.zeros(K)
            for k, theta in enumerate(EXPERT_THETAS):
                rc_k = compute(step.matched_ask, step.matched_bid,
                               mid + theta * sc, mid - theta * sc,
                               step.mid_price, step.prev_mid_price, self.sim.inventory)
                cf[k] = rc_k.pnl

            weights *= np.exp(self.eta_mw * cf)
            weights /= weights.sum()

            # Actual PnL from selected expert
            p_ask = self.sim.ask_order.price if self.sim.ask_order else step.event.best_ask
            p_bid = self.sim.bid_order.price if self.sim.bid_order else step.event.best_bid
            rc = compute(step.matched_ask, step.matched_bid, p_ask, p_bid,
                         step.mid_price, step.prev_mid_price, self.sim.inventory)

            res.total_pnl    += rc.pnl
            res.total_reward += rc.pnl
            res.n_steps      += 1
            res.inv_path.append(self.sim.inventory)
            res.spread_history.append(step.event.spread)

            if not step.done:
                selected = int(np.random.choice(K, p=weights))
                if abs(self.sim.inventory) >= config.MAX_INVENTORY:
                    self.sim.place_market_order()
                else:
                    self._place(selected)

        return res

    def _place(self, idx: int) -> None:
        theta = EXPERT_THETAS[idx]
        self.sim.place_orders(theta, theta)
