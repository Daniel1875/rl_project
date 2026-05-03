"""Fixed-spread and random benchmark strategies — Section 5.1."""

from __future__ import annotations

import numpy as np

from market_maker import config
from market_maker.data.loader import DayData
from market_maker.data.simulator import LOBSimulator, StepResult
from market_maker.agent.reward import compute
from market_maker.agent.agent import EpisodeResult


class FixedSpreadAgent:
    """Always quotes at a fixed symmetric distance θ from mid-price."""

    def __init__(self, theta: int = 2):
        self.theta = theta
        self.sim   = LOBSimulator()

    def run_episode(self, day: DayData) -> EpisodeResult:
        self.sim.reset(day)
        self.sim.place_orders(self.theta, self.theta)
        res = EpisodeResult()

        while not self.sim.done:
            step: StepResult = self.sim.step()
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
                if abs(self.sim.inventory) >= config.MAX_INVENTORY:
                    self.sim.place_market_order()
                else:
                    self.sim.place_orders(self.theta, self.theta)

        return res


class RandomAgent:
    """Picks a random action from Table 1 each step."""

    def __init__(self):
        self.sim = LOBSimulator()

    def run_episode(self, day: DayData) -> EpisodeResult:
        self.sim.reset(day)
        res = EpisodeResult()
        self._apply(np.random.randint(config.N_ACTIONS))

        while not self.sim.done:
            step: StepResult = self.sim.step()
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
                if abs(self.sim.inventory) >= config.MAX_INVENTORY:
                    self._apply(9)
                else:
                    self._apply(np.random.randint(config.N_ACTIONS))

        return res

    def _apply(self, action: int) -> None:
        if action == 9:
            self.sim.place_market_order()
        else:
            ta, tb = config.ACTIONS[action]
            self.sim.place_orders(ta, tb)
