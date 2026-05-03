"""
Market making agent — Section 4 of Spooner et al. (2018).

Consolidated agent: SARSA(λ) + LCTC + asymmetric-dampened reward.

Episode-loop optimisations
--------------------------
* q_values_and_indices(state) tiles each state exactly ONCE.
* The returned tile indices (ai, mi, fi) are passed directly to update()
  as `cur_indices`, eliminating the duplicate active_tiles() call that
  the old update(state, …, next_state, …) performed internally.
* q_next_val = qs_next[next_action] is computed from the same forward pass
  used for action selection, so update() never calls q_values() a second time.
* Together these reduce active_tiles() calls from 9/step → 3/step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from market_maker import config
from market_maker.data.loader import DayData
from market_maker.data.simulator import LOBSimulator, StepResult
from market_maker.agent.state import StateBuilder
from market_maker.agent.reward import (
    RewardComponents, compute, AsymmetricDampedReward,
)
from market_maker.agent.sarsa import SarsaLambda


# ---------------------------------------------------------------------------
# Episode statistics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EpisodeResult:
    total_pnl:      float = 0.0
    total_reward:   float = 0.0
    n_steps:        int   = 0
    inv_path:       List[int]   = field(default_factory=list)
    spread_history: List[float] = field(default_factory=list)

    @property
    def mean_abs_position(self) -> float:
        return float(np.mean(np.abs(self.inv_path))) if self.inv_path else 0.0

    def normalised_daily_pnl(self) -> float:
        """Total PnL divided by mean market spread (Section 5)."""
        ms = float(np.mean(self.spread_history)) if self.spread_history else 0.0
        return self.total_pnl / ms if ms > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _legal_mask(inventory: int) -> np.ndarray:
    mask = np.ones(config.N_ACTIONS, dtype=bool)
    if abs(inventory) >= config.MAX_INVENTORY:
        mask[:9] = False   # force market-order clear (action 9)
    return mask


def _thetas(action: int):
    return (0, 0) if action == 9 else config.ACTIONS[action]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MarketMakingAgent:

    def __init__(
        self,
        reward_fn: Optional[Callable[[RewardComponents], float]] = None,
        sarsa_kwargs: Optional[dict] = None,
    ):
        self.reward_fn  = reward_fn or AsymmetricDampedReward(config.DAMPING_ETA)
        self.sim        = LOBSimulator()
        self.state_bld  = StateBuilder()
        self.learner    = SarsaLambda(**(sarsa_kwargs or {}))

    def run_episode(self, day: DayData, learn: bool = True) -> EpisodeResult:
        if learn:
            self.learner.new_episode()
        else:
            self.learner.reset_traces()
        self.state_bld.reset()

        ev0  = self.sim.reset(day)
        res  = EpisodeResult()

        # ── Initial state + action ───────────────────────────────────────
        theta_a, theta_b = _thetas(0)
        state  = self.state_bld.build(ev0, ev0.mid_price, 0, theta_a, theta_b)
        mask   = _legal_mask(0)

        # Tile state once; keep Q-values on device and copy back only the
        # selected action/value.
        action, q_cur_val, cur_idx = self.learner.select_action_and_indices(
            state, mask, explore=learn
        )
        self._apply(action)

        # ── Main event loop ──────────────────────────────────────────────
        while not self.sim.done:
            step: StepResult = self.sim.step()

            p_ask = self.sim.ask_order.price if self.sim.ask_order else step.event.best_ask
            p_bid = self.sim.bid_order.price if self.sim.bid_order else step.event.best_bid

            rc = compute(
                matched_ask=step.matched_ask,
                matched_bid=step.matched_bid,
                p_ask=p_ask,
                p_bid=p_bid,
                mid=step.mid_price,
                prev_mid=step.prev_mid_price,
                inventory=self.sim.inventory,
            )
            reward = self.reward_fn(rc)

            ta, tb = _thetas(action)
            next_state = self.state_bld.build(
                step.event, step.prev_mid_price,
                self.sim.inventory, ta, tb,
            )
            next_mask = _legal_mask(self.sim.inventory)

            if step.done:
                # Terminal step: q_next = 0, no next action needed
                if learn:
                    self.learner.update(cur_idx, action, reward, q_cur_val, 0.0, done=True)
                next_action = 0       # placeholder; episode ends
                next_q_val  = 0.0
                next_idx    = cur_idx # placeholder
            else:
                # Tile next_state once; keep Q-values on device and copy back
                # only the selected action/value.
                next_action, next_q_val, next_idx = self.learner.select_action_and_indices(
                    next_state, next_mask, explore=learn
                )
                if learn:
                    self.learner.update(
                        cur_idx, action, reward,
                        q_cur_val, next_q_val, done=False,
                    )
                self._apply(next_action)

            res.total_pnl    += rc.pnl
            res.total_reward += reward
            res.n_steps      += 1
            res.inv_path.append(self.sim.inventory)
            res.spread_history.append(step.event.spread)

            # Carry forward — cur_idx is next_idx; no retiling on next iteration
            state      = next_state
            cur_idx    = next_idx
            action     = next_action
            q_cur_val  = next_q_val
        return res

    def _apply(self, action: int) -> None:
        if action == 9:
            self.sim.place_market_order()
        else:
            ta, tb = config.ACTIONS[action]
            self.sim.place_orders(ta, tb)
