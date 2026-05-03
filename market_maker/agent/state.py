"""
State feature extraction -- Section 4.3 of Spooner et al. (2018).

9-element state vector:
  [0] inventory / MAX_INVENTORY          (agent)
  [1] theta_a                            (agent)
  [2] theta_b                            (agent)
  [3] bid-ask spread s(t)                (market)
  [4] mid-price move dm(t)               (market)
  [5] queue imbalance                    (market)
  [6] signed volume of last trade        (market)
  [7] volatility (rolling std of dm)     (market)
  [8] RSI of mid-price                   (market)
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Deque

import numpy as np

from market_maker import config

if TYPE_CHECKING:
    from market_maker.data.simulator import EventView


class StateBuilder:

    def __init__(
        self,
        vol_window: int = config.VOLATILITY_WINDOW,
        rsi_window: int = config.RSI_WINDOW,
    ):
        self._dm_buf:  Deque[float] = deque(maxlen=vol_window)
        self._rsi_deltas: Deque[float] = deque(maxlen=rsi_window)
        self._dm_sum = 0.0
        self._dm_sumsq = 0.0
        self._rsi_gain_sum = 0.0
        self._rsi_loss_sum = 0.0
        self._last_rsi_mid: float | None = None
        self._state = np.empty(9, dtype=np.float64)

    def reset(self) -> None:
        self._dm_buf.clear()
        self._rsi_deltas.clear()
        self._dm_sum = 0.0
        self._dm_sumsq = 0.0
        self._rsi_gain_sum = 0.0
        self._rsi_loss_sum = 0.0
        self._last_rsi_mid = None

    def build(
        self,
        event:     "EventView",
        prev_mid:  float,
        inventory: int,
        theta_ask: float,
        theta_bid: float,
    ) -> np.ndarray:
        mid = event.mid_price
        dm  = mid - prev_mid
        self._append_dm(dm)
        self._append_rsi_mid(mid)

        state = self._state
        state[0] = inventory / config.MAX_INVENTORY
        state[1] = theta_ask   # Python int from config.ACTIONS; no float() needed
        state[2] = theta_bid
        state[3] = event.spread
        state[4] = dm

        # Inlined _queue_imbalance — avoids static-method dispatch (~1.3 µs saved)
        vb = event.bid_sz[0]
        va = event.ask_sz[0]
        t  = vb + va
        state[5] = (vb - va) / t if t > 0 else 0.0

        # Inlined _signed_volume — avoids static-method dispatch (~0.8 µs saved)
        if event.action in ('T', 'F'):
            sign = 1.0 if event.side == 'B' else -1.0 if event.side == 'A' else 0.0
            state[6] = sign * event.size / config.ORDER_SIZE
        else:
            state[6] = 0.0

        state[7] = self._volatility()
        state[8] = self._rsi()
        return state

    @staticmethod
    def _queue_imbalance(ev: "EventView") -> float:
        """Kept for external callers; build() inlines this for hot-path speed."""
        vb = float(ev.bid_sz[0])
        va = float(ev.ask_sz[0])
        total = vb + va
        return (vb - va) / total if total > 0 else 0.0

    @staticmethod
    def _signed_volume(ev: "EventView") -> float:
        """Kept for external callers; build() inlines this for hot-path speed."""
        if ev.action not in ('T', 'F'):
            return 0.0
        sign = 1.0 if ev.side == 'B' else -1.0 if ev.side == 'A' else 0.0
        return sign * ev.size / config.ORDER_SIZE

    def _volatility(self) -> float:
        n = len(self._dm_buf)
        if n < 2:
            return 0.0
        mean = self._dm_sum / n
        var  = self._dm_sumsq / n - mean * mean
        # math.sqrt is ~5x faster than np.sqrt for a scalar (no ufunc dispatch).
        return math.sqrt(var) if var > 0.0 else 0.0

    def _rsi(self) -> float:
        if not self._rsi_deltas:
            return 50.0
        if self._rsi_loss_sum == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + self._rsi_gain_sum / self._rsi_loss_sum)

    def _append_dm(self, dm: float) -> None:
        if len(self._dm_buf) == self._dm_buf.maxlen:
            old = self._dm_buf[0]
            self._dm_sum -= old
            self._dm_sumsq -= old * old
        self._dm_buf.append(dm)
        self._dm_sum += dm
        self._dm_sumsq += dm * dm

    def _append_rsi_mid(self, mid: float) -> None:
        if self._last_rsi_mid is None:
            self._last_rsi_mid = mid
            return
        delta = mid - self._last_rsi_mid
        self._last_rsi_mid = mid
        if len(self._rsi_deltas) == self._rsi_deltas.maxlen:
            old = self._rsi_deltas[0]
            if old > 0:
                self._rsi_gain_sum -= old
            else:
                self._rsi_loss_sum += old
        self._rsi_deltas.append(delta)
        if delta > 0:
            self._rsi_gain_sum += delta
        else:
            self._rsi_loss_sum -= delta
