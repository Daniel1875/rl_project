"""
Event-driven LOB replay simulator — Section 3 of Spooner et al. (2018).

Replays one DayData episode event-by-event.  The agent places a single
bid and ask limit order; execution and cancellation are handled according
to queue-position tracking and the uniform-cancellation assumption.

DayData is columnar (numpy arrays).  We extract per-event data into a
lightweight EventView object so downstream code (agent, state, benchmarks)
can use attribute access without touching the raw arrays.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

from market_maker import config
from market_maker.data.loader import DayData


# ---------------------------------------------------------------------------
# EventView — per-row snapshot extracted from columnar DayData
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EventView:
    """Lightweight snapshot of one DayData row at a given index."""
    action:    str
    side:      str
    price:     float
    size:      int
    bid_px:    np.ndarray   # shape (5,)
    bid_sz:    np.ndarray   # shape (5,)
    ask_px:    np.ndarray   # shape (5,)
    ask_sz:    np.ndarray   # shape (5,)
    mid_price: float
    spread:    float
    best_bid:  float
    best_ask:  float


def _view(day: DayData, i: int) -> EventView:
    """Build an EventView from DayData index i."""
    mid = day.mid(i)
    spd = day.spread(i)
    bb  = day.best_bid(i)
    ba  = day.best_ask(i)
    return EventView(
        action    = str(day.actions[i]),
        side      = str(day.sides[i]),
        price     = float(day.prices[i]),
        size      = int(day.sizes[i]),
        bid_px    = day.bid_px[i],      # view into (N,5) array row
        bid_sz    = day.bid_sz[i],
        ask_px    = day.ask_px[i],
        ask_sz    = day.ask_sz[i],
        mid_price = mid,
        spread    = spd,
        best_bid  = bb,
        best_ask  = ba,
    )


# ---------------------------------------------------------------------------
# Agent order bookkeeping
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AgentOrder:
    side:     str    # 'ask' or 'bid'
    price:    float  # quoted price in dollars
    size:     int    # remaining shares
    v_ahead:  float  # volume ahead in queue (same price level)
    v_behind: float  # volume behind
    active:   bool = True


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StepResult:
    event:          EventView
    matched_ask:    int    # shares executed on agent's ask
    matched_bid:    int    # shares executed on agent's bid
    mid_price:      float
    prev_mid_price: float
    spread_scale:   float  # Spread(t) = moving-avg half-spread
    done:           bool


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class LOBSimulator:
    """Replays one trading day as an RL episode."""

    def __init__(self):
        self._day:  Optional[DayData] = None
        self._idx:  int = 0
        self.ask_order: Optional[AgentOrder] = None
        self.bid_order: Optional[AgentOrder] = None
        self.inventory: int = 0
        self._mid_prev: float = 0.0
        self._spread_ma: Deque[float] = deque(maxlen=config.SPREAD_MA_WINDOW)
        self._spread_sum: float = 0.0
        self._event = EventView('', '', 0.0, 0, np.empty(5), np.empty(5),
                                np.empty(5), np.empty(5), 0.0, 0.0, 0.0, 0.0)
        self._step_result = StepResult(self._event, 0, 0, 0.0, 0.0, 0.0, False)
        # Per-episode pre-computed arrays (populated in reset()).
        # Eliminates per-event str()/float()/int() conversions and conditional
        # mid-price logic inside the hot _view_into() path.
        self._actions:     list       = []
        self._sides:       list       = []
        self._best_bids:   np.ndarray = np.empty(0)
        self._best_asks:   np.ndarray = np.empty(0)
        self._mid_prices:  np.ndarray = np.empty(0)
        self._spreads_arr: np.ndarray = np.empty(0)

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self, day: DayData) -> EventView:
        self._day = day
        self._idx = 0
        self.ask_order = None
        self.bid_order = None
        self.inventory = 0
        self._spread_ma.clear()
        self._spread_sum = 0.0

        # Pre-compute per-episode derived arrays once (vectorised).
        # Replaces str()/float()/int() calls and the conditional mid-price
        # branch that would otherwise run inside _view_into() ~100k times.
        self._actions = day.actions.tolist()   # numpy str array → Python list
        self._sides   = day.sides.tolist()     # same; list[i] needs no conversion
        bb = day.bid_px[:, 0]                  # (N,) 1-D view, no copy
        ba = day.ask_px[:, 0]
        self._best_bids   = bb
        self._best_asks   = ba
        both = (bb > 0) & (ba > 0)
        self._mid_prices  = np.where(both, (bb + ba) * 0.5,
                                     np.where(bb > 0, bb, ba))
        self._spreads_arr = np.maximum(0.0, ba - bb)

        ev = self._view_into(0)
        self._mid_prev = ev.mid_price
        if ev.spread > 0:
            self._append_spread_half(ev.spread / 2.0)
        return ev

    @property
    def done(self) -> bool:
        return self._day is None or self._idx >= len(self._day)

    @property
    def current_event(self) -> EventView:
        # Last visible book snapshot.  After step() processes event i, the
        # agent must re-quote from event i's book, not peek at row i+1.
        return self._event

    def spread_scale(self) -> float:
        return self._spread_sum / len(self._spread_ma) if self._spread_ma else 0.005

    # ------------------------------------------------------------------
    # Agent order placement
    # ------------------------------------------------------------------

    def place_orders(self, theta_ask: int, theta_bid: int) -> None:
        """Place/replace agent's limit orders (Eqs. 1–2).

        If the computed price snaps to the same level as the existing live
        order, we keep that order intact (preserving queue position and
        remaining size).  Only when the price actually changes do we cancel
        and re-join at the back of the new level.
        """
        ev  = self.current_event
        sc  = self.spread_scale()
        mid = ev.mid_price

        new_ask = self._make_order('ask', mid + theta_ask * sc, ev)
        new_bid = self._make_order('bid', mid - theta_bid * sc, ev)

        # Preserve queue position when price is unchanged
        if (self.ask_order and self.ask_order.active and
                abs(self.ask_order.price - new_ask.price) < 1e-6):
            new_ask.v_ahead  = self.ask_order.v_ahead
            new_ask.v_behind = self.ask_order.v_behind
            new_ask.size     = self.ask_order.size

        if (self.bid_order and self.bid_order.active and
                abs(self.bid_order.price - new_bid.price) < 1e-6):
            new_bid.v_ahead  = self.bid_order.v_ahead
            new_bid.v_behind = self.bid_order.v_behind
            new_bid.size     = self.bid_order.size

        self.ask_order = new_ask
        self.bid_order = new_bid

    def place_market_order(self) -> None:
        """Action 9: clear inventory with a market order.

        Paper Table 1: Size_m = -Inv(t_i)  (alpha = 1, full clearing).
        We assume the market order fills immediately and completely —
        the agent's order size is small relative to total market volume
        so slippage is negligible (same assumption as the paper).
        """
        self.ask_order = None
        self.bid_order = None
        self.inventory = 0

    # ------------------------------------------------------------------
    # Advance one event
    # ------------------------------------------------------------------

    def step(self) -> StepResult:
        assert not self.done
        ev = self._view_into(self._idx)
        self._idx += 1

        mid_now  = ev.mid_price
        mid_prev = self._mid_prev
        s = ev.spread
        if s > 0:
            self._append_spread_half(s / 2.0)

        matched_ask = matched_bid = 0

        if ev.action == 'A':
            self._process_add(ev)
        elif ev.action in ('T', 'F'):
            matched_ask, matched_bid = self._process_trade(ev)
        elif ev.action in ('C', 'D'):
            self._process_cancel(ev)

        # Update inventory
        self.inventory += matched_bid
        self.inventory -= matched_ask
        self.inventory = max(config.MIN_INVENTORY,
                             min(config.MAX_INVENTORY, self.inventory))
        self._mid_prev = mid_now

        res = self._step_result
        res.event = ev
        res.matched_ask = matched_ask
        res.matched_bid = matched_bid
        res.mid_price = mid_now
        res.prev_mid_price = mid_prev
        res.spread_scale = self.spread_scale()
        res.done = self.done
        return res

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_order(self, side: str, target_price: float,
                    ev: EventView) -> AgentOrder:
        """
        Snap target_price to the nearest existing LOB level and
        record the volume ahead (agent joins back of queue).
        """
        if side == 'ask':
            prices = ev.ask_px
            sizes  = ev.ask_sz
        else:
            prices = ev.bid_px
            sizes  = ev.bid_sz

        valid = np.where(sizes > 0)[0]
        if len(valid) == 0:
            level = 0
        else:
            level = valid[np.argmin(np.abs(prices[valid] - target_price))]

        snapped_price = float(prices[level]) if prices[level] > 0 else target_price
        v_ahead = float(sizes[level])

        return AgentOrder(
            side     = side,
            price    = snapped_price,
            size     = config.ORDER_SIZE,
            v_ahead  = v_ahead,
            v_behind = 0.0,
        )

    def _process_trade(self, ev: EventView) -> Tuple[int, int]:
        """
        A trade at ev.price × ev.size.  If the agent has an order at
        the same price level, consume v_ahead first, then execute.
        """
        matched_ask = matched_bid = 0
        for order, out_attr in ((self.ask_order, 'ask'),
                                (self.bid_order, 'bid')):
            if order is None or not order.active:
                continue
            if abs(order.price - ev.price) > 1e-6:
                continue
            if not self._trade_hits_order(ev, order):
                continue
            remaining = ev.size
            if order.v_ahead > 0:
                consumed = min(float(remaining), order.v_ahead)
                order.v_ahead -= consumed
                remaining = max(0.0, float(remaining) - consumed)
            if remaining > 0 and order.v_ahead <= 0:
                vol = int(min(remaining, order.size))
                order.size -= vol
                if out_attr == 'ask':
                    matched_ask += vol
                else:
                    matched_bid += vol
                if order.size <= 0:
                    order.active = False
        return matched_ask, matched_bid

    def _process_add(self, ev: EventView) -> None:
        """Track newly added visible volume behind the agent at its price."""
        for order in (self.ask_order, self.bid_order):
            if order is None or not order.active:
                continue
            if abs(order.price - ev.price) > 1e-6:
                continue
            if not self._book_event_matches_order(ev, order):
                continue
            order.v_behind += ev.size

    def _process_cancel(self, ev: EventView) -> None:
        """
        Uniform-cancellation assumption (Section 3):
        proportion of cancellation volume that was ahead of the agent =
        v_ahead / (v_ahead + order.size + v_behind).
        """
        for order in (self.ask_order, self.bid_order):
            if order is None or not order.active:
                continue
            if abs(order.price - ev.price) > 1e-6:
                continue
            if not self._book_event_matches_order(ev, order):
                continue
            total = order.v_ahead + order.size + order.v_behind
            if total <= 0:
                continue
            p_ahead = order.v_ahead / total
            order.v_ahead  = max(0.0, order.v_ahead  - ev.size * p_ahead)
            order.v_behind = max(0.0, order.v_behind - ev.size * (1 - p_ahead))

    @staticmethod
    def _book_event_matches_order(ev: EventView, order: AgentOrder) -> bool:
        if ev.side == 'A':
            return order.side == 'ask'
        if ev.side == 'B':
            return order.side == 'bid'
        return False

    @staticmethod
    def _trade_hits_order(ev: EventView, order: AgentOrder) -> bool:
        if ev.side == 'B':
            return order.side == 'ask'
        if ev.side == 'A':
            return order.side == 'bid'
        return False

    def _append_spread_half(self, half_spread: float) -> None:
        if len(self._spread_ma) == self._spread_ma.maxlen:
            self._spread_sum -= self._spread_ma[0]
        self._spread_ma.append(half_spread)
        self._spread_sum += half_spread

    def _view_into(self, i: int) -> EventView:
        # Hot path: ~100k calls per episode.  All per-event conversions (str,
        # float, int, conditional mid) were moved to reset() so this is now
        # pure array lookups — no type conversion on the critical path.
        day = self._day
        ev  = self._event
        ev.action    = self._actions[i]       # plain Python str, no str() call
        ev.side      = self._sides[i]         # plain Python str, no str() call
        ev.price     = day.prices[i]          # numpy float64 scalar
        ev.size      = day.sizes[i]           # numpy int64 scalar
        ev.bid_px    = day.bid_px[i]
        ev.bid_sz    = day.bid_sz[i]
        ev.ask_px    = day.ask_px[i]
        ev.ask_sz    = day.ask_sz[i]
        ev.mid_price = self._mid_prices[i]    # pre-computed, no branch
        ev.spread    = self._spreads_arr[i]   # pre-computed, no max()/float()
        ev.best_bid  = self._best_bids[i]     # 1-D array lookup
        ev.best_ask  = self._best_asks[i]     # 1-D array lookup
        return ev
