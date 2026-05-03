"""
Hash-based tile coding and Linear Combination of Tile Codings (LCTC).
Section 4.3, Equation 7 of Spooner et al. (2018).

Uses a fixed-size hash table to avoid exponential memory blowup.
active_tiles() is fully vectorised: no Python loops over tilings or dims.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from market_maker import config


class TileCoding:
    """
    Hash-based tile coding with a fixed memory vector.

    Each tiling is staggered in coordinate space before binning.  The resulting
    active tile coordinates are then hashed into the fixed memory vector.
    """

    def __init__(
        self,
        n_tilings: int,
        ranges: List[Tuple[float, float]],
        n_tiles: int = config.N_TILES,
        memory_size: int = config.MEMORY_SIZE,
    ):
        self.n_tilings = n_tilings
        self.n_dims = len(ranges)
        self.n_tiles = n_tiles
        self.memory_size = memory_size

        lo = np.array([r[0] for r in ranges], dtype=np.float64)
        hi = np.array([r[1] for r in ranges], dtype=np.float64)
        self._lo = lo
        self._span = np.where(hi > lo, hi - lo, 1.0)

        tiling_offsets = np.arange(n_tilings, dtype=np.float64)[:, None]
        dim_offsets = np.arange(1, self.n_dims + 1, dtype=np.float64)[None, :]
        self._coord_offsets = (tiling_offsets * dim_offsets / n_tilings) % 1.0

        all_primes = np.array([
            2654435761, 805459861, 2246822519, 1640531527,
            3266489917, 2654435789, 805459891, 1234567891,
            987654329, 1111111121,
        ], dtype=np.uint64)
        self._primes = all_primes[:self.n_dims]
        self._tiling_base = (
            np.arange(n_tilings, dtype=np.uint64) * np.uint64(2654435761)
        )
        self._max_tile = np.uint64(self.n_tiles - 1)
        self._memory_size_u = np.uint64(self.memory_size)
        self._scratch_xn = np.empty(self.n_dims, dtype=np.float64)
        self._scratch_shifted = np.empty((n_tilings, self.n_dims), dtype=np.float64)
        self._scratch_coords = np.empty((n_tilings, self.n_dims), dtype=np.uint64)
        self._scratch_hash = np.empty(n_tilings, dtype=np.uint64)

        self.weights = np.zeros(memory_size, dtype=np.float64)

    def active_tiles(self, x: np.ndarray) -> np.ndarray:
        """Return active tile indices as an int64 array of shape (n_tilings,)."""
        out = np.empty(self.n_tilings, dtype=np.int64)
        self.active_tiles_into(x, out)
        return out

    def active_tiles_into(self, x: np.ndarray, out: np.ndarray) -> None:
        """Write active tile indices into a caller-provided int64 array."""
        np.subtract(x, self._lo, out=self._scratch_xn)
        np.divide(self._scratch_xn, self._span, out=self._scratch_xn)
        np.clip(self._scratch_xn, 0.0, 1.0, out=self._scratch_xn)
        self._scratch_xn *= self.n_tiles
        self.active_tiles_from_xn(self._scratch_xn, out)

    def normalize_into(self, x: np.ndarray, out: np.ndarray) -> None:
        """Normalize x → [0, n_tiles] range, storing result in `out`.

        Called once for the full 9-D state; agent/market sub-views of `out`
        are then passed to active_tiles_from_xn() to skip the duplicate
        normalize step for the 3-D and 6-D codings.
        """
        np.subtract(x, self._lo, out=out)
        np.divide(out, self._span, out=out)
        np.clip(out, 0.0, 1.0, out=out)
        out *= self.n_tiles

    def active_tiles_from_xn(self, xn: np.ndarray, out: np.ndarray) -> None:
        """Compute tile indices from a pre-normalized xn (already in [0, n_tiles]).

        Skips the 4-op normalize step.  Use after calling normalize_into() on
        the parent coding — LCTC.active_indices_into() exploits this to share
        one normalization pass across all three codings (~37 µs saved per event).
        """
        np.add(xn[None, :], self._coord_offsets, out=self._scratch_shifted)
        np.copyto(self._scratch_coords, self._scratch_shifted, casting="unsafe")
        np.minimum(self._scratch_coords, self._max_tile, out=self._scratch_coords)
        np.multiply(self._scratch_coords, self._primes, out=self._scratch_coords)
        np.sum(self._scratch_coords, axis=1, dtype=np.uint64, out=self._scratch_hash)
        np.add(self._scratch_hash, self._tiling_base, out=self._scratch_hash)
        np.remainder(self._scratch_hash, self._memory_size_u, out=self._scratch_hash)
        np.copyto(out, self._scratch_hash, casting="unsafe")

    def value(self, x: np.ndarray) -> float:
        return float(self.weights[self.active_tiles(x)].sum())


class LCTC:
    """
    Linear Combination of Tile Codings (Eq. 7).

    Three TileCoding objects are used for agent-state, market-state, and full
    state with fixed mixing weights lambda_i = (0.6, 0.1, 0.3).
    """

    AGENT_RANGES: List[Tuple[float, float]] = [
        (-1.0, 1.0),
        (0.0, 6.0),
        (0.0, 6.0),
    ]
    MARKET_RANGES: List[Tuple[float, float]] = [
        (0.0, 2.0),
        (-2.0, 2.0),
        (-1.0, 1.0),
        (-5.0, 5.0),
        (0.0, 1.0),
        (0.0, 100.0),
    ]
    FULL_RANGES = AGENT_RANGES + MARKET_RANGES

    def __init__(
        self,
        n_tilings: int = config.N_TILINGS,
        n_tiles: int = config.N_TILES,
        memory_size: int = config.MEMORY_SIZE,
        lctc_weights=config.LCTC_WEIGHTS,
    ):
        self.lambdas = np.array(lctc_weights, dtype=np.float64)
        assert abs(self.lambdas.sum() - 1.0) < 1e-6, "LCTC weights must sum to 1"

        self.tc_agent = TileCoding(n_tilings, self.AGENT_RANGES, n_tiles, memory_size)
        self.tc_market = TileCoding(n_tilings, self.MARKET_RANGES, n_tiles, memory_size)
        self.tc_full = TileCoding(n_tilings, self.FULL_RANGES, n_tiles, memory_size)
        self._tcs = [self.tc_agent, self.tc_market, self.tc_full]
        # Shared 9-D scratch for active_indices_into() — normalized once per event.
        self._scratch_xn = np.empty(len(self.FULL_RANGES), dtype=np.float64)

    def value(self, state: np.ndarray) -> float:
        return (
            self.lambdas[0] * self.tc_agent.value(state[:3])
            + self.lambdas[1] * self.tc_market.value(state[3:])
            + self.lambdas[2] * self.tc_full.value(state)
        )

    def active_indices(self, state: np.ndarray):
        """Return (agent_idx, market_idx, full_idx), each shape (n_tilings,)."""
        return (
            self.tc_agent.active_tiles(state[:3]),
            self.tc_market.active_tiles(state[3:]),
            self.tc_full.active_tiles(state),
        )

    def active_indices_into(
        self,
        state: np.ndarray,
        out_a: np.ndarray,
        out_m: np.ndarray,
        out_f: np.ndarray,
    ) -> None:
        """Write (agent, market, full) tile indices, sharing one normalization pass.

        FULL_RANGES = AGENT_RANGES + MARKET_RANGES, so the full-state
        normalization yields identical values for dims [:3] (agent) and [3:]
        (market).  Normalizing once and reusing sub-views avoids ~37 µs of
        redundant NumPy ops per event (~62 min saved over 1000 episodes).
        """
        self.tc_full.normalize_into(state, self._scratch_xn)          # 4 ops, done once
        self.tc_agent.active_tiles_from_xn(self._scratch_xn[:3], out_a)  # 8 ops
        self.tc_market.active_tiles_from_xn(self._scratch_xn[3:], out_m) # 8 ops
        self.tc_full.active_tiles_from_xn(self._scratch_xn, out_f)       # 8 ops

    def update_weights(self, delta: float, alpha: float, traces: List[np.ndarray]) -> None:
        """Apply w += alpha * delta * e for each coding."""
        coeff = alpha * delta
        for tc, e in zip(self._tcs, traces):
            tc.weights += coeff * e
