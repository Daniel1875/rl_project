"""
SARSA(lambda) with eligibility traces -- Section 4.4 of Spooner et al. (2018).

One set of weights per action approximates Q(s, a).  All N_ACTIONS weight
vectors share the same TD-error at each step; traces are maintained per
action per LCTC coding.

GPU design
----------
All weight and trace arrays are stored as two contiguous PyTorch tensors of
shape (n_actions, 3, memory_size) in bfloat16 on CUDA or float32 on CPU.
This enables:

  Trace decay:    traces.mul_(gam_lam)                  -- 1 GPU kernel
  Weight update:  weights.add_(traces, alpha=coeff)     -- 1 GPU kernel (fused)
  Scatter set:    traces[a, k, idx] = lw[k]             -- O(n_tilings) scatter

  Q-value batch:  for all 10 actions simultaneously with 3 gather ops rather
                  than 30 separate active_tiles calls.

active_tiles() stays on CPU/NumPy (cheap: 32-element arrays); the resulting
indices are transferred to the device once per step.

Performance notes
-----------------
* Use q_values_and_indices() in the episode loop (instead of separate
  q_values() + action selection) so each state is tiled exactly ONCE.
* Pass the returned (ai, mi, fi) tuple directly to update() as `cur_indices`
  to avoid recomputing the same tile hashes inside the update.
* Together, these cut active_tiles() calls from 9/step → 3/step (3×).
* epsilon is updated once per episode (in new_episode()) rather than per
  LOB event -- matching Table 2 where ε_T = 1000 *episodes*.

Epsilon decay fix
-----------------
The paper (Table 2) sets ε_T = 1000 alongside "training episodes = 1000".
The correct interpretation is that ε decays linearly over 1000 *episodes*,
not over 1000 individual LOB events (a single day has 50k-300k events).
Updating ε per-event caused ε to hit the floor in <1% of episode 0, leaving
the agent essentially greedy for the entire training run.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from market_maker import config
from market_maker.agent.tile_coding import TileCoding, LCTC


def _get_device() -> "torch.device":
    """
    Pick the best available compute device.

    GPU (CUDA) is strongly preferred: the stacked (n_actions, 3, memory_size)
    bfloat16 tensor is 12 MB at MEMORY_SIZE=200_000 (vs 24 MB float32) and
    all weight/trace ops are single GPU kernels with half the memory bandwidth.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required.  Install with:  pip install torch"
        )
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        gb = props.total_memory / 1024**3
        print(f"[sarsa] CUDA GPU: {props.name}  ({gb:.1f} GB VRAM)  [bfloat16]")
    else:
        dev = torch.device("cpu")
        print(
            "[sarsa] No CUDA GPU found -- running on CPU.\n"
            "        For full training speed, use a CUDA-enabled GPU.\n"
            "        On CPU, set config.MEMORY_SIZE = 10_000 to keep\n"
            "        each episode under ~2 minutes."
        )
    return dev


class SarsaLambda:
    CHECKPOINT_VERSION = 2

    """
    SARSA(lambda) value function learner backed by PyTorch tensors.

    Fast path (episode loop):
      1. Call q_values_and_indices(state) -> (qs, idx) at each step.
      2. Select action via _select_from_qs(qs, mask, explore).
      3. Call update(cur_idx, action, reward, q_next_val, done)
         using the idx from STEP 1 of the PREVIOUS iteration and
         q_next_val = qs_next[next_action] from the current iteration.

    This ensures each state is tiled exactly once per step (3 active_tiles
    calls vs. 9 in the naive implementation).
    """

    def __init__(
        self,
        n_actions:     int   = config.N_ACTIONS,
        n_tilings:     int   = config.N_TILINGS,
        n_tiles:       int   = config.N_TILES,
        memory_size:   int   = config.MEMORY_SIZE,
        lctc_weights         = config.LCTC_WEIGHTS,
        alpha:         float = config.LEARNING_RATE,
        gamma:         float = config.DISCOUNT,
        trace_decay:   float = config.TRACE_PARAM,
        epsilon_start: float = config.EPSILON_START,
        epsilon_floor: float = config.EPSILON_FLOOR,
        epsilon_T:     int   = config.EPSILON_T,
    ):
        self.n_actions     = n_actions
        self.memory_size   = memory_size
        self.alpha         = alpha
        self.gamma         = gamma
        self.trace_decay   = trace_decay
        self.epsilon_floor = epsilon_floor
        self.epsilon_T     = epsilon_T
        self.epsilon       = epsilon_start  # current exploration rate

        # Episode counter -- epsilon is updated once per episode (not per event)
        self._episode = 0

        self._device = _get_device()
        # bfloat16 on CUDA: halves weight/trace tensor size (12 MB vs 24 MB at
        # MEMORY_SIZE=200_000) and halves memory-bandwidth cost of trace-decay
        # and weight-update kernels.  float32 on CPU (bf16 is emulated, slower).
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32

        # LCTC mixing weights as a device tensor: shape (3,)
        lw = torch.tensor(lctc_weights, dtype=self._dtype, device=self._device)
        self._lw = lw

        # Decay constant (γλ) as a device scalar
        self._gam_lam = torch.tensor(
            gamma * trace_decay, dtype=self._dtype, device=self._device
        )
        self._gamma_t = torch.tensor(gamma, dtype=self._dtype, device=self._device)
        self._alpha_t = torch.tensor(alpha, dtype=self._dtype, device=self._device)

        # Shared tile-coding objects (CPU) -- used only for active_tiles()
        self._tc_agent  = TileCoding(n_tilings, LCTC.AGENT_RANGES,  n_tiles, memory_size)
        self._tc_market = TileCoding(n_tilings, LCTC.MARKET_RANGES, n_tiles, memory_size)
        self._tc_full   = TileCoding(n_tilings, LCTC.FULL_RANGES,   n_tiles, memory_size)
        # Shared 9-D normalization scratch — normalize_into() writes here once per
        # event; _active() then passes [:3] / [3:] views to skip re-normalizing.
        self._xn_full   = np.empty(len(LCTC.FULL_RANGES), dtype=np.float64)
        self._active_slot = 0

        # Stacked weight and trace tensors: (n_actions, 3, memory_size)
        # bfloat16 on CUDA / float32 on CPU. Coding axis: [0=agent, 1=market, 2=full]
        self._weights = torch.zeros(
            n_actions, 3, memory_size, dtype=self._dtype, device=self._device
        )
        self._traces = torch.zeros_like(self._weights)
        self._idx_slots = [
            (
                torch.empty(n_tilings, dtype=torch.long, device=self._device),
                torch.empty(n_tilings, dtype=torch.long, device=self._device),
                torch.empty(n_tilings, dtype=torch.long, device=self._device),
            )
            for _ in range(2)
        ]
        self._cpu_idx_slots = [
            (
                np.empty(n_tilings, dtype=np.int64),
                np.empty(n_tilings, dtype=np.int64),
                np.empty(n_tilings, dtype=np.int64),
            )
            for _ in range(2)
        ]
        self._cpu_idx_tensors = [
            tuple(torch.from_numpy(arr) for arr in slot)
            for slot in self._cpu_idx_slots
        ]
        self._mask_tensor = torch.empty(n_actions, dtype=torch.bool, device=self._device)
        self._use_cuda_graphs = (
            self._device.type == "cuda"
            and hasattr(torch.cuda, "CUDAGraph")
            and hasattr(torch.cuda, "graph")
        )
        self._cuda_graphs = [None] * n_actions
        self._q_cuda_graph = None
        if self._use_cuda_graphs:
            self._q_graph_ai = torch.empty(n_tilings, dtype=torch.long, device=self._device)
            self._q_graph_mi = torch.empty(n_tilings, dtype=torch.long, device=self._device)
            self._q_graph_fi = torch.empty(n_tilings, dtype=torch.long, device=self._device)
            self._q_graph_out = torch.empty(n_actions, dtype=self._dtype, device=self._device)
            self._graph_ai = torch.empty(n_tilings, dtype=torch.long, device=self._device)
            self._graph_mi = torch.empty(n_tilings, dtype=torch.long, device=self._device)
            self._graph_fi = torch.empty(n_tilings, dtype=torch.long, device=self._device)
            self._graph_reward = torch.zeros((), dtype=self._dtype, device=self._device)
            self._graph_q_cur = torch.zeros((), dtype=self._dtype, device=self._device)
            self._graph_q_next = torch.zeros((), dtype=self._dtype, device=self._device)
            self._graph_scaled_traces = torch.empty_like(self._weights)

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def new_episode(self) -> None:
        """Reset traces and advance epsilon by one episode."""
        self._traces.zero_()
        self._episode += 1
        frac = min(self._episode / max(self.epsilon_T, 1), 1.0)
        self.epsilon = (self.epsilon_floor
                        + (config.EPSILON_START - self.epsilon_floor) * (1.0 - frac))

    def reset_traces(self) -> None:
        """Reset eligibility traces without changing epsilon or episode count."""
        self._traces.zero_()

    # ------------------------------------------------------------------
    # Internal: compute active tile indices for all 3 codings
    # ------------------------------------------------------------------

    def _active(self, state: np.ndarray) -> Tuple:
        """Return reusable device index tensors for the three active codings.

        Normalizes the full 9-D state once into self._xn_full, then passes
        sub-views [:3] and [3:] to the agent and market codings, skipping
        their individual normalize steps (~37 µs saved per call).
        """
        slot = self._active_slot
        ai, mi, fi = self._idx_slots[self._active_slot]
        self._active_slot = 1 - self._active_slot
        cai, cmi, cfi = self._cpu_idx_slots[slot]
        tai, tmi, tfi = self._cpu_idx_tensors[slot]
        self._tc_full.normalize_into(state, self._xn_full)
        self._tc_agent.active_tiles_from_xn(self._xn_full[:3], cai)
        self._tc_market.active_tiles_from_xn(self._xn_full[3:], cmi)
        self._tc_full.active_tiles_from_xn(self._xn_full, cfi)
        ai.copy_(tai, non_blocking=True)
        mi.copy_(tmi, non_blocking=True)
        fi.copy_(tfi, non_blocking=True)
        return ai, mi, fi

    # ------------------------------------------------------------------
    # Q-values + indices (primary API for the episode loop)
    # ------------------------------------------------------------------

    def q_values_and_indices(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, Tuple]:
        """
        Compute Q(s, a) for all actions AND return the tile indices.

        Returns
        -------
        qs : ndarray shape (n_actions,) float64
        idx : (ai, mi, fi)  -- device LongTensors, each shape (n_tilings,)

        Pass idx directly to update() as cur_indices to avoid retiling.
        """
        qs_t, (ai, mi, fi) = self.q_values_tensor_and_indices(state)
        qs = qs_t.float().cpu().numpy().astype(np.float64)  # bf16 has no numpy backend
        return qs, (ai, mi, fi)

    def q_values_tensor_and_indices(self, state: np.ndarray):
        """Compute Q(s, a) on device and return the device tensor plus indices."""
        ai, mi, fi = self._active(state)
        if self._use_cuda_graphs:
            self._q_graph_ai.copy_(ai, non_blocking=True)
            self._q_graph_mi.copy_(mi, non_blocking=True)
            self._q_graph_fi.copy_(fi, non_blocking=True)
            if self._q_cuda_graph is None:
                self._q_cuda_graph = self._capture_q_graph()
            self._q_cuda_graph.replay()
            return self._q_graph_out, (ai, mi, fi)
        va = self._weights[:, 0][:, ai].sum(dim=1)
        vm = self._weights[:, 1][:, mi].sum(dim=1)
        vf = self._weights[:, 2][:, fi].sum(dim=1)
        qs = self._lw[0] * va + self._lw[1] * vm + self._lw[2] * vf
        return qs, (ai, mi, fi)

    def _capture_q_graph(self):
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._q_graph_body()
        return graph

    def _q_graph_body(self) -> None:
        va = self._weights[:, 0][:, self._q_graph_ai].sum(dim=1)
        vm = self._weights[:, 1][:, self._q_graph_mi].sum(dim=1)
        vf = self._weights[:, 2][:, self._q_graph_fi].sum(dim=1)
        self._q_graph_out.copy_(self._lw[0] * va + self._lw[1] * vm + self._lw[2] * vf)

    def select_action_and_indices(
        self,
        state: np.ndarray,
        mask: Optional[np.ndarray] = None,
        explore: bool = True,
    ):
        """
        Select an action while keeping Q-values on device.

        Returns (action, q_action, idx), copying only the selected action and
        selected Q scalar back to Python instead of the full Q-vector.
        """
        qs, idx = self.q_values_tensor_and_indices(state)
        if explore and np.random.random() < self.epsilon:
            legal = np.where(mask)[0] if mask is not None else np.arange(self.n_actions)
            action = int(np.random.choice(legal))
            return action, float(qs[action].item()), idx

        if mask is not None:
            self._mask_tensor.copy_(torch.from_numpy(mask), non_blocking=True)
            qs_select = qs.masked_fill(~self._mask_tensor, -torch.inf)
        else:
            qs_select = qs

        action_t = torch.argmax(qs_select)
        action = int(action_t.item())
        return action, float(qs[action].item()), idx

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Convenience wrapper that discards tile indices."""
        qs, _ = self.q_values_and_indices(state)
        return qs

    # ------------------------------------------------------------------
    # Action selection (operate on pre-computed Q-values)
    # ------------------------------------------------------------------

    def _select_from_qs(
        self,
        qs:      np.ndarray,
        mask:    Optional[np.ndarray],
        explore: bool,
    ) -> int:
        """
        Epsilon-greedy (explore=True) or greedy (explore=False) action
        selection from pre-computed Q-values.
        """
        if explore and np.random.random() < self.epsilon:
            legal = np.where(mask)[0] if mask is not None else np.arange(self.n_actions)
            return int(np.random.choice(legal))
        if mask is not None:
            qs = np.where(mask, qs, -np.inf)
        return int(np.argmax(qs))

    def greedy_action(self, state: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> int:
        """Convenience: greedy action directly from state (retiles)."""
        qs, _ = self.q_values_and_indices(state)
        return self._select_from_qs(qs, mask, explore=False)

    def epsilon_greedy(self, state: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> int:
        """Convenience: epsilon-greedy action directly from state (retiles)."""
        qs, _ = self.q_values_and_indices(state)
        return self._select_from_qs(qs, mask, explore=True)

    # ------------------------------------------------------------------
    # SARSA(lambda) update -- Sutton & Barto Ch. 12
    # ------------------------------------------------------------------

    def update(
        self,
        cur_indices,       # (ai_s, mi_s, fi_s) from q_values_and_indices(state)
        action:      int,
        reward:      float,
        q_cur_val:   float,  # Q(s, a) from the earlier q_values_and_indices call
        q_next_val:  float,  # Q(s', a') already computed; 0.0 if terminal
        done:        bool,
    ) -> float:
        """
        SARSA(λ) weight update.

        Parameters
        ----------
        cur_indices : tuple of device LongTensors
            Tile indices for the *current* state (from q_values_and_indices).
            Passing these in avoids retiling the same state a second time.
        action : int
            Action taken in the current state.
        reward : float
            Observed reward r(t).
        q_cur_val : float
            Q(s, a) from the forward pass used for action selection.
        q_next_val : float
            Q(s', a') already computed by the caller (0.0 for terminal steps).
            This eliminates the redundant q_values(next_state) call that the
            old update() performed internally.
        done : bool
            True if this is the last step of the episode.
        """
        ai_s, mi_s, fi_s = cur_indices

        # TD-error (pure Python float -- no scalar GPU tensor needed)
        delta = reward + self.gamma * q_next_val - q_cur_val

        if self._use_cuda_graphs:
            self._update_graph_inputs(ai_s, mi_s, fi_s, reward, q_cur_val, q_next_val)
            graph = self._cuda_graphs[action]
            if graph is None:
                graph = self._capture_update_graph(action)
                self._cuda_graphs[action] = graph
            graph.replay()
            if done:
                self._traces.zero_()
            return delta

        # 1. Decay ALL traces in a single GPU kernel
        self._traces.mul_(self._gam_lam)

        # 2. Replacing traces: set active tiles for (state, action) to lambda_i
        self._traces[action, 0, ai_s] = self._lw[0]
        self._traces[action, 1, mi_s] = self._lw[1]
        self._traces[action, 2, fi_s] = self._lw[2]

        # 3. Update ALL weight vectors in a single fused GPU kernel
        self._weights.add_(self._traces, alpha=float(self.alpha * delta))

        if done:
            self._traces.zero_()

        return delta

    def _update_graph_inputs(
        self,
        ai_s,
        mi_s,
        fi_s,
        reward: float,
        q_cur_val: float,
        q_next_val: float,
    ) -> None:
        self._graph_ai.copy_(ai_s, non_blocking=True)
        self._graph_mi.copy_(mi_s, non_blocking=True)
        self._graph_fi.copy_(fi_s, non_blocking=True)
        self._graph_reward.fill_(float(reward))
        self._graph_q_cur.fill_(float(q_cur_val))
        self._graph_q_next.fill_(float(q_next_val))

    def _capture_update_graph(self, action: int):
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._update_graph_body(action)
        return graph

    def _update_graph_body(self, action: int) -> None:
        delta = self._graph_reward + self._gamma_t * self._graph_q_next - self._graph_q_cur
        scale = self._alpha_t * delta
        self._traces.mul_(self._gam_lam)
        self._traces[action, 0, self._graph_ai] = self._lw[0]
        self._traces[action, 1, self._graph_mi] = self._lw[1]
        self._traces[action, 2, self._graph_fi] = self._lw[2]
        torch.mul(self._traces, scale, out=self._graph_scaled_traces)
        self._weights.add_(self._graph_scaled_traces)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _checkpoint_payload(self) -> dict:
        weights = self._weights.detach()
        if weights.device.type != "cpu":
            weights = weights.cpu()
        return {
            "format_version": self.CHECKPOINT_VERSION,
            "weights": weights,
            "episode": self._episode,
            "epsilon": self.epsilon,
            "memory_size": self.memory_size,
            "n_actions": self.n_actions,
            "dtype": str(self._weights.dtype),
        }

    def save(self, path: str, atomic: bool = False) -> None:
        payload = self._checkpoint_payload()
        if not atomic:
            torch.save(payload, path)
            return

        import os
        tmp_path = f"{path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def load(self, path: str) -> bool:
        import os
        if not os.path.exists(path):
            return False

        try:
            ckpt = torch.load(path, map_location=self._device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self._device)
        except Exception:
            # Back-compat for user-owned checkpoints containing objects that
            # PyTorch's restricted weights-only loader refuses to unpickle.
            ckpt = torch.load(path, map_location=self._device)

        if isinstance(ckpt, torch.Tensor):
            weights = ckpt
            episode = 0
            epsilon = self.epsilon
        else:
            weights = ckpt["weights"]
            # Back-compat: old checkpoints saved "step" instead of "episode".
            episode = int(ckpt.get("episode", ckpt.get("step", 0)))
            epsilon = float(ckpt.get("epsilon", self.epsilon))

        if tuple(weights.shape) != tuple(self._weights.shape):
            raise ValueError(
                "Checkpoint weight shape mismatch: "
                f"checkpoint {tuple(weights.shape)} vs current {tuple(self._weights.shape)}. "
                "Check MEMORY_SIZE, N_ACTIONS, and LCTC layout."
            )
        self._weights.copy_(weights)
        self.reset_traces()
        self._cuda_graphs = [None] * self.n_actions
        self._q_cuda_graph = None
        self._episode = episode
        self.epsilon = epsilon
        return True
