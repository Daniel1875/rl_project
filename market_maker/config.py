"""
Hyperparameters — Table 2 of Spooner et al. (2018).

All values match the paper exactly unless noted otherwise.
"""

# --- Training schedule ---
TRAIN_EPISODES   = 1000   # episodes (days) sampled for training
TEST_SAMPLE_DAYS = 40     # held-out test days (chronologically last)
VAL_SAMPLE_DAYS  = 28     # validation days immediately before test

# --- Tile coding ---
N_TILINGS     = 32
N_TILES       = 8         # tiles per dimension per tiling
MEMORY_SIZE   = 200_000   # hash table size per tile coding
               # Paper uses 10^7. With a CUDA GPU the stacked weight/trace
               # tensor (10 x 3 x MEMORY_SIZE float32) is processed in a
               # single kernel; 200_000 gives 24 MB tensors and ~0.3 ms/step.
               # Without CUDA, each episode at this size takes ~15 min.
               # CPU-only users should set this to 10_000 (~2 min/episode).
LCTC_WEIGHTS  = (0.6, 0.1, 0.3)  # λ_i for [agent, market, full]

# --- TD learning ---
LEARNING_RATE  = 0.001    # α
DISCOUNT       = 0.97     # γ
TRACE_PARAM    = 0.96     # λ (eligibility trace decay)

# --- Exploration ---
EPSILON_START = 0.7
EPSILON_FLOOR = 0.0001
EPSILON_T     = 1000      # decay over this many *training episodes* (not LOB events)
                          # ε drops linearly from EPSILON_START → EPSILON_FLOOR
                          # across EPSILON_T episodes, then stays at the floor.

# --- Trading / inventory ---
ORDER_SIZE    = 100        # shares per limit order
               # NOTE: paper uses ω=1000 for ~1 GBp UK stocks.
               # BIIB trades at ~$300/share so 1000 shares = $300K/order,
               # which would dominate the book. We scale down to 100 shares
               # (~$30K) to keep the agent's impact negligible (Section 3).
MIN_INVENTORY = -1000
MAX_INVENTORY =  1000     # scaled proportionally with ORDER_SIZE

# --- Reward dampening ---
DAMPING_ETA = 0.6         # η

# --- Spread scale factor ---
SPREAD_MA_WINDOW = 100    # events for moving-average half-spread

# --- State feature lookbacks ---
VOLATILITY_WINDOW = 50
RSI_WINDOW        = 14

# --- Action space (Table 1) ---
# (theta_ask, theta_bid); action 9 = market order to clear inventory
ACTIONS = [
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (1, 3),
    (3, 1),
    (2, 5),
    (5, 2),
]
N_ACTIONS = 10  # 9 limit order pairs + 1 market order

# --- Regular trading hours (UTC) ---
RTH_START_HOUR   = 14   # 09:30 ET = 14:30 UTC
RTH_START_MINUTE = 30
RTH_END_HOUR     = 21   # 16:00 ET = 21:00 UTC
RTH_END_MINUTE   = 0
