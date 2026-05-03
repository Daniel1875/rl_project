"""
Reward functions — Section 4.2, Equations 3–6 of Spooner et al. (2018).

  Ψ(t) = ψ_a + ψ_b + Inv(t) · Δm(t)               (Eq. 3)
  r_pnl        = Ψ(t)                               (Eq. 4)
  r_symm       = Ψ(t) − η · Inv(t) · Δm(t)         (Eq. 5)
  r_asymm      = Ψ(t) − max(0, η · Inv(t) · Δm(t)) (Eq. 6)  ← consolidated
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RewardComponents:
    psi_ask:  float   # matched_ask × (p_a − mid)
    psi_bid:  float   # matched_bid × (mid − p_b)
    inv_term: float   # Inv(t) × Δm(t)
    pnl:      float   # Ψ(t)

    def symm_damped(self, eta: float) -> float:
        return self.pnl - eta * self.inv_term

    def asymm_damped(self, eta: float) -> float:
        return self.pnl - max(0.0, eta * self.inv_term)


def compute(
    matched_ask: int,
    matched_bid: int,
    p_ask:       float,
    p_bid:       float,
    mid:         float,
    prev_mid:    float,
    inventory:   int,
) -> RewardComponents:
    psi_ask  = matched_ask * (p_ask - mid)
    psi_bid  = matched_bid * (mid   - p_bid)
    inv_term = inventory * (mid - prev_mid)
    return RewardComponents(psi_ask, psi_bid, inv_term,
                            psi_ask + psi_bid + inv_term)


class PnLReward:
    def __call__(self, rc: RewardComponents) -> float:
        return rc.pnl


class SymmetricDampedReward:
    def __init__(self, eta: float):
        self.eta = eta
    def __call__(self, rc: RewardComponents) -> float:
        return rc.symm_damped(self.eta)


class AsymmetricDampedReward:
    def __init__(self, eta: float):
        self.eta = eta
    def __call__(self, rc: RewardComponents) -> float:
        return rc.asymm_damped(self.eta)
