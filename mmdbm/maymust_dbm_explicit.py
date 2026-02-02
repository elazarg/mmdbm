"""
Option 2: Explicit upper(a-e) matrix

Keep may/must polarity explicit but add the missing bound.
This enables direct E → A → E and A → E → A propagation.

Structure:
- EE[i,j] = upper(e_i - e_j)  [may, over-approx]
- AA[i,j] = lower(a_i - a_j)  [must, under-approx]
- EA[i,j] = upper(e_{i+1} - a_{j+1})  [may]
- AE[i,j] = lower(a_{i+1} - e_{j+1})  [must]
- AE_upper[i,j] = upper(a_{i+1} - e_{j+1})  [NEW - enables E→A→E]

From AE_upper we derive: lower(e - a) = -upper(a - e) = -AE_upper

Key insight: AE_upper has OPPOSITE polarity from AE:
- AE (lower) uses MAX for tightening, MIN for join (under-approx)
- AE_upper (upper) uses MIN for tightening, MAX for join (over-approx)
"""

from dataclasses import dataclass
import numpy as np
import math
from typing import Tuple

INF = float('inf')
NINF = float('-inf')

Block = np.ndarray


@dataclass(frozen=True)
class State:
    """
    Combined may-must DBM state with explicit upper(a-e).

    May (over-approx, tighten with MIN, join with MAX):
        EE, EA, AE_upper
    Must (under-approx, tighten with MAX, join with MIN):
        AA, AE
    """
    EE: Block       # (nE+1) x (nE+1), upper(e - e)
    AA: Block       # (nA+1) x (nA+1), lower(a - a)
    EA: Block       # nE x nA, upper(e - a)
    AE: Block       # nA x nE, lower(a - e)
    AE_upper: Block # nA x nE, upper(a - e) [NEW]

    @property
    def nE(self) -> int:
        return self.EE.shape[0] - 1

    @property
    def nA(self) -> int:
        return self.AA.shape[0] - 1


def make_state(EE, AA, EA, AE, AE_upper=None) -> State:
    """Create a state from array-likes."""
    EA_arr = np.array(EA, dtype=float)
    AE_arr = np.array(AE, dtype=float)
    if AE_upper is None:
        # Initialize AE_upper to INF (no upper bound on a - e)
        AE_upper = np.full_like(AE_arr, INF)
    return State(
        EE=np.array(EE, dtype=float),
        AA=np.array(AA, dtype=float),
        EA=EA_arr,
        AE=AE_arr,
        AE_upper=np.array(AE_upper, dtype=float),
    )


def top_state(nE: int, nA: int) -> State:
    """Create top (unconstrained) state."""
    EE = np.full((nE + 1, nE + 1), INF)
    np.fill_diagonal(EE, 0.0)
    AA = np.full((nA + 1, nA + 1), NINF)
    np.fill_diagonal(AA, 0.0)
    EA = np.full((nE, nA), INF)
    AE = np.full((nA, nE), NINF)
    AE_upper = np.full((nA, nE), INF)
    return State(EE=EE, AA=AA, EA=EA, AE=AE, AE_upper=AE_upper)


def closure(state: State) -> State:
    """
    Compute closure with full bidirectional propagation.

    Key additions from original:
    - E → A → E path using EA + AE_upper
    - A → E → A path using AE + (-EA) or AE_upper derivations
    - Consistency between AE and AE_upper: AE ≤ AE_upper (lower ≤ upper)
    - Cross-band: EA ≤ -AE and AE_upper ≥ -EA (dual consistency)
    """
    EE = state.EE.copy()
    AA = state.AA.copy()
    EA = state.EA.copy()
    AE = state.AE.copy()
    AE_upper = state.AE_upper.copy()

    nE = EE.shape[0] - 1
    nA = AA.shape[0] - 1

    np.fill_diagonal(EE, 0.0)
    np.fill_diagonal(AA, 0.0)

    changed = True
    while changed:
        changed = False

        # 1. EE Floyd-Warshall
        for k in range(nE + 1):
            for i in range(nE + 1):
                ik = EE[i, k]
                if math.isinf(ik):
                    continue
                for j in range(nE + 1):
                    v = ik + EE[k, j]
                    if v < EE[i, j]:
                        EE[i, j] = v
                        changed = True
        np.fill_diagonal(EE, 0.0)

        # 2. AA Floyd-Warshall
        for k in range(nA + 1):
            for i in range(nA + 1):
                ik = AA[i, k]
                if math.isinf(ik):
                    continue
                for j in range(nA + 1):
                    v = ik + AA[k, j]
                    if v > AA[i, j]:
                        AA[i, j] = v
                        changed = True
        np.fill_diagonal(AA, 0.0)

        # 3. EA via EE: upper(e_i - a_j) ≤ upper(e_i - e_k) + upper(e_k - a_j)
        for i in range(nE):
            for j in range(nA):
                best = EA[i, j]
                for k in range(nE):
                    v = EE[i + 1, k + 1] + EA[k, j]
                    if v < best:
                        best = v
                # EA via AA: upper(e_i - a_j) ≤ upper(e_i - a_k) - lower(a_j - a_k)
                for k in range(nA):
                    v = EA[i, k] - AA[j + 1, k + 1]
                    if v < best:
                        best = v
                if best < EA[i, j]:
                    EA[i, j] = best
                    changed = True

        # 4. AE via AA: lower(a_i - e_j) ≥ lower(a_i - a_k) + lower(a_k - e_j)
        for i in range(nA):
            for j in range(nE):
                best = AE[i, j]
                for k in range(nA):
                    v = AA[i + 1, k + 1] + AE[k, j]
                    if v > best:
                        best = v
                # AE via EE: lower(a_i - e_j) ≥ lower(a_i - e_k) - upper(e_j - e_k)
                for k in range(nE):
                    v = AE[i, k] - EE[j + 1, k + 1]
                    if v > best:
                        best = v
                if best > AE[i, j]:
                    AE[i, j] = best
                    changed = True

        # 5. AE_upper via AA: upper(a_i - e_j) ≤ upper(a_i - a_k) + upper(a_k - e_j)
        # upper(a_i - a_k) = -lower(a_k - a_i) = -AA[k+1, i+1]
        for i in range(nA):
            for j in range(nE):
                best = AE_upper[i, j]
                for k in range(nA):
                    v = -AA[k + 1, i + 1] + AE_upper[k, j]
                    if v < best:
                        best = v
                # AE_upper via EE: upper(a_i - e_j) ≤ upper(a_i - e_k) + upper(e_k - e_j)
                # upper(e_k - e_j) = EE[k+1, j+1]
                for k in range(nE):
                    v = AE_upper[i, k] + EE[k + 1, j + 1]
                    if v < best:
                        best = v
                if best < AE_upper[i, j]:
                    AE_upper[i, j] = best
                    changed = True

        # 6. EE via Mixed (E → A → E): upper(e_i - e_j) ≤ upper(e_i - a_k) + upper(a_k - e_j)
        if nA > 0:
            for i in range(1, nE + 1):
                for j in range(1, nE + 1):
                    for k in range(nA):
                        ea = EA[i - 1, k]
                        ae_up = AE_upper[k, j - 1]
                        if not math.isinf(ea) and not math.isinf(ae_up):
                            v = ea + ae_up
                            if v < EE[i, j]:
                                EE[i, j] = v
                                changed = True

        # 7. AA via Mixed (A → E → A): lower(a_i - a_j) ≥ lower(a_i - e_k) + lower(e_k - a_j)
        # lower(e_k - a_j) = -upper(a_j - e_k) = -AE_upper[j-1, k]
        if nE > 0:
            for i in range(1, nA + 1):
                for j in range(1, nA + 1):
                    for k in range(nE):
                        ae = AE[i - 1, k]
                        ae_up_j = AE_upper[j - 1, k]
                        if not math.isinf(ae) and not math.isinf(ae_up_j):
                            lower_e_a = -ae_up_j
                            v = ae + lower_e_a
                            if v > AA[i, j]:
                                AA[i, j] = v
                                changed = True

        # 8. Cross-band consistency
        # EA[i,j] ≤ -AE[j,i] (upper(e-a) ≤ -lower(a-e))
        # AE[j,i] ≥ -EA[i,j] (lower(a-e) ≥ -upper(e-a))
        # AE_upper[j,i] ≥ -EA[i,j] would mean upper(a-e) ≥ -upper(e-a)...
        #   which is always true if lower(a-e) ≥ -upper(e-a)
        # Actually: AE_upper should satisfy AE ≤ AE_upper
        for i in range(nE):
            for j in range(nA):
                neg_ae = -AE[j, i]
                if neg_ae < EA[i, j]:
                    EA[i, j] = neg_ae
                    changed = True
                neg_ea = -EA[i, j]
                if neg_ea > AE[j, i]:
                    AE[j, i] = neg_ea
                    changed = True

        # 9. AE interval consistency: lower ≤ upper
        for i in range(nA):
            for j in range(nE):
                if AE[i, j] > AE_upper[i, j]:
                    # Derive upper from lower at minimum
                    AE_upper[i, j] = AE[i, j]
                    changed = True

        # 10. AE_upper / EA cross-consistency
        # upper(a - e) = -lower(e - a), and we want lower(e-a) ≥ -upper(a-e)
        # So if we set lower(e-a) = -AE_upper, then EA should satisfy EA ≥ -AE_upper...
        # No wait, EA is upper(e-a), not lower(e-a).
        # What we can derive: lower(e-a) = -upper(a-e) = -AE_upper
        # For consistency: lower(e-a) ≤ upper(e-a), so -AE_upper ≤ EA
        for i in range(nE):
            for j in range(nA):
                lower_e_a = -AE_upper[j, i]
                if lower_e_a > EA[i, j]:
                    # This is an inconsistency: lower > upper
                    EE[i + 1, i + 1] = -1  # Mark as bottom
                    changed = True

        # 11. Unary to mixed
        for i in range(nE):
            upper_e = EE[i + 1, 0]
            for j in range(nA):
                lower_a = AA[j + 1, 0]
                v = upper_e - lower_a
                if v < EA[i, j]:
                    EA[i, j] = v
                    changed = True

        for i in range(nA):
            lower_a = AA[i + 1, 0]
            upper_a = -AA[0, i + 1]
            for j in range(nE):
                upper_e = EE[j + 1, 0]
                lower_e = -EE[0, j + 1]
                # lower(a - e) ≥ lower(a) - upper(e)
                v = lower_a - upper_e
                if v > AE[i, j]:
                    AE[i, j] = v
                    changed = True
                # upper(a - e) ≤ upper(a) - lower(e)
                v = upper_a - lower_e
                if v < AE_upper[i, j]:
                    AE_upper[i, j] = v
                    changed = True

        # 12. Mixed to unary
        for i in range(nA):
            for j in range(nE):
                lower_e = -EE[0, j + 1]
                v = AE[i, j] + lower_e  # lower(a) ≥ lower(a - e) + lower(e)
                if v > AA[i + 1, 0]:
                    AA[i + 1, 0] = v
                    changed = True

        for i in range(nE):
            for j in range(nA):
                upper_a = -AA[0, j + 1]
                v = EA[i, j] + upper_a  # upper(e) ≤ upper(e - a) + upper(a)
                if v < EE[i + 1, 0]:
                    EE[i + 1, 0] = v
                    changed = True

    return State(EE=EE, AA=AA, EA=EA, AE=AE, AE_upper=AE_upper)


def is_bottom(state: State) -> bool:
    """Check if state is inconsistent."""
    st = closure(state)
    if np.any(np.diag(st.EE) < 0):
        return True
    if np.any(np.diag(st.AA) > 0):
        return True
    # Check AE interval consistency
    if np.any(st.AE > st.AE_upper):
        return True
    return False


def join(s1: State, s2: State) -> State:
    """Join (least upper bound) of two states."""
    return State(
        EE=np.maximum(s1.EE, s2.EE),           # may: loosen upper
        AA=np.minimum(s1.AA, s2.AA),           # must: loosen lower
        EA=np.maximum(s1.EA, s2.EA),           # may: loosen upper
        AE=np.minimum(s1.AE, s2.AE),           # must: loosen lower
        AE_upper=np.maximum(s1.AE_upper, s2.AE_upper),  # may: loosen upper
    )


def meet(s1: State, s2: State) -> State:
    """Meet (greatest lower bound) of two states."""
    return State(
        EE=np.minimum(s1.EE, s2.EE),
        AA=np.maximum(s1.AA, s2.AA),
        EA=np.minimum(s1.EA, s2.EA),
        AE=np.maximum(s1.AE, s2.AE),
        AE_upper=np.minimum(s1.AE_upper, s2.AE_upper),
    )


def satisfies(state: State, rho: dict) -> bool:
    """Check if concrete state rho satisfies all constraints."""
    EE, AA, EA, AE, AE_upper = state.EE, state.AA, state.EA, state.AE, state.AE_upper
    nE, nA = state.nE, state.nA

    def ve(i): return 0 if i == 0 else int(rho[f"e{i}"])
    def va(j): return 0 if j == 0 else int(rho[f"a{j}"])

    # EE
    for i in range(nE + 1):
        for j in range(nE + 1):
            if not math.isinf(EE[i, j]) and ve(i) - ve(j) > EE[i, j]:
                return False

    # AA
    for i in range(nA + 1):
        for j in range(nA + 1):
            if not math.isinf(AA[i, j]) and va(i) - va(j) < AA[i, j]:
                return False

    # EA
    for i in range(nE):
        for j in range(nA):
            if not math.isinf(EA[i, j]) and ve(i + 1) - va(j + 1) > EA[i, j]:
                return False

    # AE and AE_upper
    for i in range(nA):
        for j in range(nE):
            diff = va(i + 1) - ve(j + 1)
            if not math.isinf(AE[i, j]) and diff < AE[i, j]:
                return False
            if not math.isinf(AE_upper[i, j]) and diff > AE_upper[i, j]:
                return False

    return True


def gamma_enumerate(state: State, e_vals: list, a_vals: list) -> list:
    """Enumerate all satisfying assignments on given grid."""
    nE, nA = state.nE, state.nA
    results = []

    def gen_e(prefix, i):
        if i > nE:
            gen_a(prefix, 1)
            return
        for v in e_vals:
            p = dict(prefix)
            p[f"e{i}"] = v
            gen_e(p, i + 1)

    def gen_a(prefix, j):
        if j > nA:
            if satisfies(state, prefix):
                results.append(prefix)
            return
        for v in a_vals:
            p = dict(prefix)
            p[f"a{j}"] = v
            gen_a(p, j + 1)

    gen_e({}, 1)
    return results


# Transformer helpers
def assign_e_interval(state: State, i: int, L: int, U: int) -> State:
    """Assign e_i to interval [L, U]."""
    EE = state.EE.copy()
    EA = state.EA.copy()
    AE = state.AE.copy()
    AE_upper = state.AE_upper.copy()

    for t in range(state.nE + 1):
        if t != i:
            EE[i, t] = INF
            EE[t, i] = INF
    EE[i, i] = 0
    EE[i, 0] = U
    EE[0, i] = -L

    if state.nA > 0:
        EA[i - 1, :] = INF
        AE[:, i - 1] = NINF
        AE_upper[:, i - 1] = INF

    return closure(State(EE=EE, AA=state.AA, EA=EA, AE=AE, AE_upper=AE_upper))


def assign_a_interval(state: State, j: int, L: int, U: int) -> State:
    """Assign a_j to interval [L, U]."""
    AA = state.AA.copy()
    EA = state.EA.copy()
    AE = state.AE.copy()
    AE_upper = state.AE_upper.copy()

    for t in range(state.nA + 1):
        if t != j:
            AA[j, t] = NINF
            AA[t, j] = NINF
    AA[j, j] = 0
    AA[j, 0] = L
    AA[0, j] = -U

    if state.nE > 0:
        EA[:, j - 1] = INF
        AE[j - 1, :] = NINF
        AE_upper[j - 1, :] = INF

    return closure(State(EE=state.EE, AA=AA, EA=EA, AE=AE, AE_upper=AE_upper))


def guard_ea_le(state: State, i: int, j: int, c: int) -> State:
    """Add constraint e_i - a_j ≤ c."""
    EA = state.EA.copy()
    EA[i - 1, j - 1] = min(EA[i - 1, j - 1], c)
    return closure(State(EE=state.EE, AA=state.AA, EA=EA, AE=state.AE, AE_upper=state.AE_upper))


def guard_ae_ge(state: State, i: int, j: int, c: int) -> State:
    """Add constraint a_i - e_j ≥ c."""
    AE = state.AE.copy()
    AE[i - 1, j - 1] = max(AE[i - 1, j - 1], c)
    return closure(State(EE=state.EE, AA=state.AA, EA=state.EA, AE=AE, AE_upper=state.AE_upper))


def guard_ae_le(state: State, i: int, j: int, c: int) -> State:
    """Add constraint a_i - e_j ≤ c (uses AE_upper)."""
    AE_upper = state.AE_upper.copy()
    AE_upper[i - 1, j - 1] = min(AE_upper[i - 1, j - 1], c)
    return closure(State(EE=state.EE, AA=state.AA, EA=state.EA, AE=state.AE, AE_upper=AE_upper))
