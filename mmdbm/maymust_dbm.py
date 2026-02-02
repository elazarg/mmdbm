"""
May-Must Difference-Bound Matrix (DBM) Abstract Domain

A relational numerical domain that combines over-approximation and under-approximation
to verify properties that standard DBM cannot express precisely.

KEY INSIGHT
===========
For array bounds checking, we need:
- UPPER bound on index (over-approximation): "index could be at most X"
- LOWER bound on size (under-approximation): "size is definitely at least Y"

Standard DBM tracks intervals [L, U] for everything, but:
- We don't CARE about lower bound of index (after verifying index ≥ 0)
- We don't CARE about upper bound of size (knowing size ≤ 1000 doesn't help safety)

More critically, standard DBM uses the SAME approximation direction for everything.
After a join: size ≤ max(10, 7) = 10, telling us "size could be up to 10".
But for safety, we need: size ≥ min(10, 7) = 7, telling us "size is definitely at least 7".

This domain splits variables into:
- E (may/over-approx): track UPPER bounds, join takes MAX
- A (must/under-approx): track LOWER bounds, join takes MIN
- Mixed: track INTERVALS on e - a differences, enabling bidirectional propagation

DOMAIN STRUCTURE
================
- EE[i,j] = upper(e_i - e_j)     # lower = -EE^T
- AA[i,j] = lower(a_i - a_j)     # upper = -AA^T
- Mixed_upper[i,j] = upper(e_{i+1} - a_{j+1})
- Mixed_lower[i,j] = lower(e_{i+1} - a_{j+1})

The interval [Mixed_lower, Mixed_upper] on e - a enables:
- E → A → E propagation: upper(e_i - e_j) ≤ upper(e_i - a_k) + upper(a_k - e_j)
                                          = Mixed_upper[i,k] + (-Mixed_lower[j,k])
- A → E → A propagation: lower(a_i - a_j) ≥ lower(a_i - e_k) + lower(e_k - a_j)
                                          = (-Mixed_upper[k,i]) + Mixed_lower[k,j]

EXAMPLE: ARRAY BOUNDS CHECKING
==============================
```
// Branch 1: initialized 10 elements, accessing index 5
// Branch 2: initialized 7 elements, accessing index 3

After JOIN:
- e_index ≤ max(5, 3) = 5   (over-approx: index could be up to 5)
- a_size ≥ min(10, 7) = 7   (under-approx: size definitely at least 7)

Safety check: e_index < a_size
- upper(index) = 5
- lower(size) = 7
- 5 < 7 ✓ SAFE!

Standard DBM would give: size ∈ [7, 10]
The "could be up to 10" is useless for safety verification.
```
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import math

INF = float('inf')
NINF = float('-inf')

Block = np.ndarray


@dataclass(frozen=True)
class State:
    """
    Abstract state in the May-Must DBM domain.

    Attributes:
        EE: (nE+1) × (nE+1) matrix, EE[i,j] = upper(e_i - e_j)
        AA: (nA+1) × (nA+1) matrix, AA[i,j] = lower(a_i - a_j)
        Mixed_upper: nE × nA matrix, upper(e_{i+1} - a_{j+1})
        Mixed_lower: nE × nA matrix, lower(e_{i+1} - a_{j+1})

    Index 0 represents the constant 0 (for unary constraints).
    """
    EE: Block
    AA: Block
    Mixed_upper: Block
    Mixed_lower: Block

    @property
    def nE(self) -> int:
        """Number of e (may) variables."""
        return self.EE.shape[0] - 1

    @property
    def nA(self) -> int:
        """Number of a (must) variables."""
        return self.AA.shape[0] - 1

    def e_upper(self, i: int) -> float:
        """Upper bound on e_i."""
        return self.EE[i, 0]

    def e_lower(self, i: int) -> float:
        """Lower bound on e_i."""
        return -self.EE[0, i]

    def a_lower(self, j: int) -> float:
        """Lower bound on a_j (the primary bound we track)."""
        return self.AA[j, 0]

    def a_upper(self, j: int) -> float:
        """Upper bound on a_j."""
        return -self.AA[0, j]


def make_state(EE, AA, Mixed_upper, Mixed_lower) -> State:
    """Create a state from array-likes."""
    return State(
        EE=np.array(EE, dtype=float),
        AA=np.array(AA, dtype=float),
        Mixed_upper=np.array(Mixed_upper, dtype=float),
        Mixed_lower=np.array(Mixed_lower, dtype=float),
    )


def top(nE: int, nA: int) -> State:
    """
    Create top (unconstrained) state.

    All e variables: [-∞, +∞]
    All a variables: [-∞, +∞]
    All e - a differences: [-∞, +∞]
    """
    EE = np.full((nE + 1, nE + 1), INF)
    np.fill_diagonal(EE, 0.0)
    AA = np.full((nA + 1, nA + 1), NINF)
    np.fill_diagonal(AA, 0.0)
    Mixed_upper = np.full((nE, nA), INF) if nE > 0 and nA > 0 else np.empty((nE, nA))
    Mixed_lower = np.full((nE, nA), NINF) if nE > 0 and nA > 0 else np.empty((nE, nA))
    return State(EE=EE, AA=AA, Mixed_upper=Mixed_upper, Mixed_lower=Mixed_lower)


def bottom(nE: int, nA: int) -> State:
    """Create bottom (inconsistent) state."""
    st = top(nE, nA)
    EE = st.EE.copy()
    EE[0, 0] = -1  # Negative diagonal indicates bottom
    return State(EE=EE, AA=st.AA, Mixed_upper=st.Mixed_upper, Mixed_lower=st.Mixed_lower)


def closure(state: State) -> State:
    """
    Compute the closure (canonical form) of a state.

    Propagation paths:
    1. EE Floyd-Warshall: e_i → e_k → e_j
    2. AA Floyd-Warshall: a_i → a_k → a_j
    3. Mixed via EE: e_i → e_k → a_j
    4. Mixed via AA: e_i → a_k → a_j
    5. EE via Mixed (E→A→E): e_i → a_k → e_j  [KEY ADDITION]
    6. AA via Mixed (A→E→A): a_i → e_k → a_j  [KEY ADDITION]
    7. Interval consistency: Mixed_lower ≤ Mixed_upper
    8. Unary ↔ Mixed propagation
    """
    EE = state.EE.copy()
    AA = state.AA.copy()
    M_up = state.Mixed_upper.copy()
    M_lo = state.Mixed_lower.copy()

    nE = EE.shape[0] - 1
    nA = AA.shape[0] - 1

    np.fill_diagonal(EE, 0.0)
    np.fill_diagonal(AA, 0.0)

    changed = True
    iterations = 0
    max_iterations = (nE + nA + 2) ** 2  # Safety bound

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        # 1. EE Floyd-Warshall: upper(e_i - e_j) ≤ upper(e_i - e_k) + upper(e_k - e_j)
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

        # 2. AA Floyd-Warshall: lower(a_i - a_j) ≥ lower(a_i - a_k) + lower(a_k - a_j)
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

        # 3. Mixed_upper via EE: upper(e_i - a_j) ≤ upper(e_i - e_k) + upper(e_k - a_j)
        for i in range(nE):
            for j in range(nA):
                best = M_up[i, j]
                for k in range(nE):
                    v = EE[i + 1, k + 1] + M_up[k, j]
                    if v < best:
                        best = v
                # Also via AA: upper(e_i - a_j) ≤ upper(e_i - a_k) + upper(a_k - a_j)
                # upper(a_k - a_j) = -lower(a_j - a_k) = -AA[j+1, k+1]
                for k in range(nA):
                    v = M_up[i, k] - AA[j + 1, k + 1]
                    if v < best:
                        best = v
                if best < M_up[i, j]:
                    M_up[i, j] = best
                    changed = True

        # 4. Mixed_lower via EE: lower(e_i - a_j) ≥ lower(e_i - e_k) + lower(e_k - a_j)
        # lower(e_i - e_k) = -upper(e_k - e_i) = -EE[k+1, i+1]
        for i in range(nE):
            for j in range(nA):
                best = M_lo[i, j]
                for k in range(nE):
                    v = -EE[k + 1, i + 1] + M_lo[k, j]
                    if v > best:
                        best = v
                # Also via AA: lower(e_i - a_j) ≥ lower(e_i - a_k) + lower(a_k - a_j)
                for k in range(nA):
                    v = M_lo[i, k] + AA[k + 1, j + 1]
                    if v > best:
                        best = v
                if best > M_lo[i, j]:
                    M_lo[i, j] = best
                    changed = True

        # 5. EE via Mixed (E → A → E): upper(e_i - e_j) ≤ upper(e_i - a_k) + upper(a_k - e_j)
        # upper(a_k - e_j) = -lower(e_j - a_k) = -M_lo[j-1, k]
        if nA > 0:
            for i in range(1, nE + 1):
                for j in range(1, nE + 1):
                    for k in range(nA):
                        up_e_a = M_up[i - 1, k]
                        lo_e_a = M_lo[j - 1, k]
                        if not math.isinf(up_e_a) and not math.isinf(lo_e_a):
                            up_a_e = -lo_e_a  # upper(a - e) = -lower(e - a)
                            v = up_e_a + up_a_e
                            if v < EE[i, j]:
                                EE[i, j] = v
                                changed = True

        # 6. AA via Mixed (A → E → A): lower(a_i - a_j) ≥ lower(a_i - e_k) + lower(e_k - a_j)
        # lower(a_i - e_k) = -upper(e_k - a_i) = -M_up[k, i-1]
        if nE > 0:
            for i in range(1, nA + 1):
                for j in range(1, nA + 1):
                    for k in range(nE):
                        up_e_a_i = M_up[k, i - 1]
                        lo_e_a_j = M_lo[k, j - 1]
                        if not math.isinf(up_e_a_i) and not math.isinf(lo_e_a_j):
                            lo_a_e = -up_e_a_i
                            v = lo_a_e + lo_e_a_j
                            if v > AA[i, j]:
                                AA[i, j] = v
                                changed = True

        # 7. Interval consistency: lower ≤ upper
        for i in range(nE):
            for j in range(nA):
                if M_lo[i, j] > M_up[i, j]:
                    EE[0, 0] = -1  # Mark as bottom
                    changed = True

        # 8. Unary to Mixed
        for i in range(nE):
            up_e = EE[i + 1, 0]
            lo_e = -EE[0, i + 1]
            for j in range(nA):
                lo_a = AA[j + 1, 0]
                up_a = -AA[0, j + 1]
                # upper(e - a) ≤ upper(e) - lower(a)
                v = up_e - lo_a
                if v < M_up[i, j]:
                    M_up[i, j] = v
                    changed = True
                # lower(e - a) ≥ lower(e) - upper(a)
                v = lo_e - up_a
                if v > M_lo[i, j]:
                    M_lo[i, j] = v
                    changed = True

        # 9. Mixed to Unary
        for i in range(nE):
            for j in range(nA):
                up_a = -AA[0, j + 1]
                lo_a = AA[j + 1, 0]
                # upper(e) ≤ upper(e - a) + upper(a)
                v = M_up[i, j] + up_a
                if v < EE[i + 1, 0]:
                    EE[i + 1, 0] = v
                    changed = True
                # lower(e) ≥ lower(e - a) + lower(a)
                v = M_lo[i, j] + lo_a
                if -v < EE[0, i + 1]:
                    EE[0, i + 1] = -v
                    changed = True

        for j in range(nA):
            for i in range(nE):
                up_e = EE[i + 1, 0]
                lo_e = -EE[0, i + 1]
                # lower(a) ≥ lower(e) - upper(e - a)
                v = lo_e - M_up[i, j]
                if v > AA[j + 1, 0]:
                    AA[j + 1, 0] = v
                    changed = True
                # upper(a) ≤ upper(e) - lower(e - a)
                v = up_e - M_lo[i, j]
                if -v < AA[0, j + 1]:
                    AA[0, j + 1] = -v
                    changed = True

    return State(EE=EE, AA=AA, Mixed_upper=M_up, Mixed_lower=M_lo)


def is_bottom(state: State) -> bool:
    """Check if state is inconsistent (empty concretization)."""
    st = closure(state)
    if np.any(np.diag(st.EE) < 0):
        return True
    if np.any(np.diag(st.AA) > 0):
        return True
    if st.Mixed_upper.size > 0 and np.any(st.Mixed_lower > st.Mixed_upper):
        return True
    return False


def join(s1: State, s2: State) -> State:
    """
    Join (least upper bound) of two states.

    - EE, Mixed_upper: MAX (loosen upper bounds) - over-approximation
    - AA, Mixed_lower: MIN (loosen lower bounds) - under-approximation
    """
    return State(
        EE=np.maximum(s1.EE, s2.EE),
        AA=np.minimum(s1.AA, s2.AA),
        Mixed_upper=np.maximum(s1.Mixed_upper, s2.Mixed_upper),
        Mixed_lower=np.minimum(s1.Mixed_lower, s2.Mixed_lower),
    )


def meet(s1: State, s2: State) -> State:
    """Meet (greatest lower bound) of two states."""
    return State(
        EE=np.minimum(s1.EE, s2.EE),
        AA=np.maximum(s1.AA, s2.AA),
        Mixed_upper=np.minimum(s1.Mixed_upper, s2.Mixed_upper),
        Mixed_lower=np.maximum(s1.Mixed_lower, s2.Mixed_lower),
    )


def widen(s1: State, s2: State) -> State:
    """
    Widening operator for loop analysis.

    - For upper bounds (EE, Mixed_upper): if s2 > s1, go to +∞
    - For lower bounds (AA, Mixed_lower): if s2 < s1, go to -∞
    """
    EE = np.where(s2.EE > s1.EE, INF, s1.EE)
    AA = np.where(s2.AA < s1.AA, NINF, s1.AA)
    M_up = np.where(s2.Mixed_upper > s1.Mixed_upper, INF, s1.Mixed_upper)
    M_lo = np.where(s2.Mixed_lower < s1.Mixed_lower, NINF, s1.Mixed_lower)
    return State(EE=EE, AA=AA, Mixed_upper=M_up, Mixed_lower=M_lo)


# ============================================================================
# Transformers (Abstract Semantics)
# ============================================================================

def assign_e_const(state: State, i: int, c: int) -> State:
    """e_i := c"""
    return assign_e_interval(state, i, c, c)


def assign_e_interval(state: State, i: int, L: int, U: int) -> State:
    """e_i := [L, U] (non-deterministic assignment)"""
    EE = state.EE.copy()
    M_up = state.Mixed_upper.copy()
    M_lo = state.Mixed_lower.copy()

    # Forget old constraints
    for t in range(state.nE + 1):
        if t != i:
            EE[i, t] = INF
            EE[t, i] = INF
    EE[i, i] = 0
    EE[i, 0] = U   # e_i ≤ U
    EE[0, i] = -L  # e_i ≥ L

    if state.nA > 0:
        M_up[i - 1, :] = INF
        M_lo[i - 1, :] = NINF

    return closure(State(EE=EE, AA=state.AA, Mixed_upper=M_up, Mixed_lower=M_lo))


def assign_a_const(state: State, j: int, c: int) -> State:
    """a_j := c"""
    return assign_a_interval(state, j, c, c)


def assign_a_interval(state: State, j: int, L: int, U: int) -> State:
    """a_j := [L, U]"""
    AA = state.AA.copy()
    M_up = state.Mixed_upper.copy()
    M_lo = state.Mixed_lower.copy()

    for t in range(state.nA + 1):
        if t != j:
            AA[j, t] = NINF
            AA[t, j] = NINF
    AA[j, j] = 0
    AA[j, 0] = L   # a_j ≥ L
    AA[0, j] = -U  # a_j ≤ U

    if state.nE > 0:
        M_up[:, j - 1] = INF
        M_lo[:, j - 1] = NINF

    return closure(State(EE=state.EE, AA=AA, Mixed_upper=M_up, Mixed_lower=M_lo))


def assign_e_from_a(state: State, i: int, j: int, c: int = 0) -> State:
    """e_i := a_j + c"""
    EE = state.EE.copy()
    M_up = state.Mixed_upper.copy()
    M_lo = state.Mixed_lower.copy()

    # Forget e_i
    for t in range(state.nE + 1):
        if t != i:
            EE[i, t] = INF
            EE[t, i] = INF
    EE[i, i] = 0
    if state.nA > 0:
        M_up[i - 1, :] = INF
        M_lo[i - 1, :] = NINF

    # e_i - a_j = c (exact relationship)
    M_up[i - 1, j - 1] = c
    M_lo[i - 1, j - 1] = c

    return closure(State(EE=EE, AA=state.AA, Mixed_upper=M_up, Mixed_lower=M_lo))


def assign_a_from_e(state: State, j: int, i: int, c: int = 0) -> State:
    """a_j := e_i + c"""
    AA = state.AA.copy()
    M_up = state.Mixed_upper.copy()
    M_lo = state.Mixed_lower.copy()

    for t in range(state.nA + 1):
        if t != j:
            AA[j, t] = NINF
            AA[t, j] = NINF
    AA[j, j] = 0
    if state.nE > 0:
        M_up[:, j - 1] = INF
        M_lo[:, j - 1] = NINF

    # e_i - a_j = -c
    M_up[i - 1, j - 1] = -c
    M_lo[i - 1, j - 1] = -c

    return closure(State(EE=state.EE, AA=AA, Mixed_upper=M_up, Mixed_lower=M_lo))


def guard_e_le(state: State, i: int, c: int) -> State:
    """assume(e_i ≤ c)"""
    EE = state.EE.copy()
    EE[i, 0] = min(EE[i, 0], c)
    return closure(State(EE=EE, AA=state.AA, Mixed_upper=state.Mixed_upper, Mixed_lower=state.Mixed_lower))


def guard_e_ge(state: State, i: int, c: int) -> State:
    """assume(e_i ≥ c)"""
    EE = state.EE.copy()
    EE[0, i] = min(EE[0, i], -c)
    return closure(State(EE=EE, AA=state.AA, Mixed_upper=state.Mixed_upper, Mixed_lower=state.Mixed_lower))


def guard_a_ge(state: State, j: int, c: int) -> State:
    """assume(a_j ≥ c)"""
    AA = state.AA.copy()
    AA[j, 0] = max(AA[j, 0], c)
    return closure(State(EE=state.EE, AA=AA, Mixed_upper=state.Mixed_upper, Mixed_lower=state.Mixed_lower))


def guard_a_le(state: State, j: int, c: int) -> State:
    """assume(a_j ≤ c)"""
    AA = state.AA.copy()
    AA[0, j] = max(AA[0, j], -c)
    return closure(State(EE=state.EE, AA=AA, Mixed_upper=state.Mixed_upper, Mixed_lower=state.Mixed_lower))


def guard_e_minus_a_le(state: State, i: int, j: int, c: int) -> State:
    """assume(e_i - a_j ≤ c)"""
    M_up = state.Mixed_upper.copy()
    M_up[i - 1, j - 1] = min(M_up[i - 1, j - 1], c)
    return closure(State(EE=state.EE, AA=state.AA, Mixed_upper=M_up, Mixed_lower=state.Mixed_lower))


def guard_e_minus_a_ge(state: State, i: int, j: int, c: int) -> State:
    """assume(e_i - a_j ≥ c)"""
    M_lo = state.Mixed_lower.copy()
    M_lo[i - 1, j - 1] = max(M_lo[i - 1, j - 1], c)
    return closure(State(EE=state.EE, AA=state.AA, Mixed_upper=state.Mixed_upper, Mixed_lower=M_lo))


def guard_e_lt_a(state: State, i: int, j: int) -> State:
    """assume(e_i < a_j), i.e., e_i - a_j ≤ -1"""
    return guard_e_minus_a_le(state, i, j, -1)


# ============================================================================
# Queries
# ============================================================================

def check_e_lt_a(state: State, i: int, j: int) -> bool:
    """
    Check if e_i < a_j holds for ALL concrete states.

    This is the key query for array bounds checking:
    - i is the index variable (may/over-approx)
    - j is the size variable (must/under-approx)

    Returns True if upper(e_i) < lower(a_j).
    """
    st = closure(state)
    return st.e_upper(i) < st.a_lower(j)


def check_e_le_a(state: State, i: int, j: int) -> bool:
    """Check if e_i ≤ a_j holds for all concrete states."""
    st = closure(state)
    return st.e_upper(i) <= st.a_lower(j)


def check_safe_access(state: State, index_var: int, size_var: int) -> bool:
    """
    Check if array access is safe: 0 ≤ index < size.

    Args:
        index_var: The e variable representing the index (1-based)
        size_var: The a variable representing the array size (1-based)

    Returns True if the access is provably safe.
    """
    st = closure(state)
    # index ≥ 0
    if st.e_lower(index_var) < 0:
        return False
    # index < size
    return check_e_lt_a(st, index_var, size_var)


# ============================================================================
# Concretization (for testing)
# ============================================================================

def satisfies(state: State, rho: Dict[str, int]) -> bool:
    """Check if concrete assignment rho satisfies all constraints."""
    EE, AA = state.EE, state.AA
    M_up, M_lo = state.Mixed_upper, state.Mixed_lower
    nE, nA = state.nE, state.nA

    def ve(i): return 0 if i == 0 else int(rho.get(f"e{i}", 0))
    def va(j): return 0 if j == 0 else int(rho.get(f"a{j}", 0))

    # EE constraints
    for i in range(nE + 1):
        for j in range(nE + 1):
            if not math.isinf(EE[i, j]) and ve(i) - ve(j) > EE[i, j]:
                return False

    # AA constraints
    for i in range(nA + 1):
        for j in range(nA + 1):
            if not math.isinf(AA[i, j]) and va(i) - va(j) < AA[i, j]:
                return False

    # Mixed constraints
    for i in range(nE):
        for j in range(nA):
            diff = ve(i + 1) - va(j + 1)
            if not math.isinf(M_up[i, j]) and diff > M_up[i, j]:
                return False
            if not math.isinf(M_lo[i, j]) and diff < M_lo[i, j]:
                return False

    return True


def gamma_enumerate(state: State, e_vals: List[int], a_vals: List[int]) -> List[Dict[str, int]]:
    """Enumerate all satisfying assignments on given grid (for testing)."""
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


# ============================================================================
# Translation to/from Explicit Representation (Pedagogic)
# ============================================================================

def to_explicit(state: State):
    """
    Convert to explicit representation (5 matrices).

    The explicit representation separates:
    - EA = upper(e - a) = Mixed_upper
    - AE = lower(a - e) = -Mixed_upper^T (derived)
    - AE_upper = upper(a - e) = -Mixed_lower^T

    This shows the relationship between the two representations.
    """
    # Import here to avoid circular dependency
    from . import maymust_dbm_explicit as explicit

    nE, nA = state.nE, state.nA

    # EA = Mixed_upper
    EA = state.Mixed_upper.copy()

    # AE = lower(a - e) = -upper(e - a) for transposed indices
    # Actually: lower(a_i - e_j) comes from upper(e_j - a_i)
    # AE[i,j] = lower(a_{i+1} - e_{j+1})
    # This equals -upper(e_{j+1} - a_{i+1}) = -Mixed_upper[j, i]
    AE = np.full((nA, nE), NINF)
    for i in range(nA):
        for j in range(nE):
            AE[i, j] = -state.Mixed_upper[j, i]

    # AE_upper = upper(a - e) = -lower(e - a) for transposed indices
    # AE_upper[i,j] = upper(a_{i+1} - e_{j+1}) = -lower(e_{j+1} - a_{i+1}) = -Mixed_lower[j, i]
    AE_upper = np.full((nA, nE), INF)
    for i in range(nA):
        for j in range(nE):
            AE_upper[i, j] = -state.Mixed_lower[j, i]

    return explicit.State(
        EE=state.EE.copy(),
        AA=state.AA.copy(),
        EA=EA,
        AE=AE,
        AE_upper=AE_upper
    )


def from_explicit(explicit_state) -> State:
    """Convert from explicit representation back to interval representation."""
    nE = explicit_state.nE
    nA = explicit_state.nA

    # Mixed_upper = EA
    Mixed_upper = explicit_state.EA.copy()

    # Mixed_lower[i,j] = lower(e_{i+1} - a_{j+1})
    # = -upper(a_{j+1} - e_{i+1}) = -AE_upper[j, i]
    Mixed_lower = np.full((nE, nA), NINF)
    for i in range(nE):
        for j in range(nA):
            Mixed_lower[i, j] = -explicit_state.AE_upper[j, i]

    return State(
        EE=explicit_state.EE.copy(),
        AA=explicit_state.AA.copy(),
        Mixed_upper=Mixed_upper,
        Mixed_lower=Mixed_lower
    )
