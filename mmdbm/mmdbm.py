"""
mmdbm: Combined May–Must Difference-Bound Domain (NumPy-based implementation)

Public API:

- make_state(EE, AA, EA, AE) -> State
- get_blocks(state) -> (EE, AA, EA, AE)
- closure(state) -> State
- satisfies(state, rho: dict[str, int]) -> bool
- gamma_enumerate(state, e_vals: list[int], a_vals: list[int]) -> list[dict[str, int]]
- alpha_outer_EE(P: list[dict[str, int]]) -> np.ndarray
- alpha_outer_EA(S: list[dict[str, int]]) -> np.ndarray
- alpha_inner_AA(Q: list[dict[str, int]]) -> np.ndarray
- alpha_inner_AE(S: list[dict[str, int]]) -> np.ndarray
- meet(s1: State, s2: State) -> State
- join(s1: State, s2: State) -> State
- is_bottom(state: State) -> bool

Blocks are NumPy ndarrays (dtype=float to support +/-inf).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np

# Keep names typed as int per your request (values are inf/-inf).
INF: int = math.inf  # type: ignore[assignment]
NINF: int = -math.inf  # type: ignore[assignment]

Block = np.ndarray


@dataclass
class State:
    EE: Block  # (nE+1) x (nE+1) upper bounds (<=)
    AA: Block  # (nA+1) x (nA+1) lower bounds (>=)
    EA: Block  # nE x nA upper bounds (<=)
    AE: Block  # nA x nE lower bounds (>=)


# ---------------- Helpers ----------------


def _copy(M: Block) -> Block:
    return np.array(M, dtype=float, copy=True)


def _zero_closed_blocks(nE: int, nA: int) -> tuple[Block, Block, Block, Block]:
    EE = np.full((nE + 1, nE + 1), INF, dtype=float)
    AA = np.full((nA + 1, nA + 1), NINF, dtype=float)
    EA = np.full((nE, nA), INF, dtype=float)
    AE = np.full((nA, nE), NINF, dtype=float)
    np.fill_diagonal(EE, 0.0)
    np.fill_diagonal(AA, 0.0)
    return EE, AA, EA, AE


def _max_e_index(dicts: list[dict[str, int]]) -> int:
    m = 0
    for d in dicts:
        for k in d:
            if k.startswith("e"):
                try:
                    m = max(m, int(k[1:]))
                except Exception:
                    pass
    return m


def _max_a_index(dicts: list[dict[str, int]]) -> int:
    m = 0
    for d in dicts:
        for k in d:
            if k.startswith("a"):
                try:
                    m = max(m, int(k[1:]))
                except Exception:
                    pass
    return m


# ------------- State API -------------


def _normalize_to_shapes(
    EE: Block, AA: Block, EA: Block, AE: Block
) -> tuple[Block, Block, Block, Block]:
    """Pad/truncate all blocks so they are mutually consistent.

    nE := max(EE.shape[0]-1, EA.shape[0], AE.shape[1])
    nA := max(AA.shape[0]-1, EA.shape[1], AE.shape[0])

    EE/AA are (n+1)x(n+1); EA is nE x nA; AE is nA x nE.
    Unfilled entries get tops: EE,EA -> +inf; AA,AE -> -inf; and diag(EE)=diag(AA)=0.
    """
    EE = np.array(EE, dtype=float, copy=True)
    AA = np.array(AA, dtype=float, copy=True)
    EA = np.array(EA, dtype=float, copy=True)
    AE = np.array(AE, dtype=float, copy=True)

    # current sizes (if a block is 0-d or 1-d, treat missing dims as zero)
    ne_from_EE = max(0, EE.shape[0] - 1) if EE.ndim == 2 else 0
    na_from_AA = max(0, AA.shape[0] - 1) if AA.ndim == 2 else 0
    ne_from_EA = EA.shape[0] if EA.ndim == 2 else 0
    na_from_EA = EA.shape[1] if EA.ndim == 2 else 0
    na_from_AE = AE.shape[0] if AE.ndim == 2 else 0
    ne_from_AE = AE.shape[1] if AE.ndim == 2 else 0

    nE = max(ne_from_EE, ne_from_EA, ne_from_AE)
    nA = max(na_from_AA, na_from_EA, na_from_AE)

    # allocate canonical shapes with tops
    EE_new = np.full((nE + 1, nE + 1), INF, dtype=float)
    AA_new = np.full((nA + 1, nA + 1), NINF, dtype=float)
    EA_new = np.full((nE, nA), INF, dtype=float)
    AE_new = np.full((nA, nE), NINF, dtype=float)
    np.fill_diagonal(EE_new, 0.0)
    np.fill_diagonal(AA_new, 0.0)

    # copy intersections
    if EE.ndim == 2:
        i = min(EE.shape[0], nE + 1)
        j = min(EE.shape[1], nE + 1)
        EE_new[:i, :j] = EE[:i, :j]
        np.fill_diagonal(EE_new, 0.0)  # keep diag normalized
    if AA.ndim == 2:
        i = min(AA.shape[0], nA + 1)
        j = min(AA.shape[1], nA + 1)
        AA_new[:i, :j] = AA[:i, :j]
        np.fill_diagonal(AA_new, 0.0)
    if EA.ndim == 2:
        i = min(EA.shape[0], nE)
        j = min(EA.shape[1], nA)
        EA_new[:i, :j] = EA[:i, :j]
    if AE.ndim == 2:
        i = min(AE.shape[0], nA)
        j = min(AE.shape[1], nE)
        AE_new[:i, :j] = AE[:i, :j]

    return EE_new, AA_new, EA_new, AE_new


def make_state(EE: Block, AA: Block, EA: Block, AE: Block) -> State:
    """Create an abstract state and normalize block shapes to a consistent (nE, nA)."""
    EE2, AA2, EA2, AE2 = _normalize_to_shapes(EE, AA, EA, AE)
    return State(EE=EE2, AA=AA2, EA=EA2, AE=AE2)


def get_blocks(state: State) -> tuple[Block, Block, Block, Block]:
    return state.EE, state.AA, state.EA, state.AE


# ------------- Closure -------------


def closure(state: State) -> State:
    EE, AA, EA, AE = get_blocks(state)
    EE = _copy(EE)
    AA = _copy(AA)
    EA = _copy(EA)
    AE = _copy(AE)

    nE = EE.shape[0] - 1
    nA = AA.shape[0] - 1

    # Normalize diagonals
    np.fill_diagonal(EE, 0.0)
    np.fill_diagonal(AA, 0.0)

    changed = True
    while changed:
        changed = False

        # Same-class closure
        # EE: min-plus Floyd–Warshall
        for k in range(nE + 1):
            for i in range(nE + 1):
                ik = EE[i, k]
                if math.isinf(ik):  # treat inf as neutral
                    continue
                for j in range(nE + 1):
                    v = ik + EE[k, j]
                    if v < EE[i, j]:
                        EE[i, j] = v
                        changed = True
        np.fill_diagonal(EE, 0.0)

        # AA: max-plus Floyd–Warshall
        for k in range(nA + 1):
            for i in range(nA + 1):
                ik = AA[i, k]
                if math.isinf(ik):  # treat +/-inf as neutral for propagation
                    continue
                for j in range(nA + 1):
                    v = ik + AA[k, j]
                    if v > AA[i, j]:
                        AA[i, j] = v
                        changed = True
        np.fill_diagonal(AA, 0.0)

        # Mixed closure
        # EA <= min(EE + EA, EA - AA)
        for i in range(nE):
            for j in range(nA):
                best = EA[i, j]
                for k in range(nE):
                    v = EE[i + 1, k + 1] + EA[k, j]
                    if v < best:
                        best = v
                for k in range(nA):
                    v = EA[i, k] - AA[j + 1, k + 1]
                    if v < best:
                        best = v
                if best < EA[i, j]:
                    EA[i, j] = best
                    changed = True

        # AE >= max(AA + AE, AE - EE)
        for i in range(nA):
            for j in range(nE):
                best = AE[i, j]
                for k in range(nA):
                    v = AA[i + 1, k + 1] + AE[k, j]
                    if v > best:
                        best = v
                for k in range(nE):
                    v = AE[i, k] - EE[j + 1, k + 1]
                    if v > best:
                        best = v
                if best > AE[i, j]:
                    AE[i, j] = best
                    changed = True

        # Unary propagation (both directions)
        # From unaries to mixed
        for i in range(nE):
            Uei = EE[i + 1, 0]
            for j in range(nA):
                Laj = AA[j + 1, 0]
                b = Uei - Laj
                if b < EA[i, j]:
                    EA[i, j] = b
                    changed = True

        for i in range(nA):
            Lai = AA[i + 1, 0]
            for j in range(nE):
                Uej = EE[j + 1, 0]
                lb = Lai - Uej
                if lb > AE[i, j]:
                    AE[i, j] = lb
                    changed = True

        # From mixed to unaries
        for i in range(nA):
            for j in range(nE):
                v = AE[i, j] - EE[0, j + 1]
                if v > AA[i + 1, 0]:
                    AA[i + 1, 0] = v
                    changed = True

        for i in range(nE):
            for j in range(nA):
                v = EA[i, j] - AA[0, j + 1]
                if v < EE[i + 1, 0]:
                    EE[i + 1, 0] = v
                    changed = True

    return State(EE=EE, AA=AA, EA=EA, AE=AE)


# ------------- Concretization helpers -------------


def satisfies(state: State, rho: dict[str, int]) -> bool:
    """
    Treat any ±inf entry as 'no constraint'. This makes intentionally 'loose'
    matrices in tests behave as intended.
    """
    EE, AA, EA, AE = get_blocks(state)
    nE = EE.shape[0] - 1
    nA = AA.shape[0] - 1

    def ve(i: int) -> int:
        return 0 if i == 0 else int(rho[f"e{i}"])

    def va(j: int) -> int:
        return 0 if j == 0 else int(rho[f"a{j}"])

    # EE: e_i - e_j <= EE[i,j]
    for i in range(nE + 1):
        ei = ve(i)
        for j in range(nE + 1):
            ub = EE[i, j]
            if math.isinf(ub):  # ±inf -> no constraint
                continue
            if ei - ve(j) > ub:
                return False

    # AA: a_i - a_j >= AA[i,j]
    for i in range(nA + 1):
        ai = va(i)
        for j in range(nA + 1):
            lb = AA[i, j]
            if math.isinf(lb):  # ±inf -> no constraint
                continue
            if ai - va(j) < lb:
                return False

    # EA: e_i - a_j <= EA[i-1,j-1]
    for i in range(1, nE + 1):
        ei = ve(i)
        for j in range(1, nA + 1):
            ub = EA[i - 1, j - 1]
            if math.isinf(ub):
                continue
            if ei - va(j) > ub:
                return False

    # AE: a_i - e_j >= AE[i-1,j-1]
    for i in range(1, nA + 1):
        ai = va(i)
        for j in range(1, nE + 1):
            lb = AE[i - 1, j - 1]
            if math.isinf(lb):
                continue
            if ai - ve(j) < lb:
                return False

    return True


def gamma_enumerate(
    state: State, e_vals: list[int], a_vals: list[int]
) -> list[dict[str, int]]:
    EE, AA, _, _ = get_blocks(state)
    nE = EE.shape[0] - 1
    nA = AA.shape[0] - 1

    envs: list[dict[str, int]] = []

    def gen_e(prefix: dict[str, int], i: int) -> None:
        if i > nE:
            gen_a(prefix, 1)
            return
        for v in e_vals:
            p = dict(prefix)
            p[f"e{i}"] = int(v)
            gen_e(p, i + 1)

    def gen_a(prefix: dict[str, int], j: int) -> None:
        if j > nA:
            if satisfies(state, prefix):
                envs.append(prefix)
            return
        for v in a_vals:
            p = dict(prefix)
            p[f"a{j}"] = int(v)
            gen_a(p, j + 1)

    gen_e({}, 1)
    return envs


# ------------- Abstractions (α) -------------


def alpha_outer_EE(P: list[dict[str, int]]) -> Block:
    """Left adjoint for EE: sup of differences (incl. 0-index)."""
    nE = _max_e_index(P)
    EE, _, _, _ = _zero_closed_blocks(nE, 0)
    EE[:, :] = INF
    np.fill_diagonal(EE, 0.0)

    if not P:
        return EE

    def ve(d: dict[str, int], idx: int) -> int:
        return 0 if idx == 0 else int(d[f"e{idx}"])

    for i in range(nE + 1):
        for j in range(nE + 1):
            supv = NINF
            for d in P:
                supv = max(supv, ve(d, i) - ve(d, j))
            EE[i, j] = float(supv)
    return EE


def alpha_outer_EA(S: list[dict[str, int]]) -> Block:
    """Left adjoint for EA: sup of e_i - a_j over S."""
    nE = _max_e_index(S)
    nA = _max_a_index(S)
    EA = np.full((nE, nA), INF, dtype=float)
    if not S:
        return EA

    for i in range(1, nE + 1):
        for j in range(1, nA + 1):
            supv = NINF
            for d in S:
                if f"e{i}" in d and f"a{j}" in d:
                    supv = max(supv, int(d[f"e{i}"]) - int(d[f"a{j}"]))
            EA[i - 1, j - 1] = float(supv)
    return EA


def alpha_inner_AA(Q: list[dict[str, int]]) -> Block:
    """
    Right adjoint for AA (inner): set unary lower bounds a_j - 0 >= L_j
    with L_j := min_Q a_j; others stay -inf; diagonal = 0.
    """
    nA = _max_a_index(Q)
    AA = np.full((nA + 1, nA + 1), NINF, dtype=float)
    np.fill_diagonal(AA, 0.0)
    if not Q:
        return AA

    for j in range(1, nA + 1):
        Lj = min(int(d[f"a{j}"]) for d in Q if f"a{j}" in d)
        AA[j, 0] = float(Lj)
    return AA


def alpha_inner_AE(S: list[dict[str, int]]) -> Block:
    """
    Right adjoint for AE (inner): AE[i,j] = inf_S (a_i - e_j), or -inf if unsampled.
    """
    nE = _max_e_index(S)
    nA = _max_a_index(S)
    AE = np.full((nA, nE), NINF, dtype=float)
    if not S:
        return AE

    for i in range(1, nA + 1):
        for j in range(1, nE + 1):
            infv: Any = INF
            for d in S:
                ai_key, ej_key = f"a{i}", f"e{j}"
                if ai_key in d and ej_key in d:
                    infv = min(infv, int(d[ai_key]) - int(d[ej_key]))
            if infv is not INF:
                AE[i - 1, j - 1] = float(infv)
    return AE


# ------------- Lattice operations -------------


def meet(s1: State, s2: State) -> State:
    EE = np.minimum(s1.EE, s2.EE)  # outer: min is tighter
    EA = np.minimum(s1.EA, s2.EA)
    AA = np.maximum(s1.AA, s2.AA)  # inner: max is tighter
    AE = np.maximum(s1.AE, s2.AE)
    return State(EE=EE, AA=AA, EA=EA, AE=AE)


def join(s1: State, s2: State) -> State:
    EE = np.maximum(s1.EE, s2.EE)  # outer: max is looser
    EA = np.maximum(s1.EA, s2.EA)
    AA = np.minimum(s1.AA, s2.AA)  # inner: min is looser
    AE = np.minimum(s1.AE, s2.AE)
    return State(EE=EE, AA=AA, EA=EA, AE=AE)


# ------------- Bottom / consistency -------------


def is_bottom(state: State) -> bool:
    """
    Inconsistency after closure if:
      - EE has a negative cycle: EE[ii] < 0
      - AA has a positive cycle: AA[ii] > 0
      - Cross-band contradiction: EA[i,j] < -AE[j,i]
    """
    st = closure(state)
    EE, AA, EA, AE = get_blocks(st)

    if np.any(np.diag(EE) < 0):
        return True
    if np.any(np.diag(AA) > 0):
        return True
    if EA.size and AE.size:
        if np.any(EA < (-AE.T)):
            return True
    return False


__all__ = [
    "INF",
    "NINF",
    "Block",
    "State",
    "make_state",
    "get_blocks",
    "closure",
    "satisfies",
    "gamma_enumerate",
    "alpha_outer_EE",
    "alpha_outer_EA",
    "alpha_inner_AA",
    "alpha_inner_AE",
    "meet",
    "join",
    "is_bottom",
]
