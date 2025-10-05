from dataclasses import replace

import numpy as np

from .lattice import State, INF, NINF, closure

# --------------------------
# Transformer API (1-based)
# --------------------------


def forget_e(state: State, i: int) -> State:
    """
    Forget may variable e_i.
    Effect (before closure):
      EE[i,*] = +INF (except EE[i,i]=0), EE[*,i] = +INF (except diag),
      EA[i-1,*] = +INF,
      AE[*,i-1] = -INF.
    Returns a *non-closed* state; call closure() after composing ops or
    rely on the assignment/guard helpers below which already close().
    """
    # implement exactly as in the spec's "Forgetting"
    EE, AA, EA, AE = state.EE.copy(), state.AA.copy(), state.EA.copy(), state.AE.copy()
    nE, nA = state.nE, state.nA
    ii = i  # because EE includes index 0; e_i maps to row/col i

    # EE row/col except diagonal
    for t in range(nE + 1):
        if t != ii:
            EE[ii, t] = INF
            EE[t, ii] = INF
    EE[ii, ii] = 0

    # Mixed
    if nA > 0:
        EA[ii - 1, :] = INF
        AE[:, ii - 1] = NINF

    return State(EE=EE, AA=AA, EA=EA, AE=AE)


def forget_a(state: State, j: int) -> State:
    EE, AA, EA, AE = state.EE.copy(), state.AA.copy(), state.EA.copy(), state.AE.copy()
    nE, nA = state.nE, state.nA
    jj = j

    for t in range(nA + 1):
        if t != jj:
            AA[jj, t] = NINF
            AA[t, jj] = NINF
    AA[jj, jj] = 0

    if nE > 0:
        EA[:, jj - 1] = INF
        AE[jj - 1, :] = NINF

    return State(EE=EE, AA=AA, EA=EA, AE=AE)


def assign_e_from_e(state: State, i: int, k: int, c: int) -> State:
    """
    e_i := e_k + c
    Spec:
      forget e_i; set EE[i,k]=c, EE[k,i]=-c;
      EA[i,*] = EA[k,*] + c; AE[* ,i] = AE[* ,k] - c; then close.
    """
    st = forget_e(state, i)
    EE, AA, EA, AE = st.EE.copy(), st.AA.copy(), st.EA.copy(), st.AE.copy()
    nE, nA = st.nE, st.nA

    EE[i, k] = np.min(EE[i, k], c)
    EE[k, i] = np.min(EE[k, i], -c)

    if nA > 0:
        EA[i - 1, :] = np.minimum(EA[i - 1, :], EA[k - 1, :] + c)
        AE[:, i - 1] = np.maximum(AE[:, i - 1], AE[:, k - 1] - c)

    return closure(State(EE=EE, AA=AA, EA=EA, AE=AE))


def assign_a_from_a(state: State, j: int, ell: int, c: int) -> State:
    st = forget_a(state, j)
    EE, AA, EA, AE = st.EE.copy(), st.AA.copy(), st.EA.copy(), st.AE.copy()
    nE, nA = st.nE, st.nA

    AA[j, ell] = max(AA[j, ell], c)
    AA[ell, j] = max(AA[ell, j], -c)

    if nE > 0:
        AE[j - 1, :] = np.maximum(AE[j - 1, :], AE[ell - 1, :] + c)
        EA[:, j - 1] = np.minimum(EA[:, j - 1], EA[:, ell - 1] - c)

    return closure(State(EE=EE, AA=AA, EA=EA, AE=AE))


def assign_e_from_e(state: State, i: int, k: int, c: int) -> State:
    st = forget_e(state, i)
    EE, AA, EA, AE = st.EE.copy(), st.AA.copy(), st.EA.copy(), st.AE.copy()
    nE, nA = st.nE, st.nA

    # scalar tighten with builtins
    EE[i, k] = min(EE[i, k], c)
    EE[k, i] = min(EE[k, i], -c)

    if nA > 0:
        EA[i - 1, :] = np.minimum(EA[i - 1, :], EA[k - 1, :] + c)
        AE[:, i - 1] = np.maximum(AE[:, i - 1], AE[:, k - 1] - c)

    return closure(State(EE=EE, AA=AA, EA=EA, AE=AE))


def assign_e_from_a(state: State, i: int, ell: int, c: int) -> State:
    """
    e_i := a_ell + c
    Spec: forget e_i; set EA[i,ell]=c and AE[ell,i]=-c; then close.
    """
    st = forget_e(state, i)
    EA, AE = st.EA.copy(), st.AE.copy()
    nE, nA = st.nE, st.nA

    EA[i - 1, ell - 1] = min(EA[i - 1, ell - 1], c)
    AE[ell - 1, i - 1] = max(AE[ell - 1, i - 1], -c)

    return closure(replace(st, EA=EA, AE=AE))


def assign_a_from_e(state: State, j: int, k: int, c: int) -> State:
    st = forget_a(state, j)
    EA, AE = st.EA.copy(), st.AE.copy()

    AE[j - 1, k - 1] = max(AE[j - 1, k - 1], c)
    EA[k - 1, j - 1] = min(EA[k - 1, j - 1], -c)  # <-- was np.min(...)
    return closure(replace(st, EA=EA, AE=AE))


def assign_e_interval(state: State, i: int, L: int, U: int) -> State:
    # forget then OVERWRITE unaries (no intersect with pre-forget)
    st = forget_e(state, i)
    EE = st.EE.copy()
    EE[i, 0] = U
    EE[0, i] = -L
    return closure(replace(st, EE=EE))


def assign_a_interval(state: State, j: int, L: int, U: int) -> State:
    # forget then OVERWRITE unaries
    st = forget_a(state, j)
    AA = st.AA.copy()
    AA[j, 0] = L
    AA[0, j] = -U
    return closure(replace(st, AA=AA))


def guard_ee_le(state: State, i: int, k: int, c: int) -> State:
    EE = state.EE.copy()
    EE[i, k] = min(EE[i, k], c)
    return closure(replace(state, EE=EE))


def guard_aa_ge(state: State, j: int, ell: int, c: int) -> State:
    AA = state.AA.copy()
    AA[j, ell] = max(AA[j, ell], c)
    return closure(replace(state, AA=AA))


def guard_ea_le(state: State, i: int, j: int, c: int) -> State:
    EA = state.EA.copy()
    EA[i - 1, j - 1] = min(EA[i - 1, j - 1], c)
    return closure(replace(state, EA=EA))


def guard_ae_ge(state: State, j: int, i: int, c: int) -> State:
    AE = state.AE.copy()
    AE[j - 1, i - 1] = max(AE[j - 1, i - 1], c)
    return closure(replace(state, AE=AE))


__all__ = [
    "forget_e",
    "forget_a",
    "assign_e_from_e",
    "assign_a_from_a",
    "assign_e_from_a",
    "assign_a_from_e",
    "assign_e_interval",
    "assign_a_interval",
    "guard_ee_le",
    "guard_aa_ge",
    "guard_ea_le",
    "guard_ae_ge",
]
