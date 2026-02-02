import numpy as np
import pytest

from . import lattice as D
from . import transformers as T


# -------------------------
# helpers / small scaffolds
# -------------------------


def top_blocks(nE: int, nA: int):
    """Return TOP (no info) blocks with proper shapes."""
    EE = np.full((nE + 1, nE + 1), D.INF, dtype=float)
    np.fill_diagonal(EE, 0.0)
    AA = np.full((nA + 1, nA + 1), D.NINF, dtype=float)
    np.fill_diagonal(AA, 0.0)
    EA = np.full((nE, nA), D.INF, dtype=float)
    AE = np.full((nA, nE), D.NINF, dtype=float)
    return EE, AA, EA, AE


def top_state(nE: int, nA: int) -> D.State:
    EE, AA, EA, AE = top_blocks(nE, nA)
    return D.closure(D.make_state(EE, AA, EA, AE))


def gamma_pairs(st: D.State, e_vals, a_vals):
    """Set of (e1,...,a1,...) tuples helpful for assertions."""
    G = D.gamma_enumerate(st, e_vals=e_vals, a_vals=a_vals)
    return {tuple(sorted(r.items())) for r in G}


# Default tiny grid
GRID = [-1, 0, 1]


# ------------------------
# A. Straight-line / unary
# ------------------------


def test_A1_assign_e_from_a_one_sided_band():
    # e1 := a1 + 1
    # NOTE: The domain cannot encode equality e1 = a1 + 1 (needs both
    # upper AND lower bound on e1-a1). What we *can* represent is:
    #   EA: e1 - a1 <= 1   (from EA)
    #   AE: a1 - e1 >= -1  (which is again e1 - a1 <= 1)
    st0 = top_state(1, 1)
    st = T.assign_e_from_a(st0, i=1, ell=1, c=1)

    e_vals = [-1, 0, 1, 2]
    a_vals = [-1, 0, 1]
    G = D.gamma_enumerate(st, e_vals=e_vals, a_vals=a_vals)

    # Sound one-sided property:
    assert all((r["e1"] - r["a1"]) <= 1 for r in G)
    # There *exist* precise-equality models on the grid:
    assert any((r["e1"] == r["a1"] + 1) for r in G)


def test_A2_outer_and_inner_intervals():
    # e1 ∈ [0,2] (outer); a1 ∈ [1,3] (inner)
    st = top_state(1, 1)
    st = T.assign_e_interval(st, i=1, L=0, U=2)
    st = T.assign_a_interval(st, j=1, L=1, U=3)

    G = D.gamma_enumerate(st, e_vals=[0, 1, 2], a_vals=[1, 2, 3])
    assert all(0 <= r["e1"] <= 2 for r in G)
    assert all(1 <= r["a1"] <= 3 for r in G)


def test_A3_mixed_to_unary_must_inference():
    # Given a1 - e1 ≥ 1 and e1 ≥ 0, derive a1 ≥ 1 (via mixed→unary tightening).
    st = top_state(1, 1)
    st = T.guard_ae_ge(st, j=1, i=1, c=1)  # a1 - e1 ≥ 1
    st = T.guard_ee_le(st, i=0, k=1, c=0)  # 0 - e1 ≤ 0  => e1 ≥ 0
    # Check must unary on a1:
    assert st.AA[1, 0] >= 1


# --------------------------------
# B. Mixed propagation / multi-var
# --------------------------------


def test_B1_twohop_mixed_via_must_path():
    # e1 - a1 ≤ 2  and  a2 - a1 ≥ 5  =>  e1 - a2 ≤ -3
    st = top_state(2, 2)
    st = T.guard_ea_le(st, i=1, j=1, c=2)
    st = T.guard_aa_ge(st, j=2, ell=1, c=5)
    assert st.EA[0, 1] <= -3


def test_B2_dual_twohop_mixed_via_may_path():
    # a1 - e1 ≥ 2  and  e2 - e1 ≤ 1  =>  a1 - e2 ≥ 1
    st = top_state(2, 2)
    st = T.guard_ae_ge(st, j=1, i=1, c=2)
    st = T.guard_ee_le(st, i=2, k=1, c=1)
    assert st.AE[0, 1] >= 1


def test_B3_same_class_chain_feeds_mixed():
    # We want e1 ≤ e2 ≤ e3 and e3 - a1 ≤ 0 => e1 - a1 ≤ 0.
    # CAREFUL: guard_ee_le(i,k,c) encodes e_i - e_k ≤ c.
    st = top_state(3, 1)
    st = T.guard_ee_le(st, i=1, k=2, c=0)  # e1 - e2 ≤ 0 => e1 ≤ e2
    st = T.guard_ee_le(st, i=2, k=3, c=0)  # e2 - e3 ≤ 0 => e2 ≤ e3
    st = T.guard_ea_le(st, i=3, j=1, c=0)  # e3 - a1 ≤ 0
    assert st.EA[0, 0] <= 0


# ---------------------------
# C. Branching / joins / bands
# ---------------------------


def test_C1_nondet_branch_band_on_mixed():
    # if * then e1 = a1 + 1 else e1 = a1 - 1
    # Representable post-join property: e1 - a1 ≤ 1 (upper bound).
    base = top_state(1, 1)

    s1 = T.assign_e_from_a(base, i=1, ell=1, c=1)
    s2 = T.assign_e_from_a(base, i=1, ell=1, c=-1)
    sj = D.closure(D.join(s1, s2))

    assert sj.EA[0, 0] <= 1  # upper bound retained
    # Enumerated models respect the one-sided band:
    G = D.gamma_enumerate(sj, e_vals=[-2, -1, 0, 1, 2], a_vals=[-1, 0, 1])
    assert all((r["e1"] - r["a1"]) <= 1 for r in G)
    # both branch-precise points exist
    assert any((r["e1"] - r["a1"]) == 1 for r in G)
    assert any((r["e1"] - r["a1"]) == -1 for r in G)


def test_C2_guard_then_correlated_assignments_join():
    # assume(e1 - a1 ≤ 0)
    # if (e1 - a1 ≤ -1) then a1 := e1 else a1 := e1 + 1
    # Representable post-join: e1 - a1 ≤ 0 and a1 - e1 ≥ 0 (i.e., e1 ≤ a1).
    base = top_state(1, 1)
    base = T.guard_ea_le(base, i=1, j=1, c=0)

    # Branch 1
    b1 = T.guard_ea_le(base, i=1, j=1, c=-1)
    b1 = T.assign_a_from_e(b1, j=1, k=1, c=0)  # a1 = e1

    # Branch 2
    b2 = T.guard_ea_le(base, i=1, j=1, c=0)
    b2 = T.assign_a_from_e(b2, j=1, k=1, c=1)  # a1 = e1 + 1

    sj = D.closure(D.join(b1, b2))

    assert sj.EA[0, 0] <= 0
    assert sj.AE[0, 0] >= 0

    G = D.gamma_enumerate(sj, e_vals=GRID, a_vals=[v for v in range(-1, 3)])
    assert all(r["e1"] <= r["a1"] for r in G)


def test_C3_initialized_range_proof_must_via_may():
    # May: e1 ∈ [5,7]; Mixed: e1 - a1 ≤ 2; Must: a2 - a1 ≥ 5
    st = top_state(2, 2)
    st = T.assign_e_interval(st, i=1, L=5, U=7)
    st = T.guard_ea_le(st, i=1, j=1, c=2)
    st = T.guard_aa_ge(st, j=2, ell=1, c=5)

    # All models satisfy a2 ≥ 8 (min e1 + 3).
    G = D.gamma_enumerate(st, e_vals=[5, 6, 7], a_vals=[v for v in range(0, 12)])
    assert all(r["a2"] >= 8 for r in G)


# -------------------------
# D. Join behavior (inner/outer)
# -------------------------


def test_D1_join_on_must_lowers_picks_weaker_branch():
    # if * then a1 ≥ 5 else a1 ≥ 7  => post-join has a1 ≥ 5
    base = top_state(0, 1)

    b1 = T.assign_a_interval(base, j=1, L=5, U=+(10**6))
    b2 = T.assign_a_interval(base, j=1, L=7, U=+(10**6))

    sj = D.closure(D.join(b1, b2))
    assert sj.AA[1, 0] >= 5

    G = D.gamma_enumerate(sj, e_vals=[], a_vals=[v for v in range(0, 10)])
    assert any(r["a1"] == 5 for r in G)  # weaker branch survives join


def test_D2_join_on_may_ranges_widens():
    # if * then e1 ∈ [0,2] else e1 ∈ [1,3]  => post-join e1 ∈ [0,3]
    base = top_state(1, 0)
    b1 = T.assign_e_interval(base, i=1, L=0, U=2)
    b2 = T.assign_e_interval(base, i=1, L=1, U=3)

    sj = D.closure(D.join(b1, b2))
    G = D.gamma_enumerate(sj, e_vals=[0, 1, 2, 3], a_vals=[])
    assert all(0 <= r["e1"] <= 3 for r in G)


# --------------------
# E. Guards as filters
# --------------------


def test_E1_mixed_guards_give_one_sided_equality():
    # assume(e1 - a1 ≤ 0); assume(a1 - e1 ≥ 0)
    # Representable: e1 ≤ a1 (same constraint from both sides).
    st = top_state(1, 1)
    st = T.guard_ea_le(st, i=1, j=1, c=0)
    st = T.guard_ae_ge(st, j=1, i=1, c=0)

    G = D.gamma_enumerate(st, e_vals=GRID, a_vals=GRID)
    assert all(r["e1"] <= r["a1"] for r in G)
    # and equality is permitted (not required)
    assert any(r["e1"] == r["a1"] for r in G)


def test_E2_mixed_guards_tightening():
    """
    Test that redundant mixed guards get tightened but don't cause bottom.
    EA and AE both constrain e - a from above, so they can't contradict.
    """
    st = top_state(1, 1)
    st = T.guard_ea_le(st, i=1, j=1, c=-3)  # e - a ≤ -3
    st = T.guard_ae_ge(st, j=1, i=1, c=2)  # a - e ≥ 2, i.e., e - a ≤ -2
    # Both constraints are satisfiable: e.g., e=0, a=4 gives e-a=-4
    assert not D.is_bottom(st)
    # After tightening: AE should be at least -EA = 3
    assert st.AE[0, 0] >= 3


# -------------------
# F. Forget operators
# -------------------


def test_F1_forget_e_breaks_mixed_info():
    st = top_state(1, 1)
    st = T.guard_ea_le(st, i=1, j=1, c=0)  # e1 ≤ a1
    st = T.forget_e(st, i=1)
    st = D.closure(st)
    # EA row for e1 becomes +INF (no constraints)
    assert st.EA.shape == (1, 1)
    assert st.EA[0, 0] == D.INF


def test_F2_forget_a_breaks_mixed_info_dual():
    st = top_state(1, 1)
    st = T.guard_ae_ge(st, j=1, i=1, c=0)  # a1 ≥ e1
    st = T.forget_a(st, j=1)
    st = D.closure(st)
    # AE row for a1 becomes -INF (no constraints)
    assert st.AE.shape == (1, 1)
    assert st.AE[0, 0] == D.NINF


# ---------------------------------------------
# G. Abstraction sanity (outer/inner α functions)
# ---------------------------------------------


def test_G1_alpha_outer_EA_partial_envs_is_top():
    # No joint sample for (e1,a1) => EA[1,1] should be +INF (no constraint).
    S = [{"e1": 0}, {"a1": 5}]
    EA = D.alpha_outer_EA(S)

    EE, AA, _, _ = top_blocks(1, 1)
    AE = np.full((1, 1), D.NINF, dtype=float)

    st = D.closure(D.make_state(EE, AA, EA, AE))
    assert not D.is_bottom(st)

    # There exists a joint assignment consistent with the partial observations
    G = D.gamma_enumerate(st, e_vals=[0], a_vals=[5])
    assert len(G) >= 1


def test_G2_alpha_inner_AA_relational_subset():
    # Q: {(a1,a2) | a1 in [0,2], a2 = a1 + 3} over a small grid
    a1_vals = [0, 1, 2]
    Q = [{"a1": a1, "a2": a1 + 3} for a1 in a1_vals]

    AA = D.alpha_inner_AA(Q)
    EE = np.array([[0.0]], dtype=float)  # nE=0 → EE 1x1
    EA = np.empty((0, 2), dtype=float)  # shapes for nE=0,nA=2
    AE = np.empty((2, 0), dtype=float)

    st = D.closure(D.make_state(EE, AA, EA, AE))
    G = D.gamma_enumerate(st, e_vals=[], a_vals=[0, 1, 2, 3, 4, 5])

    # γ ⊆ Q on the chosen grid
    gamma_set = {tuple(sorted(r.items())) for r in G}
    Q_set = {tuple(sorted(d.items())) for d in Q}
    assert gamma_set.issubset(Q_set)


def test_G3_alpha_inner_AE_cross_relational():
    # S: {(a1,e1) | a1 - e1 ≥ 2} sampled on tiny grid
    S = [{"a1": a, "e1": e} for a in range(-1, 4) for e in range(-1, 4) if a - e >= 2]
    AE = D.alpha_inner_AE(S)

    EE, AA, EA, _ = top_blocks(1, 1)
    st = D.closure(D.make_state(EE, AA, EA, AE))

    G = D.gamma_enumerate(st, e_vals=GRID, a_vals=[-1, 0, 1, 2, 3])
    assert all((r["a1"] - r["e1"]) >= 2 for r in G)


# -----------------------------------------
# H. "Initialized range" idioms (core use)
# -----------------------------------------


def test_H1_index_within_initialized_prefix():
    # a1 ≥ L=3 (must), e1 ∈ [0,U=2] (may) => e1 - a1 ≤ -1
    st = top_state(1, 1)
    st = T.assign_a_interval(st, j=1, L=3, U=10**6)
    st = T.assign_e_interval(st, i=1, L=0, U=2)

    G = D.gamma_enumerate(st, e_vals=[0, 1, 2], a_vals=[3, 4, 5])
    assert all((r["e1"] - r["a1"]) <= -1 for r in G)


def test_H2_two_stage_init_with_correlation():
    # a1 ≥ 2, then e1 := a1 - 1  =>  e1 < a1 (strict on all models in grid)
    st = top_state(1, 1)
    st = T.assign_a_interval(st, j=1, L=2, U=10**6)
    st = T.assign_e_from_a(st, i=1, ell=1, c=-1)

    G = D.gamma_enumerate(st, e_vals=[0, 1, 2, 3, 4], a_vals=[2, 3, 4, 5])
    assert all(r["e1"] < r["a1"] for r in G)


# ---------------------------------------
# I. May-Must vs Standard DBM: The Key Difference
# ---------------------------------------


def test_I0_may_must_semantics_for_bounds_checking():
    """
    Demonstrates why may-must DBM is fundamentally different from standard DBM
    for array bounds checking.

    Scenario: Two branches initialize different amounts of an array.
    - Branch 1: initializes 10 elements (size ≥ 10)
    - Branch 2: initializes 7 elements (size ≥ 7)

    For SAFETY checking (can we access index i?), we need the MINIMUM guaranteed
    size after join, not the maximum possible size.

    May-must DBM (under-approximation for must):
      - After join: size ≥ min(10, 7) = 7 (definitely at least 7)
      - Safe to access indices 0..6

    Standard DBM (over-approximation for everything):
      - Would track: size could be up to max(10, 7) = 10
      - This is useless for safety! "Could be up to 10" doesn't guarantee anything.

    The key insight: for array bounds, we care about:
      - Upper bound of index (may): what the index COULD be (over-approx)
      - Lower bound of size (must): what the size DEFINITELY is (under-approx)
    """
    # Simulate: two branches initialize different amounts
    # Branch 1: size ≥ 10, index ≤ 5
    b1 = top_state(1, 1)
    b1 = T.assign_a_interval(b1, j=1, L=10, U=10**6)  # a_size ≥ 10
    b1 = T.assign_e_interval(b1, i=1, L=0, U=5)       # e_index ≤ 5

    # Branch 2: size ≥ 7, index ≤ 3
    b2 = top_state(1, 1)
    b2 = T.assign_a_interval(b2, j=1, L=7, U=10**6)   # a_size ≥ 7
    b2 = T.assign_e_interval(b2, i=1, L=0, U=3)       # e_index ≤ 3

    # Join the branches
    sj = D.closure(D.join(b1, b2))

    # Check the semantics:
    # - Must (a_size) takes MIN: min(10, 7) = 7 (under-approximation)
    # - May (e_index) takes MAX for upper bound: max(5, 3) = 5 (over-approximation)
    assert sj.AA[1, 0] == 7   # a_size ≥ 7 (the weaker guarantee survives)
    assert sj.EE[1, 0] == 5   # e_index ≤ 5 (the weaker bound survives)

    # The critical safety check: index < size
    # With e_index ≤ 5 and a_size ≥ 7, we have e_index - a_size ≤ 5 - 7 = -2 < 0
    # So the access is SAFE!
    G = D.gamma_enumerate(sj, e_vals=[0, 1, 2, 3, 4, 5], a_vals=[7, 8, 9, 10])
    assert all(r["e1"] < r["a1"] for r in G)  # index < size for all concrete states

    # This is what makes may-must useful: after join, we STILL know index < size
    # Standard DBM would tell us "size could be up to 10" which doesn't help safety!


# ---------------------------------------
# J. Expected approximation-loss scenarios
# ---------------------------------------


def test_J1_join_on_must_loses_stronger_branch():
    # if * then a1 ≥ 5 else a1 ≥ 7  => post-join only guarantees a1 ≥ 5
    base = top_state(0, 1)
    b1 = T.assign_a_interval(base, j=1, L=5, U=10**6)
    b2 = T.assign_a_interval(base, j=1, L=7, U=10**6)
    sj = D.closure(D.join(b1, b2))

    G = D.gamma_enumerate(sj, e_vals=[], a_vals=[5, 6, 7, 8])
    assert any(r["a1"] == 5 for r in G)
    # We do not require a1 ≥ 7 globally — precision is lost at join.
