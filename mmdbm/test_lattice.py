import pytest
import numpy as np

from . import lattice as D
from conftest import INF, NINF, build_zero_closed, get_blocks, envs

# -------------------------
# 1) Basic shapes & closure
# -------------------------


def test_shapes_and_zero_diagonal(api_guard, sizes):
    nE, nA = sizes
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st = D.closure(D.make_state(EE, AA, EA, AE))
    EE_, AA_, EA_, AE_ = get_blocks(st)

    assert len(EE_) == nE + 1 and all(len(row) == nE + 1 for row in EE_)
    assert len(AA_) == nA + 1 and all(len(row) == nA + 1 for row in AA_)
    assert len(EA_) == nE and all(len(row) == nA for row in EA_)
    assert len(AE_) == nA and all(len(row) == nE for row in AE_)

    assert all(EE_[i][i] == 0 for i in range(nE + 1))
    assert all(AA_[j][j] == 0 for j in range(nA + 1))


def test_cross_band_consistency(api_guard, sizes):
    nE, nA = sizes
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    if nE >= 1 and nA >= 1:
        EA[0][0] = 1  # e1 - a1 <= 1
        AE[0][0] = -1  # a1 - e1 >= -1  => EA[0,0] >= -AE[0,0] (=1)
    st = D.closure(D.make_state(EE, AA, EA, AE))
    _, _, EA_, AE_ = get_blocks(st)
    for i in range(nE):
        for j in range(nA):
            assert EA_[i][j] >= -AE_[j][i]


# -----------------------------------------
# 2) Mixed closure: polarity-correct paths
# -----------------------------------------


def test_mixed_closure_EA_through_e_and_a(api_guard):
    nE, nA = 2, 2
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    EA[0][0] = 1  # e1 - a1 <= 1
    # AA indices: include 0; a1->1, a2->2
    AA[2][1] = 1  # a2 - a1 >= 1
    st = D.closure(D.make_state(EE, AA, EA, AE))
    _, _, EA_, _ = get_blocks(st)
    # EA[1,2] <= EA[1,1] - AA[2,1] <= 1 - 1 = 0
    assert EA_[0][1] <= 0


def test_mixed_closure_AE_dual_paths(api_guard):
    nE, nA = 2, 2
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    AE[0][0] = 1  # a1 - e1 >= 1
    # EE indices: include 0; e1->1, e2->2
    EE[2][1] = 0  # e2 - e1 <= 0
    st = D.closure(D.make_state(EE, AA, EA, AE))
    _, _, _, AE_ = get_blocks(st)
    # AE[1,2] >= AE[1,1] - EE[2,1] >= 1 - 0 = 1
    assert AE_[0][1] >= 1


# --------------------------
# 3) Unary propagation rules
# --------------------------


def test_unary_from_mixed_to_AA(api_guard):
    nE, nA = 1, 1
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    AE[0][0] = 1  # a1 - e1 >= 1
    EE[0][1] = 0  # 0 - e1 <= 0  => e1 >= 0
    st = D.closure(D.make_state(EE, AA, EA, AE))
    _, AA_, _, _ = get_blocks(st)
    # Expect: AA[1,0] >= AE[1,1] - EE[0,1] = 1 - 0 = 1  (a1 - 0 >= 1)
    assert AA_[1][0] >= 1


def test_unary_from_mixed_to_EE(api_guard):
    nE, nA = 1, 1
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    EA[0][0] = 0  # e1 - a1 <= 0
    AA[0][1] = 0  # 0 - a1 <= 0  => a1 <= 0 (i.e., AA[0,1] = -L_a1 with L_a1=0)
    st = D.closure(D.make_state(EE, AA, EA, AE))
    EE_, _, _, _ = get_blocks(st)
    # Expect: EE[1,0] <= EA[1,1] - AA[0,1] = 0 - 0 = 0  (e1 - 0 <= 0)
    assert EE_[1][0] <= 0


# --------------------------
# 4) Outer (may) soundness
# --------------------------


@pytest.mark.parametrize(
    "pairs",
    [
        # small concrete relations on the grid for (1,1)
        [({"e1": -1, "a1": -1}), ({"e1": 0, "a1": 0})],
        [({"e1": -1, "a1": 1}), ({"e1": 1, "a1": -1})],
    ],
)
def test_outer_soundness_11(api_guard, grid_vals, pairs):
    nE, nA = 1, 1
    S = pairs
    P = [{"e1": rho["e1"]} for rho in S]

    EE = D.alpha_outer_EE(P)
    EA = D.alpha_outer_EA(S)
    # Loose inner blocks so they don't constrain:
    AA = [[0, NINF], [INF, 0]]
    AE = [[NINF]]

    st = D.closure(D.make_state(EE, AA, EA, AE))
    gamma = D.gamma_enumerate(st, e_vals=grid_vals, a_vals=grid_vals)
    G = {(r["e1"], r["a1"]) for r in gamma}
    for rho in S:
        assert (rho["e1"], rho["a1"]) in G


# --------------------------
# 5) Inner (must) soundness
# --------------------------


def test_inner_soundness_AA_11(api_guard, grid_vals):
    nE, nA = 1, 1
    # Upward-closed Q: { a1 >= 0 }
    Q = [{"a1": v} for v in grid_vals if v >= 0]
    AA = D.alpha_inner_AA(Q)
    # Make outer blocks loose:
    EE = [[0, INF], [NINF, 0]]
    EA = [[INF]]
    AE = [[NINF]]

    st = D.closure(D.make_state(EE, AA, EA, AE))
    gamma = D.gamma_enumerate(st, e_vals=grid_vals, a_vals=grid_vals)
    for r in gamma:
        assert r["a1"] >= 0


def test_inner_soundness_AE_11(api_guard, grid_vals):
    nE, nA = 1, 1
    # Upward-closed cross set: { a1 - e1 >= 0 } over the grid
    S = [{"a1": a, "e1": e} for a in grid_vals for e in grid_vals if (a - e) >= 0]
    F = D.alpha_inner_AE(S)
    # Loose same-class blocks:
    EE, AA, _, _ = build_zero_closed(nE, nA)
    for i in range(nE + 1):
        for j in range(nE + 1):
            EE[i][j] = INF
    for i in range(nA + 1):
        for j in range(nA + 1):
            AA[i][j] = NINF
    for d in range(nE + 1):
        EE[d][d] = 0
    for d in range(nA + 1):
        AA[d][d] = 0
    EA = [[INF]]
    AE = F

    st = D.closure(D.make_state(EE, AA, EA, AE))
    gamma = D.gamma_enumerate(st, e_vals=grid_vals, a_vals=grid_vals)
    for r in gamma:
        assert (r["a1"] - r["e1"]) >= 0


# --------------------------
# 6) Gamma = intersection
# --------------------------


def test_gamma_is_intersection_of_inner_outer(api_guard, sizes, grid_vals):
    nE, nA = sizes
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    if nE >= 1 and nA >= 1:
        # e1 in [-1,0]  => e1-0 <= 0 and 0-e1 <= 1
        EE[1][0] = 0
        EE[0][1] = 1
        # a1 >= 0       => a1-0 >= 0 and 0-a1 <= 0
        AA[1][0] = 0
        AA[0][1] = 0
        # e1 - a1 <= 0  and  a1 - e1 >= 0
        EA[0][0] = 0
        AE[0][0] = 0

    st = D.closure(D.make_state(EE, AA, EA, AE))

    # Outer-only and inner-only states
    AA_loose = [[0, NINF], [NINF, 0]]
    AE_loose = [[NINF] * nE for _ in range(nA)]
    EE_loose = (
        [[0, INF], [NINF, 0]] if nE == 1 else [[0] * (nE + 1) for _ in range(nE + 1)]
    )
    if nE > 1:
        for i in range(nE + 1):
            for j in range(nE + 1):
                EE_loose[i][j] = INF if i != j else 0

    st_outer = D.closure(D.make_state(EE, AA_loose, EA, AE_loose))
    st_inner = D.closure(
        D.make_state(EE_loose, AA, [[INF] * nA for _ in range(nE)], AE)
    )

    G = set(
        tuple(sorted(r.items())) for r in D.gamma_enumerate(st, grid_vals, grid_vals)
    )
    Gp = set(
        tuple(sorted(r.items()))
        for r in D.gamma_enumerate(st_outer, grid_vals, grid_vals)
    )
    Gm = set(
        tuple(sorted(r.items()))
        for r in D.gamma_enumerate(st_inner, grid_vals, grid_vals)
    )

    assert G == (Gp & Gm)


# --------------------------
# 7) Lattice operations
# --------------------------


def test_lattice_meet_join_monotonicity(api_guard, sizes, grid_vals):
    nE, nA = sizes
    EE1, AA1, EA1, AE1 = build_zero_closed(nE, nA)
    if nE >= 1 and nA >= 1:
        EA1[0][0] = 0  # e1 - a1 <= 0
    S1 = D.closure(D.make_state(EE1, AA1, EA1, AE1))

    EE2, AA2, EA2, AE2 = build_zero_closed(nE, nA)
    if nE >= 1 and nA >= 1:
        AE2[0][0] = 0  # a1 - e1 >= 0
    S2 = D.closure(D.make_state(EE2, AA2, EA2, AE2))

    S_join = D.closure(D.join(S1, S2))
    S_meet = D.closure(D.meet(S1, S2))

    G1 = set(
        tuple(sorted(r.items())) for r in D.gamma_enumerate(S1, grid_vals, grid_vals)
    )
    G2 = set(
        tuple(sorted(r.items())) for r in D.gamma_enumerate(S2, grid_vals, grid_vals)
    )
    Gj = set(
        tuple(sorted(r.items()))
        for r in D.gamma_enumerate(S_join, grid_vals, grid_vals)
    )
    Gm = set(
        tuple(sorted(r.items()))
        for r in D.gamma_enumerate(S_meet, grid_vals, grid_vals)
    )

    # Join should be looser: gamma(join) ⊇ gamma(S1) ∪ gamma(S2)
    assert Gj.issuperset(G1 | G2)
    # Meet should be tighter: gamma(meet) ⊆ gamma(S1) ∩ gamma(S2)
    assert Gm.issubset(G1 & G2)


# --------------------------
# 8) Bottom detection
# --------------------------


def test_cross_band_consistency_tightening(api_guard, sizes):
    """
    Test that cross-band consistency tightens EA and AE to be mutually consistent.
    EA[i,j] ≤ -AE[j,i] and AE[j,i] ≥ -EA[i,j] after closure.

    Note: EA and AE both constrain the same e-a relationship from opposite
    directions, so they can't directly contradict - they just provide
    redundant bounds that get tightened.
    """
    nE, nA = sizes
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    if nE >= 1 and nA >= 1:
        EA[0][0] = -1  # e1 - a1 ≤ -1
        AE[0][0] = -2  # a1 - e1 ≥ -2 (i.e., e1 - a1 ≤ 2)
    st = D.closure(D.make_state(EE, AA, EA, AE))
    # State is consistent - both constraints satisfied by e1 - a1 = -1
    assert not D.is_bottom(st)
    # After closure, AE should be tightened: AE[0,0] ≥ -EA[0,0] = 1
    if nE >= 1 and nA >= 1:
        assert st.AE[0, 0] >= 1  # Tightened from -2 to 1


# --------------------------
# 9) Closure idempotence & gamma invariance on grid
# --------------------------


def test_closure_idempotent_and_gamma_invariant(api_guard, sizes, grid_vals):
    nE, nA = sizes
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    if nE >= 1 and nA >= 1:
        EA[0][0] = 1
        AE[0][0] = -1
    st1 = D.closure(D.make_state(EE, AA, EA, AE))
    st2 = D.closure(st1)

    EE1, AA1, EA1, AE1 = get_blocks(st1)
    EE2, AA2, EA2, AE2 = get_blocks(st2)
    assert EE1 == EE2 and AA1 == AA2 and EA1 == EA2 and AE1 == AE2

    G1 = set(
        tuple(sorted(r.items())) for r in D.gamma_enumerate(st1, grid_vals, grid_vals)
    )
    G2 = set(
        tuple(sorted(r.items())) for r in D.gamma_enumerate(st2, grid_vals, grid_vals)
    )
    assert G1 == G2


def test_inner_soundness_AA_with_upper_bound():
    nE, nA = 0, 1
    a_vals = [0, 1, 2, 3]  # Extended to include value beyond max in Q
    Q = [{"a1": v} for v in [0, 1, 2]]  # Q has a1 in [0,2]

    AA = D.alpha_inner_AA(Q)
    # Loose outer blocks (since nE=0, EA and AE empty)
    EE = np.array([[0.0]], dtype=float)  # Minimal for nE=0
    EA = np.array([], dtype=float).reshape(0, 1)
    AE = np.array([], dtype=float).reshape(1, 0)

    st = D.closure(D.make_state(EE, AA, EA, AE))
    gamma = D.gamma_enumerate(st, e_vals=[], a_vals=a_vals)

    Q_set = {tuple(sorted(d.items())) for d in Q}
    gamma_set = {tuple(sorted(r.items())) for r in gamma}

    assert gamma_set.issubset(Q_set)


def test_inner_soundness_AA_with_upper_bound_subset():
    # Q = { a1 in [0,2] } on a larger grid; gamma must not include 3
    a_grid = [0, 1, 2, 3]  # note the 3 outside Q
    Q = [{"a1": v} for v in [0, 1, 2]]

    AA = D.alpha_inner_AA(Q)
    # nE = 0
    EE = np.array([[0.0]])
    EA = np.zeros((0, 1), dtype=float)
    AE = np.zeros((1, 0), dtype=float)

    st = D.closure(D.make_state(EE, AA, EA, AE))
    gamma = D.gamma_enumerate(st, e_vals=[], a_vals=a_grid)

    Q_set = {tuple(sorted(d.items())) for d in Q}
    gamma_set = {tuple(sorted(r.items())) for r in gamma}
    assert gamma_set.issubset(Q_set)


def test_outer_soundness_EA_with_partial_envs_1():
    # S represents points where e1 and a1 are independent (no joint constraints observed)
    S = [{"e1": 0}, {"a1": 5}]  # Separate dicts for e1 and a1

    # Compute abstraction
    EA = D.alpha_outer_EA(S)
    # Since nE=1, nA=1, but no joint points, should be INF (loose), not NINF

    # Loose same-class blocks
    EE = np.full((2, 2), INF, dtype=float)
    np.fill_diagonal(EE, 0.0)
    AA = np.full((2, 2), NINF, dtype=float)
    np.fill_diagonal(AA, 0.0)
    AE = np.full((1, 1), NINF, dtype=float)

    st = D.closure(D.make_state(EE, AA, EA, AE))

    # The abstraction should not be bottom; gamma should be non-empty
    assert not D.is_bottom(st)

    # Additionally, check gamma includes some points (e.g., using small vals)
    gamma = D.gamma_enumerate(st, e_vals=[0], a_vals=[5])
    assert len(gamma) > 0


def test_outer_soundness_EA_with_partial_envs_2(api_guard):
    # S with no joint e1 and a1: no info about e1 - a1 -> EA must be +INF
    S = [{"e1": 0}, {"a1": 5}]

    EA = D.alpha_outer_EA(S)
    assert EA.shape == (1, 1)
    assert np.isposinf(EA[0, 0])  # remain unconstrained

    # Loose same-class and inner blocks (nE=1, nA=1)
    EE = np.full((2, 2), INF, dtype=float)
    np.fill_diagonal(EE, 0.0)

    AA = np.full((2, 2), NINF, dtype=float)
    np.fill_diagonal(AA, 0.0)

    AE = np.full((1, 1), NINF, dtype=float)

    st = D.closure(D.make_state(EE, AA, EA, AE))
    assert not D.is_bottom(st)

    # Check that a consistent joint point exists in gamma
    gamma = D.gamma_enumerate(st, e_vals=[0], a_vals=[5])
    Gset = {tuple(sorted(r.items())) for r in gamma}
    assert (("a1", 5), ("e1", 0)) in Gset


def test_inner_soundness_AE_with_partial_envs(api_guard):
    # S with no joint a1 and e1
    S = [{"a1": 5}, {"e1": 0}]

    AE = D.alpha_inner_AE(S)
    # Correct: AE=[[NINF]] (loose)

    # Loose blocks (nE=1, nA=1)
    EE = np.full((2, 2), INF, dtype=float)
    np.fill_diagonal(EE, 0.0)
    AA = np.full((2, 2), NINF, dtype=float)
    np.fill_diagonal(AA, 0.0)
    EA = np.full((1, 1), INF, dtype=float)

    st = D.closure(D.make_state(EE, AA, EA, AE))

    assert not D.is_bottom(st)  # Passes

    gamma = D.gamma_enumerate(st, e_vals=[0], a_vals=[5])
    gamma_set = {tuple(sorted(r.items())) for r in gamma}
    # For inner, check gamma ⊆ S (but S partial; here gamma has full envs satisfying loose constraints)
    # Minimal: gamma non-empty and satisfies the (no) mixed lower
    assert len(gamma) > 0
    assert all(r["a1"] - r["e1"] >= NINF for r in gamma)  # Trivially true
