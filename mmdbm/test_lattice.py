import pytest
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


def test_bottom_via_cross_inconsistency(api_guard, sizes):
    nE, nA = sizes
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    if nE >= 1 and nA >= 1:
        EA[0][0] = -1
        AE[0][0] = -2  # EA=-1, -AE=2 ⇒ EA < -AE ⇒ inconsistent
    st = D.closure(D.make_state(EE, AA, EA, AE))
    assert D.is_bottom(st)


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
