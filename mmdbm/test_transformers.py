import pytest

from . import lattice as D
from . import transformers as T
from conftest import build_zero_closed, grid_vals

# -------------------------
# 1) Assignments: same class
# -------------------------


def test_assign_e_from_e_enforces_equality(grid_vals):
    nE, nA = 2, 0
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    # e2 := e1 + 1
    st = T.assign_e_from_e(st0, i=2, k=1, c=1)

    G = D.gamma_enumerate(st, e_vals=grid_vals, a_vals=[])
    seen = {(r["e1"], r["e2"]) for r in G}
    expected = {(e1, e1 + 1) for e1 in grid_vals if (e1 + 1) in grid_vals}
    assert seen == expected


def test_assign_a_from_a_enforces_equality(grid_vals):
    nE, nA = 0, 2
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    # a2 := a1 + 1
    st = T.assign_a_from_a(st0, j=2, ell=1, c=1)

    G = D.gamma_enumerate(st, e_vals=[], a_vals=grid_vals)
    seen = {(r["a1"], r["a2"]) for r in G}
    expected = {(a1, a1 + 1) for a1 in grid_vals if (a1 + 1) in grid_vals}
    assert seen == expected


# -------------------------
# 2) Assignments: cross class
# -------------------------


def test_assign_e_from_a_equality(grid_vals):
    nE, nA = 1, 1
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    # e1 := a1 + 1
    st = T.assign_e_from_a(st0, i=1, ell=1, c=1)

    G = D.gamma_enumerate(st, e_vals=grid_vals, a_vals=grid_vals)
    seen = {(r["e1"], r["a1"]) for r in G}

    # completeness: equality solutions are included
    expected_pairs = {(a + 1, a) for a in grid_vals if (a + 1) in grid_vals}
    assert expected_pairs.issubset(seen)

    # soundness: every solution satisfies e1 - a1 ≤ 1
    assert all(r["e1"] - r["a1"] <= 1 for r in G)


def test_assign_a_from_e_equality(grid_vals):
    nE, nA = 1, 1
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    # a1 := e1 - 1  (i.e., a1 = e1 + (-1))
    st = T.assign_a_from_e(st0, j=1, k=1, c=-1)  # a := e - 1

    G = D.gamma_enumerate(st, e_vals=grid_vals, a_vals=grid_vals)
    seen = {(r["e1"], r["a1"]) for r in G}

    expected_pairs = {(e, e - 1) for e in grid_vals if (e - 1) in grid_vals}
    assert expected_pairs.issubset(seen)
    assert all(r["a1"] - r["e1"] >= -1 for r in G)  # equivalent to e - a ≤ 1


# -------------------------
# 3) Interval assignments
# -------------------------


def test_assign_e_interval(grid_vals):
    nE, nA = 1, 0
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    # e1 := [-1, 0]
    st = T.assign_e_interval(st0, i=1, L=-1, U=0)

    G = D.gamma_enumerate(st, e_vals=grid_vals, a_vals=[])
    seen = {r["e1"] for r in G}
    assert seen == {-1, 0}


def test_assign_a_interval(grid_vals):
    nE, nA = 0, 1
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    # a1 := [0, 1]
    st = T.assign_a_interval(st0, j=1, L=0, U=1)

    G = D.gamma_enumerate(st, e_vals=[], a_vals=grid_vals)
    seen = {r["a1"] for r in G}
    assert seen == {0, 1}


# -------------------------
# 4) Guards
# -------------------------


def test_guard_ee_le_filters(grid_vals):
    nE, nA = 2, 0
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    # guard: e2 - e1 <= 0  => e2 <= e1
    st = T.guard_ee_le(st0, i=2, k=1, c=0)

    G = D.gamma_enumerate(st, e_vals=grid_vals, a_vals=[])
    assert all(r["e2"] <= r["e1"] for r in G)


def test_guard_aa_ge_filters(grid_vals):
    nE, nA = 0, 2
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    # guard: a2 - a1 >= 0  => a2 >= a1
    st = T.guard_aa_ge(st0, j=2, ell=1, c=0)

    G = D.gamma_enumerate(st, e_vals=[], a_vals=grid_vals)
    assert all(r["a2"] >= r["a1"] for r in G)


def test_guard_mixed_filters(grid_vals):
    nE, nA = 1, 1
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    # guard: e1 - a1 <= 0
    st = T.guard_ea_le(st0, i=1, j=1, c=0)
    G = D.gamma_enumerate(st, e_vals=grid_vals, a_vals=grid_vals)
    assert all(r["e1"] <= r["a1"] for r in G)

    # adding a1 - e1 ≥ 0 adds nothing new (same inequality)
    st2 = T.guard_ae_ge(st, j=1, i=1, c=0)
    G2 = D.gamma_enumerate(st2, e_vals=grid_vals, a_vals=grid_vals)
    assert all(r["e1"] <= r["a1"] for r in G2)


# -------------------------
# 5) Bottom via conflicting constraints
# -------------------------


def test_conflicting_mixed_guards_bottom():
    nE, nA = 1, 1
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    st = T.guard_ea_le(st0, i=1, j=1, c=-3)
    st = T.guard_ae_ge(st, j=1, i=1, c=2)
    assert D.is_bottom(st)


# -------------------------
# 6) Forgetting erases relations
# -------------------------


def test_forget_e_erases_relations(grid_vals):
    nE, nA = 1, 1
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    # equality e1 = a1
    st1 = T.assign_e_from_a(st0, i=1, ell=1, c=0)
    G1 = D.gamma_enumerate(st1, e_vals=grid_vals, a_vals=grid_vals)

    # forget e1 => remove mixed relations on e1
    st2 = T.forget_e(st1, i=1)
    st2 = D.closure(st2)
    G2 = D.gamma_enumerate(st2, e_vals=grid_vals, a_vals=grid_vals)

    # after forgetting, e1 should be unconstrained vs a1
    assert len(G2) >= len(G1)
    # and there should exist some env with e1 != a1 (unless grid too small)
    assert any(r["e1"] != r["a1"] for r in G2)


def test_forget_a_erases_relations(grid_vals):
    nE, nA = 1, 1
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    st1 = T.assign_a_from_e(st0, j=1, k=1, c=0)  # a1 = e1
    G1 = D.gamma_enumerate(st1, e_vals=grid_vals, a_vals=grid_vals)

    st2 = T.forget_a(st1, j=1)
    st2 = D.closure(st2)
    G2 = D.gamma_enumerate(st2, e_vals=grid_vals, a_vals=grid_vals)

    assert len(G2) >= len(G1)
    assert any(r["a1"] != r["e1"] for r in G2)


# -------------------------
# 7) Idempotence under repeated application
# -------------------------


def test_reapply_same_assignment_idempotent(grid_vals):
    nE, nA = 2, 0
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    st1 = T.assign_e_from_e(st0, i=2, k=1, c=1)
    st2 = T.assign_e_from_e(st1, i=2, k=1, c=1)  # repeat

    G1 = set(tuple(sorted(r.items())) for r in D.gamma_enumerate(st1, grid_vals, []))
    G2 = set(tuple(sorted(r.items())) for r in D.gamma_enumerate(st2, grid_vals, []))
    assert G1 == G2


def test_reapply_same_guard_idempotent(grid_vals):
    nE, nA = 1, 1
    EE, AA, EA, AE = build_zero_closed(nE, nA)
    st0 = D.closure(D.make_state(EE, AA, EA, AE))

    st1 = T.guard_ea_le(st0, i=1, j=1, c=0)
    st2 = T.guard_ea_le(st1, i=1, j=1, c=0)

    G1 = set(
        tuple(sorted(r.items())) for r in D.gamma_enumerate(st1, grid_vals, grid_vals)
    )
    G2 = set(
        tuple(sorted(r.items())) for r in D.gamma_enumerate(st2, grid_vals, grid_vals)
    )
    assert G1 == G2
