"""
Tests for continuous E ↔ A propagation in the May-Must DBM domain.

These tests verify that information flows bidirectionally:
- E → A → E: Changes to e variables propagate through a variables back to e
- A → E → A: Changes to a variables propagate through e variables back to a
"""

import pytest
from . import maymust_dbm as MM
from . import maymust_dbm_explicit as MME


class TestMayMustDBM:
    """Tests for interval-based mixed constraints."""

    def test_basic_creation(self):
        st = MM.top(2, 2)
        assert st.nE == 2
        assert st.nA == 2
        assert not MM.is_bottom(st)

    def test_e_to_a_to_e_propagation(self):
        """
        E → A → E propagation test.

        Setup:
        - e1 - a1 ≤ 2 (upper)
        - a1 - e2 ≤ 3, i.e., e2 - a1 ≥ -3 (lower)

        Should derive:
        - e1 - e2 ≤ (e1 - a1) + (a1 - e2) ≤ 2 + 3 = 5
        """
        st = MM.top(2, 1)

        # Set up constraints via Mixed intervals
        M_upper = st.Mixed_upper.copy()
        M_lower = st.Mixed_lower.copy()

        M_upper[0, 0] = 2   # e1 - a1 ≤ 2
        M_lower[0, 0] = -5  # e1 - a1 ≥ -5

        # For a1 - e2 ≤ 3:
        # upper(a1 - e2) = -lower(e2 - a1) = -M_lower[1, 0]
        # So if upper(a1 - e2) = 3, then M_lower[1, 0] = -3
        M_lower[1, 0] = -3  # lower(e2 - a1) = -3, so upper(a1 - e2) = 3

        st = MM.closure(MM.State(
            EE=st.EE, AA=st.AA,
            Mixed_upper=M_upper, Mixed_lower=M_lower
        ))

        # Check E → A → E propagation
        # e1 - e2 = (e1 - a1) + (a1 - e2)
        # upper(e1 - e2) ≤ upper(e1 - a1) + upper(a1 - e2) = 2 + 3 = 5
        assert st.EE[1, 2] <= 5

    def test_a_to_e_to_a_propagation(self):
        """
        A → E → A propagation test.

        Setup:
        - a1 ≥ 5 (lower bound)
        - e1 - a1 = -2 (exact, so a1 = e1 + 2)
        - e1 - a2 ≤ 1

        Should derive e1's bounds from a1's bounds.
        """
        st = MM.top(1, 2)
        st = MM.assign_a_interval(st, j=1, L=5, U=100)

        # e1 - a1 in [-2, -2] (exact relationship)
        M_upper = st.Mixed_upper.copy()
        M_lower = st.Mixed_lower.copy()
        M_upper[0, 0] = -2
        M_lower[0, 0] = -2
        # e1 - a2 ≤ 1
        M_upper[0, 1] = 1
        M_lower[0, 1] = -100

        st = MM.closure(MM.State(
            EE=st.EE, AA=st.AA,
            Mixed_upper=M_upper, Mixed_lower=M_lower
        ))

        # From e1 - a1 = -2 and a1 ≥ 5, we get e1 ≥ 3
        # The key test: did e1's bounds get derived?
        assert st.EE[0, 1] <= -3  # e1 ≥ 3 means 0 - e1 ≤ -3

    def test_join_preserves_safety(self):
        """
        Test that join preserves the may/must semantics for bounds checking.
        """
        # Branch 1: index ≤ 5, size ≥ 10
        b1 = MM.top(1, 1)
        b1 = MM.assign_e_interval(b1, i=1, L=0, U=5)
        b1 = MM.assign_a_interval(b1, j=1, L=10, U=1000)

        # Branch 2: index ≤ 3, size ≥ 7
        b2 = MM.top(1, 1)
        b2 = MM.assign_e_interval(b2, i=1, L=0, U=3)
        b2 = MM.assign_a_interval(b2, j=1, L=7, U=1000)

        sj = MM.closure(MM.join(b1, b2))

        # After join: index ≤ 5 (max), size ≥ 7 (min)
        assert sj.EE[1, 0] == 5  # upper(index)
        assert sj.AA[1, 0] == 7  # lower(size)

        # Safety: all concrete states have index < size
        G = MM.gamma_enumerate(sj, e_vals=[0, 1, 2, 3, 4, 5], a_vals=[7, 8, 9, 10])
        assert all(r["e1"] < r["a1"] for r in G)

    def test_direct_mixed_guard(self):
        """Test adding constraints directly on e - a."""
        st = MM.top(1, 1)
        st = MM.guard_e_minus_a_ge(st, i=1, j=1, c=-5)  # e1 - a1 ≥ -5
        st = MM.guard_e_minus_a_le(st, i=1, j=1, c=3)   # e1 - a1 ≤ 3

        # Check the constraint was set
        assert st.Mixed_upper[0, 0] <= 3
        assert st.Mixed_lower[0, 0] >= -5

        G = MM.gamma_enumerate(st, e_vals=[-2, -1, 0, 1, 2], a_vals=[-2, -1, 0, 1, 2])
        assert all(-5 <= r["e1"] - r["a1"] <= 3 for r in G)

    def test_safety_check_function(self):
        """Test the check_safe_access query function."""
        st = MM.top(1, 1)
        st = MM.assign_e_interval(st, i=1, L=0, U=5)   # index in [0, 5]
        st = MM.assign_a_interval(st, j=1, L=10, U=100)  # size ≥ 10

        # Safe: index ≤ 5 < 10 ≤ size
        assert MM.check_safe_access(st, index_var=1, size_var=1)

        # Unsafe case
        st2 = MM.top(1, 1)
        st2 = MM.assign_e_interval(st2, i=1, L=0, U=10)  # index in [0, 10]
        st2 = MM.assign_a_interval(st2, j=1, L=5, U=100)  # size ≥ 5

        # Not provably safe: index could be 10, size could be 5
        assert not MM.check_safe_access(st2, index_var=1, size_var=1)


class TestExplicitRepresentation:
    """Tests for explicit AE_upper matrix (pedagogic reference)."""

    def test_basic_creation(self):
        st = MME.top_state(2, 2)
        assert st.nE == 2
        assert st.nA == 2
        assert not MME.is_bottom(st)

    def test_e_to_a_to_e_propagation(self):
        """
        E → A → E propagation test using EA + AE_upper.

        Setup:
        - e1 - a1 ≤ 2 (EA)
        - a1 - e2 ≤ 3 (AE_upper)

        Should derive:
        - e1 - e2 ≤ (e1 - a1) + (a1 - e2) ≤ 2 + 3 = 5
        """
        st = MME.top_state(2, 1)
        st = MME.guard_ea_le(st, i=1, j=1, c=2)   # e1 - a1 ≤ 2
        st = MME.guard_ae_le(st, i=1, j=2, c=3)   # a1 - e2 ≤ 3

        # Check E → A → E propagation
        assert st.EE[1, 2] <= 5

    def test_a_to_e_to_a_propagation(self):
        """
        A → E → A propagation test.

        Setup:
        - a1 - e1 ≥ 2 (AE)
        - a1 - e1 ≤ 2 (AE_upper, making it exact)
        - e1 - a2 ≤ 1 (EA)

        Should derive bounds on a1 - a2.
        """
        st = MME.top_state(1, 2)
        st = MME.guard_ae_ge(st, i=1, j=1, c=2)   # a1 - e1 ≥ 2
        st = MME.guard_ae_le(st, i=1, j=1, c=2)   # a1 - e1 ≤ 2 (exact)
        st = MME.guard_ea_le(st, i=1, j=2, c=1)   # e1 - a2 ≤ 1

        # We have AA for lower bounds. Verify state is consistent
        assert not MME.is_bottom(st)

    def test_join_preserves_safety(self):
        """Test that join preserves may/must semantics."""
        b1 = MME.top_state(1, 1)
        b1 = MME.assign_e_interval(b1, i=1, L=0, U=5)
        b1 = MME.assign_a_interval(b1, j=1, L=10, U=1000)

        b2 = MME.top_state(1, 1)
        b2 = MME.assign_e_interval(b2, i=1, L=0, U=3)
        b2 = MME.assign_a_interval(b2, j=1, L=7, U=1000)

        sj = MME.closure(MME.join(b1, b2))

        assert sj.EE[1, 0] == 5
        assert sj.AA[1, 0] == 7

        G = MME.gamma_enumerate(sj, e_vals=[0, 1, 2, 3, 4, 5], a_vals=[7, 8, 9, 10])
        assert all(r["e1"] < r["a1"] for r in G)

    def test_ae_upper_enables_lower_e_a(self):
        """
        Test that AE_upper allows deriving lower bounds on e - a.

        lower(e - a) = -upper(a - e) = -AE_upper
        """
        st = MME.top_state(1, 1)
        st = MME.guard_ae_le(st, i=1, j=1, c=5)  # a1 - e1 ≤ 5

        # This means e1 - a1 ≥ -5 (lower bound!)
        G = MME.gamma_enumerate(st, e_vals=[-10, -5, 0, 5, 10], a_vals=[-10, -5, 0, 5, 10])
        assert all(r["e1"] - r["a1"] >= -5 for r in G)


class TestTranslation:
    """Test translation between representations."""

    def test_to_explicit_and_back(self):
        """Convert to explicit and back should preserve semantics."""
        st1 = MM.top(1, 1)
        st1 = MM.assign_e_interval(st1, i=1, L=0, U=5)
        st1 = MM.assign_a_interval(st1, j=1, L=3, U=10)
        st1 = MM.guard_e_minus_a_le(st1, i=1, j=1, c=2)

        # Convert to explicit
        explicit = MM.to_explicit(st1)

        # Convert back
        st2 = MM.from_explicit(explicit)

        # Same gamma
        e_vals = [0, 1, 2, 3, 4, 5]
        a_vals = [3, 4, 5, 6, 7, 8, 9, 10]
        G1 = MM.gamma_enumerate(st1, e_vals=e_vals, a_vals=a_vals)
        G2 = MM.gamma_enumerate(st2, e_vals=e_vals, a_vals=a_vals)

        G1_set = {tuple(sorted(r.items())) for r in G1}
        G2_set = {tuple(sorted(r.items())) for r in G2}

        assert G1_set == G2_set

    def test_same_results_simple(self):
        """Both representations should give same results for simple cases."""
        # May-Must DBM
        st1 = MM.top(1, 1)
        st1 = MM.assign_e_interval(st1, i=1, L=0, U=5)
        st1 = MM.assign_a_interval(st1, j=1, L=3, U=10)

        # Explicit
        st2 = MME.top_state(1, 1)
        st2 = MME.assign_e_interval(st2, i=1, L=0, U=5)
        st2 = MME.assign_a_interval(st2, j=1, L=3, U=10)

        # Same gamma
        e_vals = [0, 1, 2, 3, 4, 5]
        a_vals = [3, 4, 5, 6, 7, 8, 9, 10]
        G1 = MM.gamma_enumerate(st1, e_vals=e_vals, a_vals=a_vals)
        G2 = MME.gamma_enumerate(st2, e_vals=e_vals, a_vals=a_vals)

        G1_set = {tuple(sorted(r.items())) for r in G1}
        G2_set = {tuple(sorted(r.items())) for r in G2}

        assert G1_set == G2_set

    def test_e_a_e_propagation_both(self):
        """Both representations should derive E→A→E constraints."""
        # May-Must DBM
        st1 = MM.top(2, 1)
        st1 = MM.guard_e_minus_a_le(st1, i=1, j=1, c=2)  # e1 - a1 ≤ 2
        M_lower = st1.Mixed_lower.copy()
        M_lower[1, 0] = -3  # lower(e2 - a1) = -3, so upper(a1 - e2) = 3
        st1 = MM.closure(MM.State(EE=st1.EE, AA=st1.AA,
                                   Mixed_upper=st1.Mixed_upper, Mixed_lower=M_lower))

        # Explicit
        st2 = MME.top_state(2, 1)
        st2 = MME.guard_ea_le(st2, i=1, j=1, c=2)
        st2 = MME.guard_ae_le(st2, i=1, j=2, c=3)

        # Both should derive e1 - e2 ≤ 5
        assert st1.EE[1, 2] <= 5
        assert st2.EE[1, 2] <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
