"""
Example Programs: Demonstrating May-Must DBM for Array Bounds Verification

This file provides concrete examples showing why the May-Must DBM domain is
valuable for verifying array bounds safety. These examples would be difficult
or impossible to verify precisely with a standard (single-approximation) DBM.

THE CORE INSIGHT
================
For array bounds checking, we need two DIFFERENT approximations:

    0 <= index < size

1. For `index`, we need an UPPER bound (over-approximation):
   "The index could be at most X" - if X < size, access is safe.

2. For `size`, we need a LOWER bound (under-approximation):
   "The size is definitely at least Y" - if index < Y, access is safe.

Standard DBM tracks everything with the same approximation direction.
After a join, you get "size could be at most max(a, b)" which is useless
for safety. May-Must DBM tracks "size is definitely at least min(a, b)".

DOMAIN STRUCTURE
================
- E variables (may): Track what values COULD exist (over-approximation)
  - Useful for: index bounds, pointer offsets, loop counters
  - Join takes MAX of upper bounds (loosens constraint)

- A variables (must): Track what values DEFINITELY exist (under-approximation)
  - Useful for: array sizes, buffer capacities, resource counts
  - Join takes MIN of lower bounds (loosens guarantee)

- Mixed constraints: Intervals on (e - a) differences
  - Enable bidirectional propagation: E -> A -> E and A -> E -> A
"""

import pytest
from . import maymust_dbm as MM


# =============================================================================
# Example 1: Basic Array Bounds Checking
# =============================================================================

class TestExample1_BasicArrayBounds:
    """
    Basic array bounds checking: verify access is within bounds after join.

    Program (pseudocode):
        if (condition) {
            arr = allocate(10);  // size >= 10
            i = 5;               // index = 5
        } else {
            arr = allocate(7);   // size >= 7
            i = 3;               // index = 3
        }
        // SAFETY CHECK: is arr[i] safe?
    """

    def test_both_branches_safe(self):
        """After join, we can still prove index < size."""
        # Branch 1: size >= 10, index = 5
        b1 = MM.top(1, 1)
        b1 = MM.assign_e_interval(b1, i=1, L=5, U=5)   # index = 5
        b1 = MM.assign_a_interval(b1, j=1, L=10, U=100)  # size >= 10

        # Branch 2: size >= 7, index = 3
        b2 = MM.top(1, 1)
        b2 = MM.assign_e_interval(b2, i=1, L=3, U=3)   # index = 3
        b2 = MM.assign_a_interval(b2, j=1, L=7, U=100)  # size >= 7

        # Join branches
        joined = MM.closure(MM.join(b1, b2))

        # After join:
        # - index: max(5, 3) = 5 (could be up to 5)
        # - size: min(10, 7) = 7 (definitely at least 7)
        assert joined.e_upper(1) == 5   # upper(index) = 5
        assert joined.a_lower(1) == 7   # lower(size) = 7

        # SAFETY: 5 < 7, so access is safe!
        assert MM.check_e_lt_a(joined, i=1, j=1)

    def test_join_preserves_lower_bound(self):
        """
        Key property: join takes MIN of size lower bounds.

        This is what distinguishes may-must from standard DBM:
        - Standard DBM: after join, "size could be 7 to 10" (useless for safety)
        - May-Must DBM: after join, "size is definitely at least 7" (useful!)
        """
        b1 = MM.top(0, 1)
        b1 = MM.assign_a_interval(b1, j=1, L=10, U=100)

        b2 = MM.top(0, 1)
        b2 = MM.assign_a_interval(b2, j=1, L=7, U=100)

        joined = MM.closure(MM.join(b1, b2))

        # Must takes MIN: min(10, 7) = 7
        assert joined.a_lower(1) == 7


# =============================================================================
# Example 2: Loop with Widening
# =============================================================================

class TestExample2_LoopWithWidening:
    """
    Loop analysis example with widening.

    Program (pseudocode):
        arr = allocate(N);  // size >= N
        i = 0;
        while (i < N) {
            access arr[i];  // SAFETY CHECK
            i = i + 1;
        }
    """

    def test_simple_loop_invariant(self):
        """
        Loop invariant: i < size holds throughout the loop.

        At loop entry with guard i < N and size >= N:
        - i could be 0..N-1 (over-approx)
        - size is at least N (under-approx)
        """
        N = 10

        # Initial state: i = 0, size >= N
        st = MM.top(1, 1)
        st = MM.assign_e_interval(st, i=1, L=0, U=0)     # i = 0
        st = MM.assign_a_interval(st, j=1, L=N, U=100)   # size >= N

        # Apply loop guard: i < size (i.e., i - size <= -1)
        st = MM.guard_e_minus_a_le(st, i=1, j=1, c=-1)

        # At this point: 0 <= i < size, and size >= N
        assert MM.check_safe_access(st, index_var=1, size_var=1)

    def test_widening_preserves_must_bounds(self):
        """
        Widening preserves must (under-approx) bounds but may lose may bounds.

        Key insight: Standard widening is aggressive on upper bounds.
        If upper(index) increases between iterations, widening goes to infinity.
        However, lower(size) is preserved because it only decreases (towards -inf).

        For loops, we typically need more sophisticated analysis:
        - Narrowing phase after widening
        - Threshold-based widening
        - Explicit loop invariant annotations

        This test verifies the basic widening behavior.
        """
        N = 10

        # State before loop iteration
        st1 = MM.top(1, 1)
        st1 = MM.assign_e_interval(st1, i=1, L=0, U=5)   # i in [0, 5]
        st1 = MM.assign_a_interval(st1, j=1, L=N, U=100)

        # State after one more iteration (i could be 0..6)
        st2 = MM.top(1, 1)
        st2 = MM.assign_e_interval(st2, i=1, L=0, U=6)
        st2 = MM.assign_a_interval(st2, j=1, L=N, U=100)

        # Widen
        widened = MM.widen(st1, st2)

        # Must bounds (lower) are preserved since they're not increasing
        assert widened.a_lower(1) >= N

        # May upper bound goes to infinity (standard widening behavior)
        # This is expected - more sophisticated analysis would be needed
        # to preserve tighter bounds through loops
        assert widened.e_upper(1) == MM.INF


# =============================================================================
# Example 3: Dynamic Array Resizing
# =============================================================================

class TestExample3_DynamicArrayResizing:
    """
    Dynamic array where size grows but old indices remain valid.

    Program (pseudocode):
        arr = allocate(initial_size);  // size >= initial_size
        // ... use arr with some index i
        arr = resize(arr, new_size);   // size >= new_size (where new_size > old_size)
        // old indices should still be valid
    """

    def test_resize_preserves_old_indices(self):
        """After resize to larger size, old indices are still valid."""
        initial_size = 5
        new_size = 10

        # Initial: size >= 5, index = 3
        st = MM.top(1, 1)
        st = MM.assign_e_interval(st, i=1, L=3, U=3)
        st = MM.assign_a_interval(st, j=1, L=initial_size, U=100)

        # Verify initial access is safe
        assert MM.check_safe_access(st, index_var=1, size_var=1)

        # Resize: size >= 10 (larger than before)
        st = MM.assign_a_interval(st, j=1, L=new_size, U=100)

        # Old index is still valid
        assert MM.check_safe_access(st, index_var=1, size_var=1)

    def test_resize_with_branching(self):
        """
        Resize to different sizes in different branches.

        if (condition) {
            arr = resize(arr, 10);  // size >= 10
        } else {
            arr = resize(arr, 15);  // size >= 15
        }
        // access arr[7]  // should be safe: min(10, 15) = 10 > 7
        """
        # Initial state with index = 7
        base = MM.top(1, 1)
        base = MM.assign_e_interval(base, i=1, L=7, U=7)

        # Branch 1: resize to 10
        b1 = MM.assign_a_interval(base, j=1, L=10, U=100)

        # Branch 2: resize to 15
        b2 = MM.assign_a_interval(base, j=1, L=15, U=100)

        # Join
        joined = MM.closure(MM.join(b1, b2))

        # After join: size >= min(10, 15) = 10
        assert joined.a_lower(1) == 10

        # Access arr[7] is safe: 7 < 10
        assert MM.check_safe_access(joined, index_var=1, size_var=1)


# =============================================================================
# Example 4: Multiple Arrays (Proving Relative Bounds)
# =============================================================================

class TestExample4_MultipleArrays:
    """
    Multiple arrays with related sizes.

    Program (pseudocode):
        src = allocate(N);     // size1 >= N
        dst = allocate(2*N);   // size2 >= 2*N
        for (i = 0; i < N; i++) {
            dst[i] = src[i];   // both accesses should be safe
        }
    """

    def test_copy_loop_both_safe(self):
        """Both source and destination accesses are safe in copy loop."""
        N = 10

        # e1 = index, a1 = src_size, a2 = dst_size
        st = MM.top(1, 2)

        # Source size >= N, dest size >= 2*N
        st = MM.assign_a_interval(st, j=1, L=N, U=100)      # src_size >= N
        st = MM.assign_a_interval(st, j=2, L=2*N, U=100)    # dst_size >= 2*N

        # Loop with i < N
        st = MM.assign_e_interval(st, i=1, L=0, U=N-1)      # index in [0, N-1]

        # Both accesses safe
        assert MM.check_e_lt_a(st, i=1, j=1)  # index < src_size
        assert MM.check_e_lt_a(st, i=1, j=2)  # index < dst_size


# =============================================================================
# Example 5: Negative Example (Unsafe Access Detection)
# =============================================================================

class TestExample5_UnsafeAccess:
    """
    Demonstrate that unsafe accesses are correctly detected as unverifiable.
    """

    def test_unsafe_after_join(self):
        """
        When branches have conflicting size/index, join may lose safety.

        if (condition) {
            arr = allocate(5);   // size >= 5
            i = 3;               // index = 3
        } else {
            arr = allocate(3);   // size >= 3
            i = 4;               // index = 4
        }
        // arr[i] is NOT provably safe: min(5,3) = 3, max(3,4) = 4, 4 >= 3
        """
        # Branch 1: size >= 5, index = 3
        b1 = MM.top(1, 1)
        b1 = MM.assign_e_interval(b1, i=1, L=3, U=3)
        b1 = MM.assign_a_interval(b1, j=1, L=5, U=100)

        # Branch 2: size >= 3, index = 4
        b2 = MM.top(1, 1)
        b2 = MM.assign_e_interval(b2, i=1, L=4, U=4)
        b2 = MM.assign_a_interval(b2, j=1, L=3, U=100)

        joined = MM.closure(MM.join(b1, b2))

        # After join:
        # - index: max(3, 4) = 4
        # - size: min(5, 3) = 3
        assert joined.e_upper(1) == 4
        assert joined.a_lower(1) == 3

        # NOT safe: 4 >= 3
        assert not MM.check_safe_access(joined, index_var=1, size_var=1)

    def test_negative_index_detected(self):
        """Negative index is detected as unsafe."""
        st = MM.top(1, 1)
        st = MM.assign_e_interval(st, i=1, L=-1, U=3)  # index could be -1
        st = MM.assign_a_interval(st, j=1, L=10, U=100)

        # Not safe due to negative lower bound
        assert not MM.check_safe_access(st, index_var=1, size_var=1)


# =============================================================================
# Example 6: Propagation Through Intermediate Variables
# =============================================================================

class TestExample6_Propagation:
    """
    Demonstrate bidirectional propagation through the domain.

    The domain propagates constraints through paths like E -> A -> E:
    If we know e1 - a1 <= 2 and a1 - e2 <= 3, we can derive e1 - e2 <= 5.
    """

    def test_e_to_a_to_e_propagation(self):
        """
        E -> A -> E: Derive EE constraint through A variable.

        Given:
            e1 - a1 <= 2
            a1 - e2 <= 3  (equivalently: e2 - a1 >= -3)
        Derive:
            e1 - e2 <= e1 - a1 + a1 - e2 <= 2 + 3 = 5
        """
        st = MM.top(2, 1)
        st = MM.guard_e_minus_a_le(st, i=1, j=1, c=2)    # e1 - a1 <= 2
        st = MM.guard_e_minus_a_ge(st, i=2, j=1, c=-3)   # e2 - a1 >= -3

        # Should derive: e1 - e2 <= 5
        assert st.EE[1, 2] <= 5

    def test_a_to_e_to_a_propagation(self):
        """
        A -> E -> A: Derive AA constraint through E variable.

        Given:
            a1 - e1 = 2  (exact)
            e1 - a2 <= 1
            a1 >= 5
        Derive:
            e1 = a1 - 2 >= 5 - 2 = 3
        """
        st = MM.top(1, 2)
        st = MM.assign_a_interval(st, j=1, L=5, U=100)   # a1 >= 5

        # a1 - e1 = 2 means e1 - a1 = -2
        M_up = st.Mixed_upper.copy()
        M_lo = st.Mixed_lower.copy()
        M_up[0, 0] = -2
        M_lo[0, 0] = -2
        st = MM.closure(MM.State(EE=st.EE, AA=st.AA, Mixed_upper=M_up, Mixed_lower=M_lo))

        # Should derive: e1 >= 3
        assert st.e_lower(1) >= 3


# =============================================================================
# Example 7: Real-World Pattern: Buffer with Length Field
# =============================================================================

class TestExample7_BufferWithLength:
    """
    Common pattern: buffer with explicit length field.

    struct Buffer {
        int* data;
        int length;  // invariant: length <= allocated capacity
    };

    We track:
    - e1 = index being accessed
    - a1 = buffer.length (definite lower bound on valid range)
    """

    def test_length_checked_access(self):
        """
        Safe access pattern:
            if (i < buf.length) {
                access buf.data[i];  // safe!
            }
        """
        st = MM.top(1, 1)
        # After guard: i < length
        st = MM.guard_e_lt_a(st, i=1, j=1)

        # Verify constraint is recorded
        assert st.Mixed_upper[0, 0] <= -1  # e1 - a1 <= -1

    def test_length_update_preserves_old_access(self):
        """
        After appending to buffer, old indices still valid.

            // Initially: i valid for length >= 5
            append(buf, x);  // length increases
            // i should still be valid
        """
        # Initial: index = 3, length >= 5
        st = MM.top(1, 1)
        st = MM.assign_e_interval(st, i=1, L=3, U=3)
        st = MM.assign_a_interval(st, j=1, L=5, U=100)

        assert MM.check_safe_access(st, index_var=1, size_var=1)

        # After append: length >= 6 (increased)
        st = MM.assign_a_interval(st, j=1, L=6, U=100)

        # Old index still valid
        assert MM.check_safe_access(st, index_var=1, size_var=1)


# =============================================================================
# Example 8: Comparison with Standard DBM
# =============================================================================

class TestExample8_ComparisonWithStandardDBM:
    """
    Demonstrate scenarios where standard DBM would fail to prove safety,
    but May-Must DBM succeeds.

    The key difference:
    - Standard DBM: join on size gives max (over-approx)
    - May-Must DBM: join on size gives min (under-approx)
    """

    def test_standard_dbm_would_fail_here(self):
        """
        Scenario where standard DBM loses too much precision.

        if (*) { size = 10; idx = 5; }
        else   { size = 7;  idx = 3; }
        // access arr[idx]

        Standard DBM after join:
            size in [7, 10]  (interval over-approx)
            idx  in [3, 5]   (interval over-approx)
            Cannot prove idx < size (5 could equal 7? No wait, 5 < 7)

        Actually standard DBM COULD prove this specific case.
        Let's make it harder:

        if (*) { size = 10; idx = 8; }
        else   { size = 15; idx = 5; }

        Standard DBM after join:
            size in [10, 15]
            idx  in [5, 8]
        Cannot prove idx < size because idx could be 8 and size could be 10.

        But actually both branches are safe:
            Branch 1: 8 < 10 ✓
            Branch 2: 5 < 15 ✓

        May-Must DBM:
            size >= min(10, 15) = 10
            idx  <= max(8, 5) = 8
        CAN prove: 8 < 10 ✓
        """
        # Branch 1: size >= 10, idx = 8
        b1 = MM.top(1, 1)
        b1 = MM.assign_e_interval(b1, i=1, L=8, U=8)
        b1 = MM.assign_a_interval(b1, j=1, L=10, U=100)

        # Branch 2: size >= 15, idx = 5
        b2 = MM.top(1, 1)
        b2 = MM.assign_e_interval(b2, i=1, L=5, U=5)
        b2 = MM.assign_a_interval(b2, j=1, L=15, U=100)

        joined = MM.closure(MM.join(b1, b2))

        # May-Must gives us:
        # - idx <= max(8, 5) = 8 (over-approx upper bound)
        # - size >= min(10, 15) = 10 (under-approx lower bound)
        assert joined.e_upper(1) == 8
        assert joined.a_lower(1) == 10

        # SAFE: 8 < 10
        assert MM.check_safe_access(joined, index_var=1, size_var=1)

    def test_precision_advantage(self):
        """
        More complex example showing precision advantage.

        Three branches with different size/index combinations:
        1. size >= 100, idx <= 10
        2. size >= 50,  idx <= 20
        3. size >= 30,  idx <= 15

        All branches are safe individually.

        Standard DBM would give:
            size in [30, 100]
            idx  in [10, 20]
        Cannot prove safety (20 vs 30 is borderline, depends on semantics)

        May-Must DBM gives:
            size >= 30 (minimum guarantee)
            idx  <= 20 (maximum possibility)
        Can prove: 20 < 30 ✓
        """
        branches = [
            (100, 10),  # (size_lower, idx_upper)
            (50, 20),
            (30, 15),
        ]

        states = []
        for size_lower, idx_upper in branches:
            st = MM.top(1, 1)
            st = MM.assign_e_interval(st, i=1, L=0, U=idx_upper)
            st = MM.assign_a_interval(st, j=1, L=size_lower, U=1000)
            states.append(st)

        # Join all branches
        joined = states[0]
        for st in states[1:]:
            joined = MM.join(joined, st)
        joined = MM.closure(joined)

        # Check results
        assert joined.e_upper(1) == 20   # max of all idx_uppers
        assert joined.a_lower(1) == 30   # min of all size_lowers

        # SAFE: 20 < 30
        assert MM.check_safe_access(joined, index_var=1, size_var=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
