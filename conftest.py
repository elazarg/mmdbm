import itertools
import math
import pytest

from mmdbm import lattice as D

INF = math.inf
NINF = -math.inf


def to_list_matrix(M):
    """Convert matrices that might be numpy arrays to vanilla lists, without importing numpy."""
    if hasattr(M, "tolist"):
        return M.tolist()
    # Already a sequence of sequences
    return [list(row) for row in M]


def get_blocks(state):
    """Uniformly extract blocks from state without assuming internal representation."""
    if hasattr(D, "get_blocks"):
        EE, AA, EA, AE = D.get_blocks(state)
    else:
        EE, AA, EA, AE = state["EE"], state["AA"], state["EA"], state["AE"]
    return (
        to_list_matrix(EE),
        to_list_matrix(AA),
        to_list_matrix(EA),
        to_list_matrix(AE),
    )


def envs(nE, nA, e_vals, a_vals):
    """Enumerate environments rho over tiny grids."""
    for e_tuple in itertools.product(e_vals, repeat=nE):
        for a_tuple in itertools.product(a_vals, repeat=nA):
            rho = {}
            for i, v in enumerate(e_tuple, 1):
                rho[f"e{i}"] = v
            for j, v in enumerate(a_tuple, 1):
                rho[f"a{j}"] = v
            yield rho


def build_zero_closed(nE, nA, fill_upper=INF, fill_lower=NINF):
    """
    Build loose (almost-top) blocks with correct diagonal zeros:
      - EE, EA are upper-bound blocks (default = +inf, EE diag 0)
      - AA, AE are lower-bound blocks (default = -inf, AA diag 0)
    Shapes: EE (nE+1)x(nE+1), AA (nA+1)x(nA+1), EA nExnA, AE nAxnE
    """
    EE = [[fill_upper for _ in range(nE + 1)] for _ in range(nE + 1)]
    AA = [[fill_lower for _ in range(nA + 1)] for _ in range(nA + 1)]
    EA = [[fill_upper for _ in range(nA)] for _ in range(nE)]
    AE = [[fill_lower for _ in range(nE)] for _ in range(nA)]
    for i in range(nE + 1):
        EE[i][i] = 0
    for j in range(nA + 1):
        AA[j][j] = 0
    return EE, AA, EA, AE


@pytest.fixture(scope="session")
def grid_vals():
    # Tiny domain as requested
    return [-1, 0, 1]


@pytest.fixture(scope="session", params=[(1, 1), (2, 2)])
def sizes(request):
    """Parametrize tests over (nE, nA) = (1,1) and (2,2)."""
    return request.param


@pytest.fixture(scope="session")
def api_guard():
    """Ensure required API exists; skip tests if not."""
    required = [
        "make_state",
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
    missing = [name for name in required if not hasattr(D, name)]
    if missing:
        pytest.skip(f"mmdbm is missing required functions: {missing}")
