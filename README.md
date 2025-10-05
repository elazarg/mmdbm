# mmdbm — Combined May–Must Difference-Bound Domain

`mmdbm` is a reference implementation of the **Combined May–Must Difference-Bound Matrix (DBM) domain**, a numerical abstract domain designed to jointly model *over-approximated (may)* and *under-approximated (must)* relational constraints.

It provides a minimal, self-contained Python module implementing the domain’s semantics and a corresponding pytest suite that checks the key algebraic and logical properties.

## Overview

The domain tracks difference constraints between two disjoint classes of variables:

* **May variables** (`e₁, e₂, …`) — outer (over-approximation) layer
* **Must variables** (`a₁, a₂, …`) — inner (under-approximation) layer

Each abstract state consists of four block matrices:

| Block | Relation      | Polarity | Meaning                |
| :---- | :------------ | :------- | :--------------------- |
| `EE`  | `eᵢ − eⱼ ≤ c` | may      | intra-may differences  |
| `AA`  | `aᵢ − aⱼ ≥ c` | must     | intra-must differences |
| `EA`  | `eᵢ − aⱼ ≤ c` | may      | may→must constraints   |
| `AE`  | `aᵢ − eⱼ ≥ c` | must     | must→may constraints   |

Closure rules maintain canonicality and ensure polarity-correct propagation across blocks.

## Features

* Canonical closure (sound and idempotent)
* Mixed-block propagation consistent with polarity
* Sound outer (`α↑`) and inner (`α↓`, right-adjoint) abstractions
* Exact γ-enumeration for small finite grids (for testing)
* Lattice operations (`meet`, `join`) and bottom detection

## Tests

The included test suite (`pytest`) checks:

* closure correctness and idempotence
* unary and mixed propagation
* soundness of outer and inner abstractions
* consistency of γ (intersection of envelopes)
* lattice monotonicity and bottom detection

Run with:

```bash
pip install -r requirements.txt
pytest -q
```

## Status

This implementation is intended for **specification and pedagogy**, not performance.
It aims to be *semantics-faithful* to the formal domain described in the paper.