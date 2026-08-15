# =============================================================================
# tests/test_correctness.py — Numerical Correctness vs Reference
# =============================================================================

import pytest

def reference_matmul_relu(a, b):
    # Pure Python / reference matrix multiplication + ReLU
    M = len(a)
    K = len(a[0])
    N = len(b[0])
    c = [[0.0 for _ in range(N)] for _ in range(M)]
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += a[i][k] * b[k][j]
            c[i][j] = max(0.0, s)
    return c

def test_matmul_relu_correctness():
    a = [[1.0, -2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, -8.0]]
    # (1*5 + -2*7) = -9 -> max(0, -9) = 0
    # (1*6 + -2*-8) = 22 -> max(0, 22) = 22
    # (3*5 + 4*7) = 43 -> max(0, 43) = 43
    # (3*6 + 4*-8) = -14 -> max(0, -14) = 0
    expected = [[0.0, 22.0], [43.0, 0.0]]
    actual = reference_matmul_relu(a, b)
    assert actual == expected
