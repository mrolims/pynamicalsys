"""
smallest_denominator finds the smallest q such that some p / q lies within
delta of x. It descends the Stern-Brocot tree in runs instead of one mediant at
a time, so its cost no longer grows like 1 / delta.

The reference below is the one-step descent (Sander and Meiss, 2020,
Algorithm 2), which is exact but whose cost grows like 1 / delta near a
rational with a small denominator. The two must agree on every input.
"""

import time

import numpy as np
import pytest

from pynamicalsys.discrete_time.rotation import smallest_denominator


def one_step_descent(x, delta):
    x = x - np.floor(x)
    if x < delta or 1.0 - x < delta:
        return 1
    p_left, q_left, p_right, q_right = 0, 1, 1, 1
    while True:
        p, q = p_left + p_right, q_left + q_right
        if abs(x - p / q) < delta:
            return q
        if x < p / q:
            p_right, q_right = p, q
        else:
            p_left, q_left = p, q


def near_rationals(max_denominator, distance):
    # Points just inside, on, and just outside the delta-neighborhood of
    # rationals with small denominators, where the one-step descent takes
    # the longest runs.
    points = []
    for q in range(1, max_denominator + 1):
        for p in range(q):
            for factor in (0.5, 0.999999, 1.0, 1.000001, 1.5, 3.0, 10.0):
                points += [p / q + factor * distance, p / q - factor * distance]
    return points


@pytest.mark.parametrize("tol", [2, 4, 6])
def test_agrees_with_the_one_step_descent(tol):
    delta = 10.0 ** (-tol)
    rng = np.random.default_rng(tol)
    points = near_rationals(25, delta) + list(rng.uniform(-3.0, 3.0, 2000))
    for x in points:
        assert smallest_denominator(x, delta) == one_step_descent(x, delta), x


def test_known_value_next_to_one_half():
    # The rationals closest to 1/2 are (n + 1) / (2n + 1), at distance
    # 1 / (2 (2n + 1)). The first one within 1e-8 of 1/2 + 3e-8 has
    # 1 / (2q) < 4e-8, that is, q = 12500001.
    assert smallest_denominator(0.5 + 3e-8, 1e-8) == 12_500_001


def test_fast_for_the_smallest_tolerance():
    # The one-step descent needs about 1e14 steps here.
    smallest_denominator(0.3, 1e-2)
    start = time.perf_counter()
    q = smallest_denominator(0.5 + 3e-15, 1e-15)
    assert time.perf_counter() - start < 1.0
    assert q > 10**14
