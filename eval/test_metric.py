"""Unit tests for the official metric copy (eval/recodai_f1.py)."""
import doctest

import numpy as np
import pandas as pd
import pytest

from eval import recodai_f1 as m

H, W = 60, 80


def rect(y0, y1, x0, x1):
    a = np.zeros((H, W), dtype=np.uint8)
    a[y0:y1, x0:x1] = 1
    return a


A = rect(5, 20, 5, 30)
B = rect(30, 50, 40, 70)
SHAPE = f"[{H}, {W}]"


def run(labels, preds, shapes=None):
    n = len(labels)
    shapes = shapes or [SHAPE if l != "authentic" else "authentic" for l in labels]
    sol = pd.DataFrame({"row_id": range(n), "annotation": labels, "shape": shapes})
    sub = pd.DataFrame({"row_id": range(n), "annotation": preds})
    return m.score(sol, sub, "row_id")


def test_official_doctests():
    res = doctest.testmod(m)
    assert res.attempted >= 5 and res.failed == 0


def test_rle_roundtrip():
    enc = m.rle_encode([A, B])
    a, b = [m.rle_decode(x, (H, W)) for x in enc.split(";")]
    assert np.array_equal(a, A) and np.array_equal(b, B)


def test_perfect_prediction():
    assert run([m.rle_encode([A, B])], [m.rle_encode([A, B])]) == 1.0


def test_authentic_predicted_authentic():
    assert run(["authentic"], ["authentic"]) == 1.0


def test_authentic_with_predicted_mask():
    assert run(["authentic"], [m.rle_encode([A])]) == 0.0


def test_forged_predicted_authentic():
    assert run([m.rle_encode([A])], ["authentic"]) == 0.0


def test_two_masks_swapped_order():
    # Hungarian matching makes instance order irrelevant.
    assert run([m.rle_encode([A, B])], [m.rle_encode([B, A])]) == 1.0


def test_missing_instance_halves_score():
    assert run([m.rle_encode([A, B])], [m.rle_encode([A])]) == pytest.approx(0.5)


def test_excess_prediction_penalty():
    # 1 GT, 2 predictions (one perfect): matched F1 = 1, penalty = 1/2.
    assert run([m.rle_encode([A])], [m.rle_encode([A, B])]) == pytest.approx(0.5)


def test_merged_prediction_is_penalised():
    # One blob covering both GT instances scores well below a perfect split.
    merged = (A | B).astype(np.uint8)
    s = run([m.rle_encode([A, B])], [m.rle_encode([merged])])
    f1_best = max(m.calculate_f1_score(merged, A), m.calculate_f1_score(merged, B))
    assert s == pytest.approx(f1_best / 2)


def test_partial_overlap_matches_pixel_f1():
    P = rect(5, 20, 5, 20)  # subset of A
    tp = P.sum(); f1 = 2 * tp / (P.sum() + A.sum())
    assert run([m.rle_encode([A])], [m.rle_encode([P])]) == pytest.approx(f1)


def test_mean_over_images():
    s = run(["authentic", m.rle_encode([A]), "authentic"],
            ["authentic", m.rle_encode([A]), m.rle_encode([A])])
    assert s == pytest.approx(2 / 3)
