import unittest

import numpy as np

from nolitisea.stationarity.recurrence import (
    determinism,
    entropy_diagonal_lines,
    extract_diagonal_lengths,
    laminarity,
    recurrence_rate,
    trapping_time,
)


class TestRQAMetrics(unittest.TestCase):
    def test_recurrence_rate_no_recurrence(self):
        """RR: zero recurrence points."""
        rmat = np.zeros((4, 4), dtype=bool)
        rr = recurrence_rate(rmat)
        self.assertAlmostEqual(rr, 0.0)

    def test_recurrence_rate_all_off_diagonal(self):
        """RR: all off-diagonal points are recurrent."""
        rmat = np.ones((3, 3), dtype=bool)
        rr = recurrence_rate(rmat)
        # M=3, total pairs = 3*2 = 6, all 6 points are True
        self.assertAlmostEqual(rr, 1.0)

    def test_determinism_no_diagonal_lines(self):
        """No diagonal lines longer or equal l_min=2, DET=0."""
        rmat = np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool
        )
        det = determinism(rmat, l_min=2)
        self.assertAlmostEqual(det, 0.0)

    def test_determinism_single_diagonal(self):
        """One diagonal of length=3, upper triangle."""
        rmat = np.zeros((5, 5), dtype=bool)
        # diagonal offset k=1: positions (0,1),(1,2),(2,3) → length=3
        for i in range(3):
            rmat[i, i + 1] = True
        det = determinism(rmat, l_min=2)
        total_rec = np.sum(rmat)
        diag_sum = 3
        expected = diag_sum / total_rec
        self.assertAlmostEqual(det, expected)

    def test_laminarity_pure_vertical(self):
        """Vertical run length = 3, LAM test."""
        rmat = np.zeros((5, 5), dtype=bool)
        # column‑0, rows 1‑3: vertical line
        rmat[1, 0] = True
        rmat[2, 0] = True
        rmat[3, 0] = True
        lam = laminarity(rmat, v_min=2)
        total_rec = np.sum(rmat)
        vert_sum = 3
        expected = vert_sum / total_rec
        self.assertAlmostEqual(lam, expected)

    def test_trapping_time(self):
        """Average vertical length test."""
        rmat = np.zeros((5, 5), dtype=bool)
        rmat[1:4, 0] = True  # len=3
        rmat[2:4, 1] = True  # len=2
        tt = trapping_time(rmat, v_min=2)
        self.assertAlmostEqual(tt, 2.5)

    def test_entropy_zero_lines(self):
        """ENTR returns zero when there are no valid diagonals."""
        rmat = np.zeros((4, 4), dtype=bool)
        entr = entropy_diagonal_lines(rmat, l_min=2)
        self.assertEqual(entr, 0.0)

    def test_extract_diag_lengths(self):
        """Upper-triangle diagonal extraction, no double-count."""
        rmat = np.zeros((5, 5), dtype=bool)
        for i in range(3):
            rmat[i, i + 1] = True
        lengths = extract_diagonal_lengths(rmat, l_min=2)
        self.assertEqual(list(lengths), [3])


if __name__ == "__main__":
    unittest.main()
