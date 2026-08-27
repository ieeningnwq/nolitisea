import unittest

import numpy as np
from scipy.spatial.distance import chebyshev, cityblock, euclidean

from nolitisea.utils.dist import pairwise_row_distance


class TestDistWrapper(unittest.TestCase):
    def setUp(self):
        # 测试样本 shape=(3,2)，强制float64，匹配jit签名
        self.x = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 3.0]], dtype=np.float64)
        self.y = np.array([[0.0, 0.0], [4.0, 6.0], [2.0, -1.0]], dtype=np.float64)

        # scipy逐行作为参考真值
        self.ref_cityblock = np.array(
            [
                cityblock(self.x[0], self.y[0]),
                cityblock(self.x[1], self.y[1]),
                cityblock(self.x[2], self.y[2]),
            ]
        )
        self.ref_euclidean = np.array(
            [
                euclidean(self.x[0], self.y[0]),
                euclidean(self.x[1], self.y[1]),
                euclidean(self.x[2], self.y[2]),
            ]
        )
        self.ref_chebyshev = np.array(
            [
                chebyshev(self.x[0], self.y[0]),
                chebyshev(self.x[1], self.y[1]),
                chebyshev(self.x[2], self.y[2]),
            ]
        )

    def test_dist_cityblock(self):
        res = pairwise_row_distance(self.x, self.y, metric="cityblock")
        np.testing.assert_allclose(res, self.ref_cityblock)

    def test_dist_euclidean(self):
        res = pairwise_row_distance(self.x, self.y, metric="euclidean")
        np.testing.assert_allclose(res, self.ref_euclidean)

    def test_dist_chebyshev(self):
        res = pairwise_row_distance(self.x, self.y, metric="chebyshev")
        np.testing.assert_allclose(res, self.ref_chebyshev)

    def test_dist_default_metric(self):
        # 默认参数 chebyshev
        res = pairwise_row_distance(self.x, self.y)
        np.testing.assert_allclose(res, self.ref_chebyshev)

    def test_dist_unknown_metric_raise(self):
        with self.assertRaises(ValueError):
            pairwise_row_distance(self.x, self.y, metric="cosine")

    def test_dist_one_sample(self):
        # 单个样本 (1,3)
        x1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        y1 = np.array([[4.0, 6.0, 8.0]], dtype=np.float64)
        d = pairwise_row_distance(x1, y1, metric="chebyshev")
        self.assertEqual(d.shape, (1,))
        self.assertAlmostEqual(d[0], 5.0)

    def test_dist_identical_points_all_zero(self):
        x_ = np.array([[2.5, -1.2], [0.0, 7.1]], dtype=np.float64)
        y_ = x_.copy()
        for met in ["cityblock", "euclidean", "chebyshev"]:
            d = pairwise_row_distance(x_, y_, metric=met)
            np.testing.assert_allclose(d, np.zeros(2))


if __name__ == "__main__":
    unittest.main()
