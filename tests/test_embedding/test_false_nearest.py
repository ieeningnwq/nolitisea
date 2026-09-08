import unittest

import numpy as np

from nolitisea.embedding.false_nearest import cao_method, kennel_method


class TestKennelMethod(unittest.TestCase):
    # Because of the binary magnifaction function used in the FNN test,
    # it's not easy to create unit-tests like AFN.  So we make-do with
    # silly tests.
    def test_line(self):
        x = np.linspace(0, 10, 1000)

        # A line has zero FNN at all embedding dimensions.
        f_dict = kennel_method(x, min_emb=1,max_emb=10, delay=1, theiler=0)
        for v in f_dict.values():
            for vv in v.values():
                self.assertAlmostEqual(vv, 0,places=2)

    def test_circle(self):
        t = np.linspace(0, 100 * np.pi, 5000)
        x = np.sin(t)

        # A circle has no FNN after d = 2.
        desired = np.zeros(10)
        desired[0] = 1.0
        
        f_dict = kennel_method(x, min_emb=1,max_emb=10, delay=25, theiler=0,n_jobs=-1)
        f1s=[]
        for v in f_dict.values():
            f1s.append(v['f1'])
        for f1, d in zip(f1s, desired):
            self.assertAlmostEqual(f1, d,places=2)  

    def test_curve(self):
        t = np.linspace(0, 100 * np.pi, 5000)
        x = np.sin(t) + np.sin(2 * t) + np.sin(3 * t) + np.sin(5 * t)
        
        f_dict = kennel_method(x, min_emb=1,max_emb=10, delay=25, theiler=10,n_jobs=-1)
        f1s=[]
        for v in f_dict.values():
            f1s.append(v['f1'])
        # Though this curve is a deformation of a circle, it has zero
        # FNN only after d = 3.
        for i,f1 in enumerate(f1s):
            if i<2:
                self.assertTrue(f1>0)
            else:
                self.assertTrue(f1==0)


class TestCaoMethod(unittest.TestCase):

    def test_noise(self):
        # Test dimension.afn() using uncorrelated random numbers.
        x = np.random.random(1000)
        theiler = 10
        metric = 'chebyshev'
        E_dict = cao_method(x, min_emb=1,max_emb=7, theiler=theiler, metric=metric)
        E=[]
        Es=[]
        for v in E_dict.values():
            E.append(v[0])
            Es.append(v[1])
        E=np.array(E)
        Es=np.array(Es)
        _, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]
        
        # The standard deviation of E2 should be ~ 0 for uncorrelated
        # random numbers [Ramdani et al., Physica D 223, 229 (2006)].
        # Additionally, the mean of E2 should be ~ 1.0.
        self.assertAlmostEqual(np.std(E2), 0,places=1)
        self.assertAlmostEqual(np.mean(E2), 1,places=1)

    def test_line(self):
        # Test dimension.afn() by embedding a line.
        # Particle moving uniformly in 1-D.
        a, b = np.random.random(2)
        t = np.arange(100)
        x = a + b * t
        dim = np.arange(1, 10 + 2)
        theiler = 10

        # Chebyshev distances between near-neighbors remain bounded.
        # This gives "cleaner" results when embedding known objects like
        # a line.  For a line, E = 1.0 for all dimensions as expected,
        # whereas it is (d + 1) / d (for cityblock) and sqrt(d + 1) /
        # sqrt(d) for Euclidean.  In both cases, E -> 1.0 at large d,
        # but E = 1.0 is definitely preferred.
        for metric in ('chebyshev', 'cityblock', 'euclidean'):
            Es_des = np.ones_like(dim)*(theiler + 1) * b

            if metric == 'chebyshev':
                E_des = np.ones_like(dim)
            elif metric == 'cityblock':
                E_des = (dim + 1) / dim
            elif metric == 'euclidean':
                E_des = np.sqrt((dim + 1) / dim)

            E_dict = cao_method(x, min_emb=1,max_emb=11, theiler=theiler, metric=metric,n_jobs=-1)
            for i, v in enumerate(E_dict.items()):
                self.assertAlmostEqual(E_des[i],  v[1][0], places=2)
                self.assertAlmostEqual(Es_des[i], v[1][1], places=2)   
            
if __name__ == '__main__':
    unittest.main()
