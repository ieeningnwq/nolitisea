import unittest
import numpy as np
from nolitisea.embedding.false_nearest import kennel_method


class TestKennelMethod(unittest.TestCase):
    # Because of the binary magnifaction function used in the FNN test,
    # it's not easy to create unit-tests like AFN.  So we make-do with
    # silly tests.
    def test_line(self):
        x = np.linspace(0, 10, 1000)

        # A line has zero FNN at all embedding dimensions.
        f_dict = kennel_method(x, min_emb=1,max_emb=10, delay=1, theiler=0)
        for _,v in f_dict.items():
            for _,vv in v.items():
                self.assertAlmostEqual(vv, 0,places=2)

    def test_circle(self):
        t = np.linspace(0, 100 * np.pi, 5000)
        x = np.sin(t)

        # A circle has no FNN after d = 2.
        desired = np.zeros(10)
        desired[0] = 1.0
        
        f_dict = kennel_method(x, min_emb=1,max_emb=10, delay=25, theiler=0,n_jobs=-1)
        f1s=[]
        for _,v in f_dict.items():
            f1s.append(v['f1'])
        for f1, d in zip(f1s, desired):
            self.assertAlmostEqual(f1, d,places=2)  

    def test_curve(self):
        t = np.linspace(0, 100 * np.pi, 5000)
        x = np.sin(t) + np.sin(2 * t) + np.sin(3 * t) + np.sin(5 * t)
        
        f_dict = kennel_method(x, min_emb=1,max_emb=10, delay=25, theiler=10,n_jobs=-1)
        f1s=[]
        for _,v in f_dict.items():
            f1s.append(v['f1'])
        # Though this curve is a deformation of a circle, it has zero
        # FNN only after d = 3.
        for i,f1 in enumerate(f1s):
            if i<2:
                self.assertTrue(f1>0)
            else:
                self.assertTrue(f1==0)
