from pypaq.lipytools.plots import two_dim
import random
import unittest

from hpmser.helpers import val_linear
from hpmser.helpers import str_floatL



class TestHelpers(unittest.TestCase):

    def test_base(self):

        fl = [random.random() for _ in range(5)]
        print(fl)
        print(str_floatL(fl))

        fl = [random.random() for _ in range(10)]
        print(fl)
        print(str_floatL(fl))


    def test_val_linear(self):

        r = 100
        counter = list(range(r))
        vals = [val_linear(
            s_val=      0.9,
            e_val=      -0.3,
            sf=         0.1,
            ef=         0.3,
            counter=    c,
            max_count=  r) for c in counter]
        two_dim(vals)

        v = val_linear(0.9,-0.3,0.1,0.3,10,100)
        print(v)
        self.assertTrue(v==0.9)