import math
from pypaq.lipytools.files import prep_folder
from pypaq.mpython.devices import DevicesPypaq
import random
import time
import unittest

from tests.envy import flush_tmp_dir

from hpmser.search_function import hpmser

HPMSER_FD = f'{flush_tmp_dir()}/hpmser'


class TestSearchFunction(unittest.TestCase):

    def setUp(self) -> None:
        prep_folder(HPMSER_FD, flush_non_empty=True)


    def test_simple_run(self):

        n_proc =            30
        av_time =           0.1  # avg num of seconds for calculation
        exception_prob =    0.01
        verb =              1

        def some_func(
                name :str,
                device :DevicesPypaq,
                a :int,
                b :float,
                c :float,
                d :float,
                wait=   0.1,
                verb=   0):
            if random.random() < exception_prob: raise Exception('RandomException')
            val = math.sin(b-a*c) - abs(a+3.1)/(d+0.5) - pow(b/2,2)/12
            time.sleep(random.random()*wait)
            if verb>0 :print(f'... {name} calculated on {device} ({a}) ({b}) >> ({val})')
            return val

        func_const = {
            'name':'pio',
            'wait': av_time*2,
            'verb': verb-1}

        psdd = {
            'a':    [-5,    5],
            'b':    [-5.0,  5.0],
            'c':    [-2.0,  2],
            'd':    (0.0, 1.5, 5)}

        srl = hpmser(
            func=               some_func,
            func_psdd=          psdd,
            func_const=         func_const,
            devices=            [None] * n_proc,
            n_loops=            500,
            hpmser_FD=          HPMSER_FD,
            raise_exceptions=   False)

        print(len(srl))
        self.assertTrue(len(srl)==500)