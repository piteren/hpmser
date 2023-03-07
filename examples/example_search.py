import math
import time
from typing import Optional

from hpmser.search_function import hpmser


def some_function(x:float, y:int, z:Optional[bool]) -> float:

    if y not in [0,1,2,3,4,5,6,7,8]:
        raise Exception('unsupported y value')

    val = 1 - x * (x - 2) + math.sin(x) - (y-3)*(y+4) + y
    if z: val += 1
    if z is None: val -= 1

    time.sleep(1)
    return val


if __name__ == "__main__":

    # parameters space definition
    psdd = {
        'x':    [-5.0, 5.0],        # range of floats
        'y':    [0, 8],             # range of ints
        'z':    (None,False,True),  # set of 3 values
    }

    hpmser(
        func=       some_function,
        func_psdd=  psdd,
        devices=    0.5,            # half of system CPUs
        n_loops=    200,
        pref_axes=  ['x','y'],      # axes to plot
    )