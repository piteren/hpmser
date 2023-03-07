from ompr.runner import RunningWorker
from pypaq.mpython.devices import DevicesPypaq
from pypaq.pms.base import POINT, get_params
from typing import Callable, Optional, List, Any


# hpmser RunningWorker (process run by OMP in hpmser)
class HRW(RunningWorker):

    def __init__(
            self,
            func: Callable,
            func_const: Optional[POINT],
            device: DevicesPypaq= None):

        self.func = func
        self.func_const = func_const if func_const else {}
        self.device = device

        # manage 'device'/'devices' & 'hpmser_mode' param in func >> set it in func if needed
        func_args = get_params(self.func)
        func_args = list(func_args['with_defaults'].keys()) + func_args['without_defaults']
        for k in ['device','devices']:
            if k in func_args:
                self.func_const[k] = self.device
        if 'hpmser_mode' in func_args: self.func_const['hpmser_mode'] = True

    # processes given spoint, passes **kwargs
    def process(
            self,
            spoint: POINT,
            **kwargs) -> Any:

        spoint_with_defaults = {}
        spoint_with_defaults.update(self.func_const)
        spoint_with_defaults.update(spoint)

        res = self.func(**spoint_with_defaults)
        if type(res) is dict: score = res['score']
        else:                 score = res

        msg = {'spoint':spoint, 'score':score}
        msg.update(kwargs)
        return msg


# returns nice string of floats list
def str_floatL(
        all_w :List[float],
        cut_above=      5,
        float_prec=     4) -> str:
    ws = '['
    if cut_above < 5: cut_above = 5 # cannot be less than 5
    if len(all_w) > cut_above:
        for w in all_w[:2]: ws += f'{w:.{float_prec}f} '
        ws += '.. '
        for w in all_w[-2:]: ws += f'{w:.{float_prec}f} '
    else:
        for w in all_w: ws += f'{w:.{float_prec}f} '
    return f'{ws[:-1]}]'