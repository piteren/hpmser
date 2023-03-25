from ompr.runner import RunningWorker
from pypaq.mpython.devices import DevicesPypaq
from pypaq.pms.paspa import PaSpa
from pypaq.pms.base import POINT, PSDD, get_params
from pypaq.lipytools.files import w_pickle, r_pickle
from pypaq.lipytools.pylogger import get_child
from typing import Callable, Optional, List, Any, Tuple

from hpmser.points_cloud import PointsCloud
from hpmser.space_estimator import SpaceEstimator



class HPMSERException(Exception):
    pass


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

    # processes given point - computes value
    def process(
            self,
            point: POINT,
            **kwargs) -> Any:

        point_with_defaults = {}
        point_with_defaults.update(self.func_const)
        point_with_defaults.update(point)

        res = self.func(**point_with_defaults)
        if type(res) is dict: value = res['value']
        else:                 value = res

        msg = {'point':point, 'value':value}
        msg.update(kwargs)
        return msg

# saves hpmser session
def save(
        psdd: PSDD,
        pcloud: PointsCloud,
        estimator: SpaceEstimator,
        folder: str):
    data = {
        'psdd':             psdd,
        'vpoints':          pcloud.vpoints,
        'estimator_type':   type(estimator),
        'estimator_state':  estimator.state}
    w_pickle(data, f'{folder}/hpmser.save')
    w_pickle(data, f'{folder}/hpmser.save.back')

# loads hpmser session
def load(folder:str, logger) -> Tuple[PSDD, PaSpa, PointsCloud, SpaceEstimator]:

    try:
        data = r_pickle(f'{folder}/hpmser.save')
    except Exception as e:
        logger.warning(f'got exception: {e} while loading hpmser, using backup file')
        data = r_pickle(f'{folder}/hpmser.save.back')

    psdd = data['psdd']

    paspa = PaSpa(
        psdd=   psdd,
        logger= get_child(logger=logger, name='paspa', change_level=10))

    pcloud = PointsCloud(
        paspa=  paspa,
        logger= logger)
    pcloud.update_cloud(vpoints=data['vpoints'])

    estimator = data['estimator_type'].from_state(state=data['estimator_state'])
    estimator.update_vpoints(vpoints=data['vpoints'], space=paspa)

    return psdd, paspa, pcloud, estimator

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

# appends POINTs fr >> to until given size reached, POINT cannot be closer than min_dist to any from to+other
def fill_up(
        fr: List[POINT],
        to: List[POINT],
        other: List[POINT],
        num: int,
        paspa: PaSpa,
        min_dist: float,
) -> Tuple[int,List[int]]:

    n_added = 0
    added_ix = []
    for ix,pc in enumerate(fr):

        candidate_ok = True
        for pr in to + other:
            if paspa.distance(pr, pc) < min_dist:
                candidate_ok = False
                break

        if candidate_ok:
            to.append(pc)
            n_added += 1
            added_ix.append(ix)

        if n_added == num: break

    return n_added, added_ix