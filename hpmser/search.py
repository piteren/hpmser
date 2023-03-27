import numpy as np
from ompr.runner import OMPRunner
import os
from pypaq.lipytools.files import prep_folder, w_pickle, r_pickle
from pypaq.lipytools.printout import stamp
from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.plots import three_dim
from pypaq.lipytools.double_hinge import double_hinge
from pypaq.lipytools.stats import mam
from pypaq.mpython.devices import DevicesPypaq, get_devices
from pypaq.pms.paspa import PaSpa
from pypaq.pms.base import PSDD, POINT, point_str
import random
import select
import sys
import time
from torchness.tbwr import TBwr
from typing import Callable, Optional, List, Dict, Tuple

from hpmser.running_worker import HRW
from hpmser.points_cloud import PointsCloud, VPoint
from hpmser.space_estimator import SpaceEstimator, RBFRegressor, loss



class HPMSERException(Exception):
    pass


# Hyper Parameters Search, runs after init
class HPMSer:

    def __init__(
            self,
            func: Callable,                                     # function which parameters need to be optimized
            func_psdd: PSDD,                                    # function parameters space definition dict (PSDD), from here points {param: arg} will be sampled
            func_const: Optional[POINT]=            None,       # func constant kwargs, will be updated with sample (point) taken from PaSpa
            devices: DevicesPypaq=                  None,       # devices to use for search, check pypaq.mpython.devices
            n_loops: int=                           500,        # number of search loops, should be multiplier of update_estimator_loops
            update_size=                            20,         # frequency of estimator & pcloud update
            explore: float=                         0.2,        # factor of loops (from the beginning) with 100% random exploration of space
            exploit: float=                         0.2,        # factor of loops (from the end) with 100% exploitation of gained knowledge
            plot_axes: Optional[List[str]]=         None,       # preferred axes for plot, put here a list of up to 3 params names ['param1',..]
            name: str=                              'hpmser',   # hpmser run name
            add_stamp=                              True,       # adds short stamp to name, when name given
            estimator_type: type(SpaceEstimator)=   RBFRegressor,
            raise_exceptions=                       True,       # forces subprocesses to raise + print exceptions (raising subprocess exception does not break hpmser process)
            hpmser_FD: str=                         '_hpmser',  # save folder
            report_N_top=                           5,          # N top VPoints to report
            do_TB=                                  True,       # plots hpmser stats with TB
            logger=                                 None,
            loglevel=                               20,
    ):

        self.func = func
        self.func_psdd = func_psdd
        self.func_const = func_const

        prep_folder(hpmser_FD)

        ### check for continuation

        name_cont = None

        results_FDL = sorted(os.listdir(hpmser_FD))
        old_results = []
        for f in results_FDL:
            if 'hpmser.save' in os.listdir(f'{hpmser_FD}/{f}'):
                old_results.append(f)

        if len(old_results):

            name_cont = old_results[-1]  # take last
            print(f'There are {len(old_results)} old searches in \'{hpmser_FD}\' folder')
            print(f'do you want to continue with the last one: {name_cont} ? .. waiting 10 sec (y/n, n-default)')

            i, o, est = select.select([sys.stdin], [], [], 10)
            if not (i and sys.stdin.readline().strip() == 'y'):
                name_cont = None

        if name_cont:   self.name = name_cont
        elif add_stamp: self.name = f'{name}_{stamp()}'

        self.run_folder = f'{hpmser_FD}/{self.name}'

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                folder=     self.run_folder,
                level=      loglevel,
                format=     ('%(asctime)s : %(message)s',"%Y-%m-%d %H:%M:%S"))
        self.logger = logger

        cont_nfo = ', continuing' if name_cont else ''
        self.logger.info(f'*** hpmser : {self.name} *** started for: {func.__name__}{cont_nfo}')

        paspa_logger = get_child(logger=self.logger, name='paspa', change_level=10)
        self.paspa = PaSpa(psdd=self.func_psdd, logger=paspa_logger)
        self.logger.info(f'\n{self.paspa}')

        self.estimator_type = estimator_type

        # load or create
        if name_cont:
            self.pcloud, self.estimator = self._load()
        else:
            self.pcloud = PointsCloud(paspa=self.paspa, logger=self.logger)
            self.estimator = estimator_type()

        devices = get_devices(devices=devices, torch_namespace=False) # manage devices
        self.logger.info(f'> hpmser resolved given devices ({len(devices)}): {devices}')

        self.ompr = OMPRunner(
            rw_class=               HRW,
            rw_init_kwargs=         {'func':func, 'func_const':func_const},
            rw_lifetime=            1,
            devices=                devices,
            name=                   'OMPRunner_hpmser',
            ordered_results=        False,
            log_RWW_exception=      self.logger.level < 20 or raise_exceptions,
            raise_RWW_exception=    self.logger.level < 11 or raise_exceptions,
            logger=                 get_child(logger=self.logger, name='ompr', change_level=10))

        self.tbwr = TBwr(logdir=self.run_folder) if do_TB else None

        # update n_loops to be multiplier of update_size
        if n_loops % update_size != 0:
            n_loops_old = n_loops
            n_loops = (int(n_loops / update_size) + 1) * update_size
            self.logger.info(f'> updated n_loops from {n_loops_old} to {n_loops} (to be multiplier of update_size)')

        # handle axes
        if plot_axes:
            not_in_psdd = [a for a in plot_axes if a not in func_psdd]
            if not_in_psdd:
                raise HPMSERException(f'given plot_axes not present in psdd: {not_in_psdd}, cannot continue')
        else:
            plot_axes = list(func_psdd.keys())[:3]

        self.run_results = self._run(
            n_loops=        n_loops,
            update_size=    update_size,
            n_devices=      len(devices),
            explore=        explore,
            exploit=        exploit,
            report_N_top=   report_N_top,
            plot_axes=      plot_axes)

    # runs search loop
    def _run(
            self,
            n_loops,
            update_size,
            n_devices,
            explore,
            exploit,
            report_N_top,
            plot_axes) -> List[Tuple[VPoint, Optional[float]]]:

        points_to_evaluate: List[POINT] = []  # POINTs to be evaluated
        points_at_workers: Dict[int, POINT] = {}  # POINTs that are being processed already {sample_num: POINT}
        vpoints_for_update: List[VPoint] = []  # evaluated points stored for next update

        num_free_rw = n_devices

        # estimator plot (test) elements
        test_points = [VPoint(self.paspa.sample_point()) for _ in range(1000)]
        xyz = [[vp.point[a] for a in plot_axes] for vp in test_points]
        columns = [] + plot_axes
        if len(columns) < 3: columns += ['estimation']

        sample_num = len(self.pcloud)  # number of next sample that will be taken and sent for processing

        self.logger.info(f'hpmser starts search loop ({n_loops})..')
        time_update = time.time()
        time_update_mavg = MovAvg(0.1)
        break_loop = False
        try:
            while True:

                ### update cloud, update estimator, prepare report

                if len(vpoints_for_update) == update_size:

                    self.pcloud.update_cloud(vpoints=vpoints_for_update) # add to Cloud
                    vpoints_evaluated = self.pcloud.vpoints

                    self.pcloud.plot(
                        name=   f'VALUES_{self.name}',
                        axes=   plot_axes,
                        folder= self.run_folder)

                    estimator_loss_new = self.estimator.update_vpoints(vpoints=vpoints_for_update, space=self.paspa)
                    estimation = self._estimate(vpoints=vpoints_evaluated)
                    vpoints_estimated = sorted(zip(vpoints_evaluated, estimation), key=lambda x:x[1], reverse=True)
                    estimator_loss_all = loss(model=self.estimator, y_new=[sp.value for sp in vpoints_evaluated], preds=estimation)

                    test_estimation = self._estimate(vpoints=test_points)
                    three_dim(
                        xyz=        [v+[e] for v,e in zip(xyz,test_estimation)],
                        name=       f'ESTIMATOR_{self.name}',
                        x_name=     columns[0],
                        y_name=     columns[1],
                        z_name=     columns[2],
                        val_name=   'est',
                        save_FD=    self.run_folder)

                    speed = (time.time() - time_update) / update_size
                    time_update = time.time()
                    diff = speed - time_update_mavg.upd(speed)
                    self.logger.info(f'___speed: {speed:.1f}s/task, diff: {"+" if diff >= 0 else "-"}{abs(diff):.1f}s')

                    if self.tbwr:

                        values = [vp.value for vp in vpoints_evaluated]
                        vmin, vavg, vmax = mam(values)
                        self.tbwr.add(vmin,                     'cloud/1.value_min',        sample_num)
                        self.tbwr.add(vavg,                     'cloud/2.value_avg',        sample_num)
                        self.tbwr.add(vmax,                     'cloud/3.value_max',        sample_num)
                        self.tbwr.add_histogram(values,         'values',                   sample_num)

                        self.tbwr.add(self.pcloud.min_nearest,  'cloud/4.nearest_dst_min',  sample_num)
                        self.tbwr.add(self.pcloud.avg_nearest,  'cloud/5.nearest_dst_avg',  sample_num)
                        self.tbwr.add(self.pcloud.max_nearest,  'cloud/6.nearest_dst_max',  sample_num)

                        emin, eavg, emax = mam(estimation)
                        self.tbwr.add(emin,                     'estimator/1.estimation_min',       sample_num)
                        self.tbwr.add(eavg,                     'estimator/2.estimation_avg',       sample_num)
                        self.tbwr.add(emax,                     'estimator/3.estimation_max',       sample_num)
                        self.tbwr.add_histogram(estimation,     'estimation',                       sample_num)

                        emin, eavg, emax = mam(test_estimation)
                        self.tbwr.add(emin,                     'estimator/4.test_estimation_min',  sample_num)
                        self.tbwr.add(eavg,                     'estimator/5.test_estimation_avg',  sample_num)
                        self.tbwr.add(emax,                     'estimator/6.test_estimation_max',  sample_num)
                        self.tbwr.add_histogram(test_estimation,'test_estimation',                  sample_num)

                        self.tbwr.add(estimator_loss_all,       'estimator/7.loss_all',             sample_num)
                        self.tbwr.add(estimator_loss_new,       'estimator/8.loss_new',             sample_num)

                        self.tbwr.add(speed, 'process/speed_s/task', sample_num)

                    nfo = f'TOP {report_N_top} VPoints by estimate (estimator: {self.estimator})\n'
                    for vpe in vpoints_estimated[:report_N_top]:
                        nfo += f'{self._vpoint_nfo(*vpe)}\n'
                    self.logger.info(nfo[:-1])

                    # check for main loop break condition
                    if len(vpoints_evaluated) >= n_loops:
                        break_loop = True

                    vpoints_for_update = []

                    self._save()

                if break_loop: break

                ### prepare points_to_evaluate <- triggered after update, or at first loop

                if not vpoints_for_update:

                    s_time = time.time()

                    points_to_evaluate = [] # flush if any

                    n_needed = update_size + num_free_rw # num points needed to generate

                    # add corners
                    if len(self.pcloud) == 0:
                        cpa, cpb = self.paspa.sample_corners()
                        points_to_evaluate += [cpa, cpb]

                    vpoints_evaluated = self.pcloud.vpoints  # vpoints currently evaluated (all)

                    points_known = [sp.point for sp in vpoints_evaluated] + list(points_at_workers.values()) # POINTs we already sampled

                    estimated_factor = double_hinge(
                        sf=explore, ef=exploit, counter=sample_num, max_count=n_loops,
                        s_val=  0.0,
                        e_val=  1.0)
                    num_estimated_points = round(estimated_factor * (n_needed - len(points_to_evaluate))) if self.estimator.fitted else 0

                    avg_nearest_start_factor = double_hinge(
                        sf=explore, ef=exploit, counter=sample_num, max_count=n_loops,
                        s_val=  1.0,
                        e_val=  0.1)
                    min_dist = self.pcloud.avg_nearest * avg_nearest_start_factor
                    while num_estimated_points:

                        points_candidates = [self.paspa.sample_point() for _ in range(n_needed*10*2)]
                        spcL = [VPoint(point=p) for p in points_candidates]
                        est_vpoints_candidates = self._estimate(vpoints=spcL)

                        ce = sorted(zip(points_candidates, est_vpoints_candidates), key=lambda x: x[1], reverse=True)
                        ce = ce[:len(points_candidates) // 2]  # half highest estimate

                        points_candidates = [c[0] for c in ce]

                        n_added, added_ix = self._fill_up(
                            fr=         points_candidates,
                            to=         points_to_evaluate,
                            other=      points_known,
                            num=        num_estimated_points,
                            min_dist=   min_dist)
                        self.logger.info(f' /// est added {n_added} {added_ix}/{len(points_candidates)}')

                        num_estimated_points -= n_added
                        min_dist = min_dist * 0.9

                    # fill up with random
                    min_dist = self.pcloud.avg_nearest
                    n_addedL = []
                    while len(points_to_evaluate) < n_needed:
                        n_added, _ = self._fill_up(
                            fr=         [self.paspa.sample_point() for _ in range(n_needed*10)],
                            to=         points_to_evaluate,
                            other=      points_known,
                            num=        n_needed - len(points_to_evaluate),
                            min_dist=   min_dist)
                        min_dist = min_dist * 0.9
                        n_addedL.append(n_added)
                    self.logger.info(f' *** randomly added {n_addedL}')

                    random.shuffle(points_to_evaluate)
                    if self.tbwr:
                        self.tbwr.add(time.time()-s_time, 'process/sampling_time', sample_num)

                ### run tasks with available devices

                while num_free_rw and points_to_evaluate:

                    point = points_to_evaluate.pop(0)
                    task = {
                        'point':        point,
                        'sample_num':   sample_num,
                        's_time':       time.time()}
                    points_at_workers[sample_num] = point

                    self.ompr.process(task)
                    num_free_rw -= 1
                    sample_num += 1

                ### get one result, report

                msg = self.ompr.get_result(block=True) # get one result
                num_free_rw += 1
                if type(msg) is dict: # str may be received here (like: 'exception while processing task XX: xxx') from ompr that does not restart exceptions

                    msg_sample_num =    msg['sample_num']
                    msg_s_time =        msg['s_time']
                    points_at_workers.pop(msg_sample_num)

                    vpoint = VPoint(point=msg['point'], value=msg['value'])
                    vpoints_for_update.append(vpoint)

                    time_taken = time.time() - msg_s_time
                    estimation = self._estimate(vpoints=[vpoint])
                    if estimation is not None: estimation = estimation[0]
                    self.logger.info(f'{self._vpoint_nfo(vpoint=vpoint, estimation=estimation)} ({time_taken:.1f}s)')

        except KeyboardInterrupt:
            self.logger.warning(' > hpmser_GX KeyboardInterrupt-ed..')
            raise KeyboardInterrupt # raise exception for OMPRunner
        
        finally:

            self._save()
            self.ompr.exit()

            vpoints_evaluated = self.pcloud.vpoints
            estimation = self._estimate(vpoints=vpoints_evaluated)
            if estimation is None:
                estimation = [None]*len(vpoints_evaluated)
            vpoints_estimated = sorted(zip(vpoints_evaluated, estimation), key=lambda x: x[1], reverse=True)

            self.logger.info(f'hpmser {self.name} finished, exits.')

            return vpoints_estimated


    # prepares estimation for Valued Points
    def _estimate(self, vpoints:List[VPoint]) -> Optional[np.ndarray]:
        if self.estimator.fitted:
            return self.estimator.predict_vpoints(vpoints=vpoints, space=self.paspa)
        return None

    # appends POINTs fr >> to until given size reached, POINT cannot be closer than min_dist to any from to+other
    def _fill_up(
            self,
            fr: List[POINT],
            to: List[POINT],
            other: List[POINT],
            num: int,
            min_dist: float,
    ) -> Tuple[int, List[int]]:
        n_added = 0
        added_ix = []
        for ix,pc in enumerate(fr):

            candidate_ok = True
            for pr in to + other:
                if self.paspa.distance(pr, pc) < min_dist:
                    candidate_ok = False
                    break

            if candidate_ok:
                to.append(pc)
                n_added += 1
                added_ix.append(ix)

            if n_added == num: break

        return n_added, added_ix

    # prepares nice str about VPoint
    def _vpoint_nfo(self, vpoint:VPoint, estimation:Optional[float]=None) -> str:

        prec = f'.{self.pcloud.prec}f'  # precision of print

        id_nfo = f'#{vpoint.id:4}' if vpoint.id is not None else '_'
        est_nfo = ''
        diff_nfo = ''
        if estimation is not None:
            est_nfo = f' est: {estimation:{prec}}'
            diff = vpoint.value - estimation
            diff_nfo = f' {"+" if diff > 0 else "-"}{abs(diff):{prec}}'
        return f'{id_nfo}{est_nfo} [val: {vpoint.value:{prec}}{diff_nfo}] {point_str(vpoint.point)}'

    # saves session
    def _save(self):
        data = {
            'func':             self.func,
            'func_psdd':        self.func_psdd,
            'func_const':       self.func_const,
            'vpoints':          self.pcloud.vpoints,
            'estimator_type':   type(self.estimator),
            'estimator_state':  self.estimator.state}
        w_pickle(data, f'{self.run_folder}/hpmser.save')
        w_pickle(data, f'{self.run_folder}/hpmser.save.back')

    # loads session data
    def _load(self) -> Tuple[PointsCloud, SpaceEstimator]:

        try:
            data = r_pickle(f'{self.run_folder}/hpmser.save')
        except Exception as e:
            self.logger.warning(f'got exception: {e} while loading hpmser, using backup file')
            data = r_pickle(f'{self.run_folder}/hpmser.save.back')

        # compatibility check
        pairs = {
            'func':             (data['func'],           self.func),
            'func_psdd':        (data['func_psdd'],      self.func_psdd),
            'func_const':       (data['func_const'],     self.func_const),
            'estimator_type':   (data['estimator_type'], self.estimator_type)}
        not_match = []
        for k in pairs:
            if pairs[k][0] != pairs[k][1]:
                not_match.append(k)
        if not_match:
            nfo = f'components of hpmser are not compatible:\n'
            for nmk in not_match:
                nfo += f'{nmk}:\n> saved: {pairs[nmk][0]}\n> given: {pairs[nmk][1]}\n'
            self.logger.error(nfo)
            raise HPMSERException(nfo)

        pcloud = PointsCloud(
            paspa=  self.paspa,
            logger= self.logger)

        estimator = data['estimator_type'].from_state(state=data['estimator_state'])

        # update objects 'states'
        if data['vpoints']:
            pcloud.update_cloud(vpoints=data['vpoints'])
            estimator.update_vpoints(vpoints=data['vpoints'], space=self.paspa)

        self.logger.info(f'hpmser loaded session from {self.run_folder} with {len(data["vpoints"])} results')

        return pcloud, estimator

    @property
    def results(self) -> List[Tuple[VPoint, Optional[float]]]:
        return self.run_results