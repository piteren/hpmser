from ompr.runner import OMPRunner
from pypaq.lipytools.printout import stamp
from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.mpython.devices import DevicesPypaq, get_devices
from pypaq.pms.paspa import PaSpa
from pypaq.pms.base import PSDD, POINT, point_str
from hpmser.points_cloud import PointsCloud, VPoint
from hpmser.space_estimator import SpaceEstimator, RBFRegressor, loss
import random
import time
from torchness.tbwr import TBwr
from typing import Callable, Optional, List, Dict, Tuple

from hpmser.helpers import HRW, fill_up, val_linear



# Hyper Parameters Searching Function (based on OMPR engine)
def hpmser(
        func: Callable,                                     # function which parameters need to be optimized
        func_psdd: PSDD,                                    # function parameters space definition dict (PSDD), from here points {param: arg} will be sampled
        func_const: Optional[POINT]=            None,       # func constant kwargs, will be updated with sample (point) taken from PaSpa
        devices: DevicesPypaq=                  None,       # devices to use for search, check pypaq.mpython.devices
        n_loops: int=                           500,        # number of search loops, should be multiplier of update_estimator_loops
        update_size=                            20,         # frequency of estimator & pcloud update
        time_explore: float=                    0.2,        # factor of loops (from the beginning) with 100% random exploration of space
        time_exploit: float=                    0.2,        # factor of loops (from the end) with 100% exploitation of gained knowledge
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
) -> List[Tuple[VPoint,float]]:

    if add_stamp: name = f'{name}_{stamp()}'

    if not logger:
        logger = get_pylogger(
            name=       name,
            folder=     f'{hpmser_FD}/{name}',
            level=      loglevel,
            format=     '%(asctime)s : %(message)s')

    logger.info(f'*** hpmser : {name} *** started for: {func.__name__}')

    # update n_loops
    if n_loops % update_size != 0:
        n_loops_old = n_loops
        n_loops = (int(n_loops / update_size) + 1) * update_size
        logger.info(f'> updated n_loops from {n_loops_old} to {n_loops}')

    paspa = PaSpa(
        psdd=   func_psdd,
        logger= get_child(logger=logger, name='paspa', change_level=10))
    logger.info(f'\n{paspa}')

    pcloud = PointsCloud(
        paspa=  paspa,
        name=   name,
        logger= logger)

    estimator = estimator_type()

    devices = get_devices(devices=devices, torch_namespace=False) # manage devices
    num_free_rw = len(devices)
    logger.info(f'> hpmser resolved given devices ({len(devices)}): {devices}')

    ompr = OMPRunner(
        rw_class=               HRW,
        rw_init_kwargs=         {'func': func, 'func_const':func_const},
        rw_lifetime=            1,
        devices=                devices,
        name=                   'OMPRunner_hpmser',
        ordered_results=        False,
        log_RWW_exception=      logger.level < 20 or raise_exceptions,
        raise_RWW_exception=    logger.level < 11 or raise_exceptions,
        logger=                 get_child(logger=logger, name='ompr', change_level=10))

    tbwr = TBwr(logdir=f'{hpmser_FD}/{name}') if do_TB else None

    sample_num = len(pcloud)  # number of next sample that will be taken and sent for processing

    points_to_evaluate: List[POINT] = []    # POINTs to be evaluated
    points_at_workers: Dict[int,POINT] = {} # POINTs that are being processed already {sample_num: POINT}
    vpoints_for_update: List[VPoint] = []   # evaluated points stored for next update

    pf = f'.{pcloud.prec}f'  # precision of print

    logger.info(f'hpmser starts search loop ({n_loops})..')
    time_update = time.time()
    time_update_mavg = MovAvg(0.1)
    break_loop = False
    try:
        while True:

            # update cloud, update estimator, prepare report
            if len(vpoints_for_update) == update_size:

                pcloud.update_cloud(vpoints=vpoints_for_update) # add to Cloud
                vpoints_evaluated = pcloud.vpoints
                avg_nearest = pcloud.avg_nearest
                pf = f'.{pcloud.prec}f'  # update precision of print

                if len(pcloud) % (5 * update_size) == 0:
                    pcloud.plot(axes=plot_axes)

                estimator_loss_new = estimator.update_vpoints(vpoints=vpoints_for_update, space=paspa)
                estimation = estimator.predict_vpoints(vpoints=vpoints_evaluated, space=paspa)

                vpoints_estimated = sorted(zip(vpoints_evaluated, estimation), key=lambda x:x[1], reverse=True)
                estimator_loss_all = loss(model=estimator, y_new=[sp.value for sp in vpoints_evaluated], preds=estimation)

                tbwr.add(pcloud.min_nearest, 'hpmser/1.nearest_min', sample_num)
                tbwr.add(avg_nearest,        'hpmser/2.nearest_avg', sample_num)
                tbwr.add(pcloud.max_nearest, 'hpmser/3.nearest_max', sample_num)
                tbwr.add(estimator_loss_all, 'hpmser/4.estimator_loss_all', sample_num)
                tbwr.add(estimator_loss_new, 'hpmser/5.estimator_loss_new', sample_num)

                speed = (time.time() - time_update) / update_size
                time_update = time.time()
                diff = speed - time_update_mavg.upd(speed)
                logger.info(f'___speed: {speed:.1}s/task, diff: {"+" if diff >= 0 else "-"}{abs(diff):.1}s')

                nfo = f'TOP {report_N_top} vpoints by estimate (estimator: {estimator})\n'
                for vpe in vpoints_estimated[:report_N_top]:
                    vp, e = vpe
                    diff = e - vp.value
                    diff_nfo = f'{"+" if diff > 0 else "-"}{abs(diff):{pf}}'
                    est_nfo = f'{e:{pf}} {diff_nfo}'
                    nfo += f'{vp.id:4} {vp.value:{pf}} [{est_nfo}] {point_str(vp.point)}\n'
                logger.info(nfo[:-1])

                # check for main loop break condition
                if len(vpoints_evaluated) >= n_loops:
                    break_loop = True

                vpoints_for_update = []

                # TODO: save here all

            if break_loop: break

            ### prepare points_to_evaluate <- triggered after update, or at first loop

            if not vpoints_for_update:

                s_time = time.time()

                points_to_evaluate = [] # flush if any

                n_needed = update_size + num_free_rw # num points needed to generate

                # add corners
                if len(pcloud) == 0:
                    cpa, cpb = paspa.sample_corners()
                    points_to_evaluate += [cpa, cpb]

                vpoints_evaluated = pcloud.vpoints  # vpoints currently evaluated (all)
                avg_nearest = pcloud.avg_nearest

                points_known = [sp.point for sp in vpoints_evaluated] + list(points_at_workers.values()) # POINTs we already sampled

                estimated_factor = val_linear(sf=time_explore, ef=time_exploit, counter=sample_num, max_count=n_loops,
                    s_val=  0.0,
                    e_val=  1.0)
                num_estimated_points = round(estimated_factor * (n_needed - len(points_to_evaluate))) if estimator.fitted else 0

                avg_nearest_start_factor = val_linear(sf=time_explore, ef=time_exploit, counter=sample_num, max_count=n_loops,
                    s_val=  1.0,
                    e_val=  0.1)
                min_dist = avg_nearest * avg_nearest_start_factor
                while num_estimated_points:

                    points_candidates = [paspa.sample_point() for _ in range(n_needed*10*2)] # TODO *2 for error filter
                    spcL = [VPoint(point=p) for p in points_candidates]
                    est_vpoints_candidates = estimator.predict_vpoints(vpoints=spcL, space=paspa)

                    ce = sorted(zip(points_candidates, est_vpoints_candidates), key=lambda x: x[1], reverse=True)
                    ce = ce[:len(points_candidates) // 2]  # half highest estimate

                    points_candidates = [c[0] for c in ce]

                    # TODO: fiter by error

                    n_added, added_ix = fill_up(
                        fr=         points_candidates,
                        to=         points_to_evaluate,
                        other=      points_known,
                        num=        num_estimated_points,
                        paspa=      paspa,
                        min_dist=   min_dist)
                    print(f'/est added {n_added} {added_ix}/{len(points_candidates)}')

                    num_estimated_points -= n_added
                    min_dist = min_dist * 0.9

                # fill up with random
                min_dist = avg_nearest
                n_addedL = []
                while len(points_to_evaluate) < n_needed:
                    n_added, _ = fill_up(
                        fr=         [paspa.sample_point() for _ in range(n_needed*10)],
                        to=         points_to_evaluate,
                        other=      points_known,
                        num=        n_needed - len(points_to_evaluate),
                        paspa=      paspa,
                        min_dist=   min_dist)
                    min_dist = min_dist * 0.9
                    n_addedL.append(n_added)
                print(f'*randomly added {n_addedL}')

                random.shuffle(points_to_evaluate)
                tbwr.add(time.time()-s_time, 'hpmser/6.sampling_time', sample_num)

            ### run tasks with available devices

            while num_free_rw and points_to_evaluate:
                logger.debug(f'> got {num_free_rw} free RW at {sample_num} sample_num start')

                point = points_to_evaluate.pop()
                task = {
                    'point':        point,
                    'sample_num':   sample_num,
                    's_time':       time.time()}
                points_at_workers[sample_num] = point

                ompr.process(task)
                num_free_rw -= 1
                sample_num += 1

            ### get one result, report

            msg = ompr.get_result(block=True) # get one result
            num_free_rw += 1
            if type(msg) is dict: # str may be received here (like: 'TASK #4 RAISED EXCEPTION') from ompr that does not restart exceptions

                msg_sample_num =    msg['sample_num']
                msg_s_time =        msg['s_time']
                points_at_workers.pop(msg_sample_num)

                vpoint = VPoint(point=msg['point'], value=msg['value'])
                vpoints_for_update.append(vpoint)

                est_nfo = ''
                vpoint_est = estimator.predict_vpoints(vpoints=[vpoint], space=paspa)[0] if estimator.fitted else None
                if vpoint_est is not None:
                    diff = vpoint_est - vpoint.value
                    diff_nfo = f'{"+" if diff>0 else "-"}{abs(diff):{pf}}'
                    est_nfo = f' [{vpoint_est:{pf}} {diff_nfo}]'

                time_taken = time.time() - msg_s_time
                logger.info(f'#{msg_sample_num:4} value: {vpoint.value:{pf}}{est_nfo} ({time_taken:.1f}s) {point_str(vpoint.point)}')

    except KeyboardInterrupt:
        logger.warning(' > hpmser_GX KeyboardInterrupt-ed..')
        raise KeyboardInterrupt # raise exception for OMPRunner

    except Exception as e:
        logger.error(f'hpmser Exception: {str(e)}')
        raise e

    finally:

        """
        srl.save(folder=f'{hpmser_FD}/{name}')

        results_str = srl.nice_str(top_npe=NPE)
        if hpmser_FD:
            with open( f'{hpmser_FD}/{name}/{name}_results.txt', 'w') as file: file.write(results_str)
        logger.info(f'\n{results_str}')
        """

        ompr.exit()

        vpoints_evaluated = pcloud.vpoints
        estimation = estimator.predict_vpoints(vpoints=vpoints_evaluated, space=paspa)
        vpoints_estimated = sorted(zip(vpoints_evaluated, estimation), key=lambda x: x[1], reverse=True)
        return vpoints_estimated