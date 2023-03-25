from ompr.runner import OMPRunner
import os
from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.printout import stamp
from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.plots import three_dim
from pypaq.lipytools.double_hinge import double_hinge
from pypaq.mpython.devices import DevicesPypaq, get_devices
from pypaq.pms.paspa import PaSpa
from pypaq.pms.base import PSDD, POINT, point_str
import random
import select
import sys
import time
from torchness.tbwr import TBwr
from typing import Callable, Optional, List, Dict, Tuple

from hpmser.helpers import HPMSERException, HRW, fill_up, save, load
from hpmser.points_cloud import PointsCloud, VPoint
from hpmser.space_estimator import SpaceEstimator, RBFRegressor, loss


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

        i, o, e = select.select([sys.stdin], [], [], 10)
        if not (i and sys.stdin.readline().strip() == 'y'):
            name_cont = None

    if name_cont:   name = name_cont
    elif add_stamp: name = f'{name}_{stamp()}'

    run_folder = f'{hpmser_FD}/{name}'

    if not logger:
        logger = get_pylogger(
            name=       name,
            folder=     run_folder,
            level=      loglevel,
            format=     '%(asctime)s : %(message)s')

    cont_nfo = ', continuing' if name_cont else ''
    logger.info(f'*** hpmser : {name} *** started for: {func.__name__}{cont_nfo}')

    if name_cont:
        psdd, paspa, pcloud, estimator = load(folder=f'{hpmser_FD}/{name}', logger=logger)
        if psdd != func_psdd:
            raise HPMSERException('parameters space differs - cannot continue!')
    else:
        paspa_logger = get_child(logger=logger, name='paspa', change_level=10)
        paspa = PaSpa(psdd=func_psdd, logger=paspa_logger)
        pcloud = PointsCloud(paspa=paspa, logger=logger)
        estimator = estimator_type()

    # update n_loops
    if n_loops % update_size != 0:
        n_loops_old = n_loops
        n_loops = (int(n_loops / update_size) + 1) * update_size
        logger.info(f'> updated n_loops from {n_loops_old} to {n_loops}')

    logger.info(f'\n{paspa}')

    # estimator plot (test) elements
    test_points = [VPoint(paspa.sample_point()) for _ in range(1000)]
    xyz = [[vp.point[a] for a in plot_axes] for vp in test_points]
    columns = [] + plot_axes
    if len(columns) < 3: columns += ['estimation']

    devices = get_devices(devices=devices, torch_namespace=False) # manage devices
    num_free_rw = len(devices)
    logger.info(f'> hpmser resolved given devices ({len(devices)}): {devices}')

    ompr = OMPRunner(
        rw_class=               HRW,
        rw_init_kwargs=         {'func':func, 'func_const':func_const},
        rw_lifetime=            1,
        devices=                devices,
        name=                   'OMPRunner_hpmser',
        ordered_results=        False,
        log_RWW_exception=      logger.level < 20 or raise_exceptions,
        raise_RWW_exception=    logger.level < 11 or raise_exceptions,
        logger=                 get_child(logger=logger, name='ompr', change_level=10))

    tbwr = TBwr(logdir=run_folder) if do_TB else None

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

            ### update cloud, update estimator, prepare report

            if len(vpoints_for_update) == update_size:

                pcloud.update_cloud(vpoints=vpoints_for_update) # add to Cloud
                vpoints_evaluated = pcloud.vpoints
                avg_nearest = pcloud.avg_nearest
                pf = f'.{pcloud.prec}f'  # update precision of print

                pcloud.plot(
                    name=   'values',
                    axes=   plot_axes,
                    folder= run_folder)

                estimator_loss_new = estimator.update_vpoints(vpoints=vpoints_for_update, space=paspa)
                estimation = estimator.predict_vpoints(vpoints=vpoints_evaluated, space=paspa)

                vpoints_estimated = sorted(zip(vpoints_evaluated, estimation), key=lambda x:x[1], reverse=True)
                estimator_loss_all = loss(model=estimator, y_new=[sp.value for sp in vpoints_evaluated], preds=estimation)

                test_estimation = estimator.predict_vpoints(vpoints=test_points, space=paspa)
                three_dim(
                    xyz=        [v+[e] for v,e in zip(xyz,test_estimation)],
                    name=       'estimator',
                    x_name=     columns[0],
                    y_name=     columns[1],
                    z_name=     columns[2],
                    val_name=   'est',
                    save_FD=    run_folder)

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

                save(
                    psdd=       func_psdd,
                    pcloud=     pcloud,
                    estimator=  estimator,
                    folder=     run_folder)

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

                estimated_factor = double_hinge(sf=time_explore, ef=time_exploit, counter=sample_num, max_count=n_loops,
                                                s_val=  0.0,
                                                e_val=  1.0)
                num_estimated_points = round(estimated_factor * (n_needed - len(points_to_evaluate))) if estimator.fitted else 0

                avg_nearest_start_factor = double_hinge(sf=time_explore, ef=time_exploit, counter=sample_num, max_count=n_loops,
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

        save(
            psdd=       func_psdd,
            pcloud=     pcloud,
            estimator=  estimator,
            folder=     run_folder)

        ompr.exit()

        vpoints_evaluated = pcloud.vpoints
        estimation = estimator.predict_vpoints(vpoints=vpoints_evaluated, space=paspa) if estimator.fitted else [0]*len(vpoints_evaluated)
        vpoints_estimated = sorted(zip(vpoints_evaluated, estimation), key=lambda x: x[1], reverse=True)

        return vpoints_estimated