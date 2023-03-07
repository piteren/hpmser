import os
from ompr.runner import OMPRunner
from pypaq.lipytools.printout import stamp
from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.stats import msmx
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.mpython.devices import DevicesPypaq, get_devices
from pypaq.pms.config_manager import ConfigManager
from pypaq.pms.paspa import PaSpa
from pypaq.pms.base import PSDD, POINT, point_str
import sys, select
import time
from torchness.tbwr import TBwr
from typing import Callable, Optional, List, Union

from hpmser.helpers import str_floatL, HRW
from hpmser.search_results import SRL


NP_SMOOTH = [3,5,9] # numbers of points for smoothing

# initial hpmser configuration
SAMPLING_CONFIG_INITIAL = {
    'np_smooth':    3,      # number of points used for smoothing
    'prob_opt':     0.3,    # probability of sampling from estimated space (optimized)
    'n_opt':        30,     # number of samples taken from estimated space
    'prob_top':     0.3,    # probability of sampling from the area of top points
    'n_top':        20}     # number of samples taken from area of top points

# hpmser configuration updated values - hpmser will be automatically updated after 'config_upd' loops
SAMPLING_CONFIG_UPD = {
    'prob_opt':     0.4,
    'prob_top':     0.6}


# Hyper Parameters Searching Function (based on OMPR engine)
def hpmser(
        func: Callable,                                         # function which parameters need to be optimized
        func_psdd: PSDD,                                        # function parameters space definition dict (PSDD), from here points {param: arg} will be sampled
        func_const: Optional[POINT]=                None,       # func constant kwargs, will be updated with sample (point) taken from PaSpa
        devices: DevicesPypaq=                      None,       # devices to use for search, check pypaq.mpython.devices
        n_loops: Optional[int]=                     2000,       # limit for number of search loops
        update_config: Optional[Union[int,str]]=    'auto',     # update config after n loops
        name: str=                                  'hpmser',   # hpmser run name
        add_stamp=                                  True,       # adds short stamp to name, when name given
        use_GX=                                     True,       # uses genetic xrossing while sampling top points
        distance_L2=                                True,       # use L2(True) or L1(False) for distance calculation
        stochastic_est: Optional[int]=              3,          # number of samples used for stochastic estimation, for 0 or None does not estimate
        plot_axes: Optional[List[str]]=             None,       # preferred axes for plot, put here a list of up to 3 params names ['param1',..]
        top_show_freq=                              20,         # how often top results summary will be printed
        raise_exceptions=                           True,       # forces subprocesses to raise + print exceptions (raising subprocess exception does not break hpmser process)
        hpmser_FD: str=                             '_hpmser',  # save folder
        do_TB=                                      True,       # plots with TB
        logger=                                     None,
        loglevel=                                   20
) -> SRL:

    if add_stamp: name = f'{name}_{stamp(letters=0)}'

    prep_folder(hpmser_FD) # create folder if needed

    # check for continuation
    srl = None
    results_FDL = sorted(os.listdir(hpmser_FD))
    if len(results_FDL):
        print(f'There are {len(results_FDL)} searches in \'{hpmser_FD}\'')
        print('do you want to continue with the last one ({results_FDL[-1]})? .. waiting 10 sec (y/n, n-default)')
        i, o, e = select.select([sys.stdin], [], [], 10)
        if i and sys.stdin.readline().strip() == 'y':
            name = results_FDL[-1]  # take last
            srl = SRL(name=name, logger=logger)
            srl.load(f'{hpmser_FD}/{name}')
            assert PaSpa(psdd=func_psdd, distance_L2=distance_L2) == srl.paspa, 'ERR: parameters space differs - cannot continue!'

    if not logger:
        logger = get_pylogger(
            name=       name,
            folder=     f'{hpmser_FD}/{name}',
            level=      loglevel,
            format=     '%(asctime)s : %(message)s')

    prep_folder(f'{hpmser_FD}/{name}') # needs to be created for ConfigManager
    config_manager = ConfigManager(
        file=           f'{hpmser_FD}/{name}/hpmser.conf',
        config=         SAMPLING_CONFIG_INITIAL,
        try_to_load=    True)
    sampling_config = config_manager.get_config() # update in case of read from existing file

    tbwr = TBwr(logdir=f'{hpmser_FD}/{name}') if do_TB else None

    logger.info(f'*** hpmser : {name} *** started for: {func.__name__}, sampling config: {sampling_config}')
    if srl: logger.info(f'> search will continue with {len(srl)} results..')

    if not srl: srl = SRL(
        paspa=  PaSpa(
            psdd=           func_psdd,
            distance_L2=    distance_L2,
            logger=         get_child(logger=logger, name='paspa', change_level=10)),
        name=   name,
        logger= logger)
    srl.plot_axes = plot_axes

    logger.info(f'\n{srl.paspa}')

    # prepare special points: corners and stochastic
    cpa, cpb = srl.paspa.sample_corners()
    special_points = {0:cpa, 1:cpb}
    if stochastic_est is None: stochastic_est = 0
    stochastic_points = {}
    stochastic_results = []

    avg_dst = srl.get_avg_dst()
    sample_num = len(srl) # number of next sample that will be taken and sent for processing

    curr_max_sr_id = None if not len(srl) else srl.get_top_SR().id  # current max SeRes id
    prev_max_sr_id = None                                           # previous max SeRes id

    scores_all = []

    devices = get_devices(devices=devices, torch_namespace=False) # manage devices
    num_free_rw = len(devices)
    logger.info(f'> hpmser resolved given devices to: {devices}')

    omp = OMPRunner(
        rw_class=               HRW,
        rw_init_kwargs=         {'func': func, 'func_const':func_const},
        rw_lifetime=            1,
        devices=                devices,
        name=                   'OMPR_NB_Hpmser',
        ordered_results=        False,
        log_RWW_exception=      logger.level < 20 or raise_exceptions,
        raise_RWW_exception=    logger.level < 11 or raise_exceptions,
        logger=                 get_child(logger=logger, name='omp', change_level=10)
    )

    if update_config == 'auto':
        update_config = n_loops // 2 if type(n_loops) is int else 1000

    top_time = time.time()
    top_speed_save = []
    logger.info(f'hpmser starts search loop..')
    logger.info(' -- id smooth [local diff_VS_est] topID:dist_to avg_distance/max_of_min_distances time')
    try:
        while True:

            new_sampling_config = config_manager.load()
            if new_sampling_config: sampling_config.update(new_sampling_config)
            if 'np_smooth' in new_sampling_config: srl.set_np_smooth(sampling_config['np_smooth'])

            # use all available devices
            while num_free_rw:
                logger.debug(f' >> got {num_free_rw} free RW at {sample_num} sample_num start')

                if sample_num == update_config:
                    logger.info(f' > updating sampling config..')
                    new_sampling_config = config_manager.update(**SAMPLING_CONFIG_UPD)
                    sampling_config.update(new_sampling_config)

                # fill stochastic points after 50 samples
                if sample_num == 50:
                    _sp = srl.get_top_SR().point                        # try with top
                    if _sp is None: _sp = srl.paspa.sample_point_GX()   # else take random
                    for ix in range(stochastic_est):
                        stochastic_points[sample_num+ix] = _sp

                spoint = None
                est_score = 0

                # use special_points
                if sample_num in special_points:    spoint = special_points[sample_num]
                if sample_num in stochastic_points: spoint = stochastic_points[sample_num]
                # or take point with policy
                if not spoint:
                    params = {
                        'prob_opt': sampling_config['prob_opt'],
                        'n_opt':    sampling_config['n_opt'],
                        'prob_top': sampling_config['prob_top'],
                        'n_top':    sampling_config['n_top'],
                        'avg_dst':  avg_dst}
                    spoint, est_score = srl.get_opt_sample_GX(**params) if use_GX else srl.get_opt_sample(**params)

                task = {
                    'spoint':           spoint,
                    'sample_num':       sample_num,
                    'est_score':        est_score,
                    's_time':           time.time()}

                omp.process(task)
                num_free_rw -= 1
                sample_num += 1

            msg = omp.get_result(block=True) # get one result
            num_free_rw += 1
            if type(msg) is dict: # we may receive str here (like: 'TASK #4 RAISED EXCEPTION') from omp that not restarts exceptions

                msg_sample_num =    msg['sample_num']
                msg_score =         msg['score']
                msg_spoint =        msg['spoint']
                msg_est_score =     msg['est_score']
                msg_s_time =        msg['s_time']

                # manage stochastic estimation without adding to the results
                if msg_sample_num in stochastic_points:
                    stochastic_results.append(msg_score)
                    if len(stochastic_results) == stochastic_est and logger.level < 21:
                        st_nfo = f'\nstochastic estimation with {stochastic_est} points:\n'
                        st_nfo += f'> results: {str_floatL(stochastic_results, float_prec=8)}\n'
                        st_nfo += f'> std_dev: {msmx(stochastic_results)["std"]:.8f}'
                        logger.info(st_nfo)

                else:
                    sr = srl.add_result(
                        point=  msg_spoint,
                        score=  msg_score)
                    logger.debug(f' >> got result #{msg_sample_num}')

                    pf = f'.{srl.prec}f' # update precision of print

                    avg_dst = srl.get_avg_dst()
                    mom_dst = srl.get_mom_dst()
                    srl.save(folder=f'{hpmser_FD}/{name}')

                    top_SR = srl.get_top_SR()

                    # gots new MAX
                    gots_new_max = False
                    if top_SR.id != curr_max_sr_id:
                        prev_max_sr_id = curr_max_sr_id
                        curr_max_sr_id = top_SR.id
                        gots_new_max = True

                    # current sr report
                    if logger.level < 21:

                        dif = sr.smooth_score - msg_est_score
                        difs = f'{"+" if dif>0 else "-"}{abs(dif):{pf}}'

                        dist_to_max = srl.paspa.distance(top_SR.point, sr.point)
                        time_passed = int(time.time() - msg_s_time)

                        srp =  f'{sr.id} {sr.smooth_score:{pf}} [{sr.score:{pf}} {difs}] {top_SR.id}:{dist_to_max:.3f}'
                        srp += f'  avg/mom:{avg_dst:.3f}/{mom_dst:.3f}  {time_passed}s'
                        if new_sampling_config: srp += f'  new sampling config: {sampling_config}'
                        logger.info(srp)

                        # new MAX report
                        if gots_new_max:

                            msr = f'_newMAX: {top_SR.id} {top_SR.smooth_score:{pf}} [{top_SR.score:{pf}}] {point_str(top_SR.point)}\n'

                            prev_sr = srl.get_SR(prev_max_sr_id)
                            dp = srl.paspa.distance(prev_sr.point, top_SR.point) if prev_sr else 0

                            msr += f' dst_prev:{dp:.3f}\n'
                            for nps in NP_SMOOTH:
                                ss_np, avd, all_sc = srl.smooth_point(top_SR, nps)
                                msr += f'  NPS:{nps} {ss_np:{pf}} [{max(all_sc):{pf}}-{min(all_sc):{pf}}] {avd:.3f}\n'
                            logger.info(f'\n{msr}')

                        if top_show_freq and len(srl) % top_show_freq == 0:
                            speed = int((time.time()-top_time) / top_show_freq)
                            top_time = time.time()
                            top_speed_save.append(speed)
                            if len(top_speed_save) > 10: top_speed_save.pop(0)
                            diff = int(speed - (sum(top_speed_save) / len(top_speed_save)))
                            logger.info(f' ### hpmser speed: {speed} sec/task, diff: {"+" if diff >=0 else "-"}{abs(diff)} sec')

                            logger.info(srl.nice_str(n_top=4, top_nps=NP_SMOOTH, all_nps=None))

                    if tbwr:
                        scores_all.append(sr.score)
                        score_diff = sr.score - msg_est_score
                        score_avg = sum(scores_all)/len(scores_all)
                        step = len(srl)
                        tbwr.add(avg_dst,         'hpmser/avg_dst',                    step)
                        tbwr.add(score_avg,       'hpmser/score_avg',                  step)
                        tbwr.add(sr.score,        'hpmser/score_current',              step)
                        tbwr.add(score_diff,      'hpmser/space_estimation_error',     step)
                        tbwr.add(abs(score_diff), 'hpmser/space_estimation_error_abs', step)

                    if len(srl) == n_loops:
                        logger.info(f'{n_loops} loops done!')
                        break

    except KeyboardInterrupt:
        logger.warning(' > hpmser_GX KeyboardInterrupt-ed..')
        raise KeyboardInterrupt # raise exception for OMPRunner

    except Exception as e:
        logger.error(f'hpmser_GX Exception: {str(e)}')
        raise e

    finally:
        omp.exit()

        srl.save(folder=f'{hpmser_FD}/{name}')

        results_str = srl.nice_str(top_nps=NP_SMOOTH)
        if hpmser_FD:
            with open( f'{hpmser_FD}/{name}/{name}_results.txt', 'w') as file: file.write(results_str)
        logger.info(f'\n{results_str}')

        return srl