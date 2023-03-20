from pypaq.lipytools.files import r_pickle, w_pickle
from pypaq.lipytools.plots import three_dim
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.pms.paspa import PaSpa
from pypaq.pms.base import POINT, point_str
import random
from typing import Sized, List, Tuple, Optional, Iterable

from hpmser.helpers import str_floatL


# Search Results List [SeRes] with some methods
class SRL(Sized):

    # single Search Result
    class SeRes:

        def __init__(
                self,
                point: POINT,
                score: Optional[float]= None # score of SeRes
        ):
            self.id: Optional[int] = None           # to be updated by SRL (id = len(SRL))
            self.point = point
            self.score = score
            self.estimate: Optional[float] = None   # to be updated by SRL

        def __str__(self):
            return f'SeRes: id:{self.id}, point:{self.point}, score:{self.score}, estimate:{self.estimate}'


    def __init__(
            self,
            paspa: Optional[PaSpa]=     None,   # parameters space of this SRL
            name: str=                  'SRL',
            npe: int=                   3,      # (NPE) Number of Points taken into account while calculating Estimate
            plot_axes: list=            None,   # list with axes names (max 3), eg: ['drop_a','drop_b','loss']
            logger=                     None,
            loglevel=                   30,
    ):

        self.name = name

        if not logger:
            logger = get_pylogger(level=loglevel)
        self.logger = logger
        self.logger.info(f'*** SRL : {self.name} *** initializing..')

        self.paspa = paspa

        self._npe = npe        # current NPE of SRL
        self.plot_axes = plot_axes


        self._srL: List[SRL.SeRes] = []    # sorted periodically by estimate
        self._sorted_and_estmated = True   # SRL status: is sorted & estimated
        self._distances = []               # distances cache (by SeRes id)
        self._scores = []                  # scores cache (by SeRes id)
        self._avg_dst = 1                  # average distance of SRL for self._npe
        self._prec = 8                     # print precision, will be updated while adding new points

    # ****************************************************************************************************** load & save

    def _get_srl_path(self, save_dir: str) -> str:
        return f'{save_dir}/{self.name}.srl'


    def _get_srl_backup_path(self, save_dir: str) -> str:
        return f'{save_dir}/{self.name}.srl.backup'

    # loads (alternatively from backup)
    def load(self, save_dir :str):

        self.logger.info(f' > SRL {self.name} loading form {save_dir}..')

        try:
            obj = r_pickle(self._get_srl_path(save_dir))
        except Exception as e:
            self.logger.warning(f' SRL {self.name} got exception: {str(e)} while loading, using backup file')
            obj = r_pickle(self._get_srl_backup_path(save_dir))

        self.paspa =                obj.paspa
        self._npe =                 obj._npe
        self.plot_axes =            obj.plot_axes

        self._srL =                 obj._srL
        self._sorted_and_estmated = obj._sorted_and_estmated
        self._distances =           obj._distances
        self._scores =              obj._scores
        self._avg_dst =             obj._avg_dst
        self._prec=                 obj._prec

        self.logger.info(f' > SRL loaded {len(self._srL)} results')

    # saves with backup
    def save(self, folder :str):

        # backup copy previous
        old_res = r_pickle(self._get_srl_path(folder))
        if old_res: w_pickle(old_res, self._get_srl_backup_path(folder))

        w_pickle(self, self._get_srl_path(folder))
        self.plot(folder=folder)

    # ************************************************************************************************ getters & setters

    # returns top SeRes (max estimate)
    def get_top_SR(self) -> SeRes or None:
        if self._srL:
            if not self._sorted_and_estmated:
                self.logger.warning('SRL asked to return top SR while SRL is not smoothed_and_sorted, running smooth_and_sort()')
                self.smooth_and_sort()
            return self._srL[0]
        return None


    def get_SR(self, id: int) -> SeRes or None:
        for sr in self._srL:
            if sr.id == id:
                return sr
        return None

    # returns distance between two points, if points are SeRes type uses cached distance
    def get_distance(self,
            pa: POINT or SeRes,
            pb: POINT or SeRes) -> float:
        if type(pa) is SRL.SeRes and type(pb) is SRL.SeRes:
            return self._distances[pa.id][pb.id]
        if type(pa) is SRL.SeRes: pa = pa.point
        if type(pb) is SRL.SeRes: pb = pb.point
        return self.paspa.distance(pa, pb)


    def get_avg_dst(self):
        return self._avg_dst

    # max of min distances of SRL: max(min_distance)
    def get_mom_dst(self):
        return max([min(d[1:]) if d[1:] else 0 for d in self._distances])

    # returns sample with policy and estimated score
    def get_opt_sample(self,
            prob_opt,   # probability of optimized sample
            n_opt,      # number of optimized samples
            prob_top,   # probability of sample from area of top
            n_top,      # number of top samples
            avg_dst     # distance for sample from area of top
    ) -> Tuple[POINT,float]:

        prob_rnd = 1 - prob_opt - prob_top
        if random.random() < prob_rnd or len(self._srL) < 10: sample = self.paspa.sample_point_GX() # one random point
        else:
            if random.random() < prob_opt/(prob_top+prob_opt): points = [self.paspa.sample_point_GX() for _ in range(n_opt + 1)] # some random points ...last for reference
            # top points
            else:
                n_top += 1 # last for reference
                if n_top > len(self._srL): n_top = len(self._srL)
                points = [self.paspa.sample_point_GX(
                    point_main=     self._srL[ix].point,
                    noise_scale=    avg_dst) for ix in range(n_top)] # top points

            scores = [self.smooth_point(p)[0] for p in points]

            all_pw = list(zip(points, scores))
            all_pw.sort(key=lambda x: x[1], reverse=True)
            maxs = all_pw[0][1]
            subs = all_pw.pop(-1)[1]
            mins = all_pw[-1][1]

            all_p, all_w = zip(*all_pw)
            all_w = [w - subs for w in all_w]
            all_p = list(all_p)
            sample = random.choices(all_p, weights=all_w, k=1)[0]
            pf = f'.{self._prec}f'
            #print(f'   % sampled #{all_p.index(sample)}/{len(all_p)} from: {maxs:{pf}}-{mins:{pf}} {str_floatL(all_w, float_prec=self._prec)}')

        est_score, _, _ =  self.smooth_point(sample)

        return sample, est_score

    # returns sample chosen with policy and its estimated score
    def get_opt_sample_GX(
            self,
            prob_opt,   # probability of optimized sample
            n_opt,      # number of optimized samples
            prob_top,   # probability of sample from area of top
            n_top,      # number of top samples
            avg_dst     # distance for sample from area of top
    ) -> Tuple[POINT,float]:

        if random.random() < 1-prob_opt-prob_top or len(self._srL) < 10:
            sample = self.paspa.sample_point_GX() # one random point
        else:
            # choice from better half of random points
            if random.random() < prob_opt/(prob_top+prob_opt):
                points = [self.paspa.sample_point_GX() for _ in range(2 * n_opt)]   # 2*some random points
                scores = [self.smooth_point(p)[0] for p in points]                  # their scores
                all_pw = list(zip(points, scores))
                all_pw.sort(key=lambda x: x[1], reverse=True)                       # sorted
                all_pw = all_pw[:len(all_pw)//2]                                    # take better half
                maxs = all_pw[0][1]
                subs = all_pw.pop(-1)[1]
                mins = all_pw[-1][1]
                all_p, all_w = zip(*all_pw)
                all_w = [w - subs for w in all_w]
                all_p = list(all_p)
                sample = random.choices(all_p, weights=all_w, k=1)[0]
                pf = f'.{self._prec}f'
                self.logger.debug(f'   % sampled #{all_p.index(sample)}/{len(all_p)} from: {maxs:{pf}}-{mins:{pf}} {str_floatL(all_w, float_prec=self._prec)}')
            # GX from top points
            else:
                n_top += 1 # last for reference
                if n_top > len(self._srL): n_top = len(self._srL)
                top_sr = self._srL[:n_top]                             # top SR
                scores = [p.estimate for p in top_sr]               # their scores
                mins = min(scores)
                scores = [s-mins for s in scores]                       # reduced scores
                sra, srb = random.choices(top_sr, weights=scores, k=2)  # two SR
                sample = self.paspa.sample_point_GX(
                    point_main=                 sra.point,
                    point_scnd=                 srb.point,
                    noise_scale=                avg_dst)
                pf = f'.{self._prec}f'
                self.logger.debug(f'   % sampled GX from: {sra.estimate:{pf}} and {srb.estimate:{pf}}')
        est_score, _, _ =  self.smooth_point(sample)
        return sample, est_score

    # returns n closest SeRes to given point
    def _get_n_closest(
            self,
            point: POINT or SeRes,
            n: Optional[int]=   None) -> List[SeRes]:
        if not n: n = self._npe
        if len(self._srL) <= n:
            return [] + self._srL
        else:
            id_dst = \
                list(zip(range(len(self._srL)), self._distances[point.id])) \
                    if type(point) is SRL.SeRes else \
                    [(sr.id, self.get_distance(point, sr.point)) for sr in self._srL]
            id_dst.sort(key=lambda x: x[1]) # sort by distance to this point
            return [self.get_SR(id[0]) for id in id_dst[:n]]

    # sets npe, then smooths and sorts
    def set_npe(self, npe: int):
        if npe != self._npe:
            self._npe = npe
            self.smooth_and_sort()

    # adds new result, caches distances, smooths & sorts
    def add_result(self,
            point: POINT,
            score: float,
            force_no_update=    False # aborts: calculation of estimate & sorting
    ) -> SeRes:

        sr = SRL.SeRes(point=point, score=score)
        sr.id = len(self._srL)

        # update cached distances
        sr_dist = []
        id_point = [(s.id,s.point) for s in self._srL]
        id_point.sort(key= lambda x:x[0]) # sort by id
        for id,point in id_point:
            d = self.paspa.distance(sr.point,point)
            sr_dist.append(d)
            self._distances[id].append(d)
        sr_dist.append(0)

        self._distances.append(sr_dist)

        self._scores.append(score) # update cached score

        self._srL.append(sr) # add SeRes

        if force_no_update:
            sr.estimate = sr.score
            self._sorted_and_estmated = False
        else: self.smooth_and_sort()

        if score > 0.01: self._prec = 4

        return sr

    # returns (estimate, average_distance, all scores sorted by distance) for given point or SeRes
    def smooth_point(self,
            point: POINT or SeRes,
            npe: Optional[int]=   None # to override self._npe 
    ) -> Tuple[float, float, List[float]]:

        # case: no points in srL
        estimate_np = 0     # smooth score for self._npe
        avg_dst_np =      1     # average distance for npe
        all_scores =     [0]    # np scores sorted by distance

        if not npe: npe = self._npe
        if self._srL:
            score_dst = \
                list(zip(self._scores, self._distances[point.id])) \
                    if type(point) is SRL.SeRes else \
                    [(sr.score, self.get_distance(point, sr.point)) for sr in self._srL]
            score_dst.sort(key=lambda x: x[1])  # sort by distance to this point
            score_dst_np = score_dst[:npe + 1] # trim to npe points (+1 point for reference)

            # case of one/two points in score_dst_np
            if len(score_dst_np) < 3:
                estimate_np = score_dst_np[0][0]  # closest point score
                all_scores = [score_dst_np[0][0]]
            else:
                all_scores, all_dst = zip(*score_dst_np)  # scores, distances

                max_dst = all_dst[-1]  # distance of last(reference) point

                # remove last (reference)
                all_dst = all_dst[:-1]
                all_scores = all_scores[:-1]

                # set weights for scores
                weights = []
                if max_dst: weights = [(max_dst-d)/max_dst for d in all_dst]  # try with distance based weights <1;0>
                if sum(weights) == 0: weights = [1] * len(all_dst)  # naive baseline / equal for case: (max_dst == 0) or (max_dst-d == 0)

                wall_scores = [all_scores[ix]*weights[ix] for ix in range(len(all_scores))]
                estimate_np = sum(wall_scores) / sum(weights) # weighted score
                avg_dst_np = sum(all_dst) / len(all_dst)

        return estimate_np, avg_dst_np, all_scores

    # sorts srL by estimate
    def _sort(self):
        self._srL.sort(key=lambda x:x.estimate, reverse=True)

    # smooths self._srL and sorts by SeRes.estimate
    def smooth_and_sort(self):

        if self._srL:
            avg_dst = []
            for sr in self._srL:
                sr.estimate, ad, _ = self.smooth_point(sr)
                avg_dst.append(ad)
            self._avg_dst = sum(avg_dst)/len(avg_dst)

            self._sort()

        self._sorted_and_estmated = True


    def log_distances(self):
        for dl in self._distances:
            s = ''
            for d in dl: s += f'{d:.2f} '
            self.logger.info(s)


    def nice_str(
            self,
            n_top=                  20,
            top_npe: Iterable[int]= (3,5,9),
            all_npe: Optional[int]= 3):

        pf = f'.{self._prec}f'

        re_str = ''
        if all_npe: re_str += f'Search run {self.name}, {len(self._srL)} results:\n\n{self.paspa}\n\n'

        if len(self._srL) < n_top: n_top = len(self._srL)
        orig_npe = self._npe
        top_npe = list(top_npe)
        for npe in top_npe:
            self.set_npe(npe)

            re_str += f'TOP{n_top} results for NPE {npe} (avg_dst:{self._avg_dst:.3f}):'
            if top_npe.index(npe) == 0: re_str += ' -- id smooth [local] [max-min] avg_dst {params..}\n'
            else: re_str += '\n'

            for srIX in range(n_top):
                sr = self._srL[srIX]
                ss_np, avg_dst, all_scores = self.smooth_point(sr)
                re_str += f'{sr.id:4d} {ss_np:{pf}} [{sr.score:{pf}}] [{max(all_scores):{pf}}-{min(all_scores):{pf}}] {avg_dst:.3f} {point_str(sr.point)}\n'

        self.set_npe(orig_npe)
        top_sr_npe = self.get_top_SR()
        n_closest = self._get_n_closest(top_sr_npe, n=self._npe)
        re_str += f'{self._npe} closest points to TOP (npe {self._npe}):\n'
        for sr in n_closest: re_str += f'{sr.id:4d} [{sr.score:{pf}}] {point_str(sr.point)}\n'

        if all_npe and len(self._srL) > n_top:
            self.set_npe(all_npe)
            re_str += f'\nALL results for NPE {all_npe} (avg_dst:{self._avg_dst:.3f}):\n'
            for sr in self._srL:
                ss_np, avg_dst, all_scores = self.smooth_point(sr)
                re_str += f'{sr.id:4d} {ss_np:{pf}} [{sr.score:{pf}}] [{max(all_scores):{pf}}-{min(all_scores):{pf}}] {avg_dst:.3f} {point_str(sr.point)}\n'

        self.set_npe(orig_npe)
        return re_str

    # 3D plot
    def plot(
            self,
            estimate=   True,   # for True color_"axis" == estimate, else score
            folder: str=    None):

        columns = sorted(list(self._srL[0].point.keys()))[:3] if not self.plot_axes else [] + self.plot_axes
        valLL = [[res.point[key] for key in columns] for res in self._srL]

        # eventually add score
        if len(columns) < 3:
            valLL = [vl + [res.score] for vl,res in zip(valLL,self._srL)]
            columns += ['score']

        # eventually add estimate (for one real axis)
        if len(columns) < 3:
            valLL = [vl + [res.estimate] for vl,res in zip(valLL,self._srL)]
            columns += ['estimate']

        # add color "axis" data
        columns += ['estimate' if estimate else 'score']
        valLL = [valL + [res.estimate if estimate else res.score] for valL, res in zip(valLL, self._srL)]

        three_dim(
            xyz=        valLL,
            name=       self.name,
            x_name=     columns[0],
            y_name=     columns[1],
            z_name=     columns[2],
            val_name=   columns[3],
            save_FD=    folder)

    @property
    def npe(self) -> int:
        return self._npe


    def __len__(self): return len(self._srL)