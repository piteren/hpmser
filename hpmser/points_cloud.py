import math
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.stats import mam
from pypaq.lipytools.plots import three_dim
from pypaq.pms.base import POINT
from pypaq.pms.paspa import PaSpa
from typing import Sized, List, Tuple, Optional, Dict, Union


# Valued Point, with id & value
class VPoint:

    def __init__(
            self,
            point: POINT,
            id: Optional[int] =     None,
            value: Optional[float]= None,
    ):
        self.point = point
        self.id = id
        self.value = value

    def __str__(self):
        return f'SeRes: id:{self.id}, point:{self.point}, value:{self.value}'


class PointsCloud(Sized):

    def __init__(
            self,
            paspa: PaSpa,   # space of this PointsCloud
            logger=     None,
            loglevel=   20):

        if not logger:
            logger = get_pylogger(level=loglevel)
        self.logger = logger
        self.logger.info('*** PointsCloud *** initializing..')

        self.paspa = paspa

        # those below are updated with each call to update_cloud()
        self._vpointsD: Dict[int, VPoint] = {}          # {id: VPoint}
        self._nearest: Dict[int, Tuple[int,float]] = {} # {id: (id,dist)}
        self.min_nearest = math.sqrt(self.paspa.dim)
        self.avg_nearest = math.sqrt(self.paspa.dim)
        self.max_nearest = math.sqrt(self.paspa.dim)

        self.prec = 8 # print precision, will be updated while adding new vpoints

    """
    def _get_srl_path(self, save_dir:str) -> str:
        return f'{save_dir}/{self.name}.srl'


    def _get_srl_backup_path(self, save_dir:str) -> str:
        return f'{save_dir}/{self.name}.srl.backup'

    # loads (alternatively from backup)
    def load(self, save_dir:str):

        self.logger.info(f' > SRL {self.name} loading form {save_dir}..')

        try:
            obj = r_pickle(self._get_srl_path(save_dir))
        except Exception as e:
            self.logger.warning(f' SRL {self.name} got exception: {str(e)} while loading, using backup file')
            obj = r_pickle(self._get_srl_backup_path(save_dir))

        self.paspa =        obj.paspa
        self._vpointsD =      obj._vpointsD
        self._nearest =     obj._nearest
        self.min_nearest =  obj.min_nearest
        self.avg_nearest =  obj.avg_nearest
        self.max_nearest =  obj.max_nearest
        self.prec=         obj.prec

        self.logger.info(f'> PointsCloud loaded {len(self)} vpoints')

    # saves with backup
    def save(self, folder :str):

        # backup copy previous
        old_res = r_pickle(self._get_srl_path(folder))
        if old_res: w_pickle(old_res, self._get_srl_backup_path(folder))

        w_pickle(self, self._get_srl_path(folder))
        #self.plot(folder=folder)
    """
    # returns distance between two vpoints
    def distance(self, vpa:VPoint, vpb:VPoint) -> float:
        return self.paspa.distance(vpa.point, vpb.point)

    # updates cloud (self) with given VPoint / list (adds new to _vpoints & updates _nearest)
    def update_cloud(self, vpoints:Union[VPoint,List[VPoint]]):

        if vpoints:

            if type(vpoints) is not list:
                vpoints = [vpoints]

            for vpoint in vpoints:

                # add to _vpoints
                vp_id = len(self)
                vpoint.id = vp_id
                self._vpointsD[vp_id] = vpoint

                # update _nearest
                his_nearest = None
                his_nearest_dist = None
                for k in self._vpointsD:
                    if k != vp_id:
                        dist = self.distance(vpoint, self._vpointsD[k])
                        if his_nearest is None or dist < his_nearest_dist:
                            his_nearest = k
                            his_nearest_dist = dist
                        if k not in self._nearest or dist < self._nearest[k][1]:
                            self._nearest[k] = vp_id, dist
                if his_nearest is not None:
                    self._nearest[vp_id] = his_nearest, his_nearest_dist

                if vpoint.value > 0.01: self.prec = 4

            self.min_nearest, self.avg_nearest, self.max_nearest = mam([v[1] for v in self._nearest.values()])

    # prepares 3D plot of Cloud
    def plot(
            self,
            name: str=                  'PointsCloud',
            axes: Optional[List[str]]=  None,   # list with axes names, 2-3, like ['drop_a','drop_b','loss']
            folder: Optional[str]=      None):

        columns = sorted(list(self._vpointsD[0].point.keys()))[:3] if not axes else [] + axes

        if len(columns) < 2:
            self.logger.warning('Cannot prepare 3D plot for less than two axes')

        else:

            valLL = [[sp.point[key] for key in columns] for sp in self._vpointsD.values()]

            # eventually add score
            if len(columns) < 4:
                columns += ['value']
                valLL = [vl + [sp.value] for vl,sp in zip(valLL, self._vpointsD.values())]

            three_dim(
                xyz=        valLL,
                name=       name,
                x_name=     columns[0],
                y_name=     columns[1],
                z_name=     columns[2],
                val_name=   'value',
                save_FD=    folder)

    @property
    def vpoints(self) -> List[VPoint]:
        return list(self._vpointsD.values())

    # number of VPoints in Cloud
    def __len__(self):
        return len(self._vpointsD)