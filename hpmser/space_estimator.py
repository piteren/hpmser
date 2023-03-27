from abc import ABC, abstractmethod
import numpy as np
from pypaq.pytypes import NPL
from pypaq.pms.base import PMSException
from pypaq.pms.paspa import PaSpa
from sklearn.svm import SVR
from typing import List, Optional, Tuple, Dict

from hpmser.points_cloud import VPoint



# MSE average loss for Estimator
def loss(
        model,
        y_test: NPL,
        X_test: Optional[NPL]=   None,
        preds: Optional[NPL]=   None,
) -> float:

    if X_test is None and preds is None:
        raise PMSException('\'X_test\' or \'preds\' must be given')

    if preds is None:
        preds = model.predict(X=X_test)

    return sum([(a - b) ** 2 for a, b in zip(preds, y_test)]) / len(y_test)


class SpaceEstimator(ABC):

    # extracts X & y from vpoints ands space
    @staticmethod
    def _extract_Xy(vpoints:List[VPoint], space:PaSpa) -> Tuple[np.ndarray, np.ndarray]:

        points = [vp.point for vp in vpoints]
        points_normalized = [space.point_normalized(p) for p in points]
        keys = sorted(list(points_normalized[0].keys()))
        points_feats = [[pn[k] for k in keys] for pn in points_normalized]

        scores = [vp.value for vp in vpoints]

        return np.asarray(points_feats), np.asarray(scores)

    # updates model with given data, returns loss of new model for given data
    def update(self, X_new:NPL, y_new:NPL) -> float:
        pass


    def update_vpoints(self, vpoints:List[VPoint], space:PaSpa) -> float:
        X, y = SpaceEstimator._extract_Xy(vpoints, space)
        return self.update(X, y)

    # predicts
    def predict(self, X:NPL) -> np.ndarray:
        pass


    def predict_vpoints(self, vpoints:List[VPoint], space:PaSpa) -> np.ndarray:
        X, y = SpaceEstimator._extract_Xy(vpoints, space)
        return self.predict(X)

    # Estimator status
    @property
    def fitted(self) -> bool:
        return False

    # Estimator state
    @property
    def state(self) -> Dict:
        return {}

    @classmethod
    @abstractmethod
    def from_state(cls, state:Dict):
        pass


    def __str__(self):
        return 'SpaceEstimator'


# Support Vector Regression (SVR) with Radial Basis Function (RBF) kernel based Space Estimator
class RBFRegressor(SpaceEstimator):

    # C & gamma parameters possible values
    VAL = {
        'c': [100, 10, 1, 0.1, 0.01],
        'g': [0.01, 0.1, 1, 10, 100]}

    def __init__(
            self,
            epsilon: float= 0.01,
            num_tries: int= 2,  # how many times param with next ix needs to improve to change ix
            seed=           123,
    ):

        self._epsilon = epsilon
        self._num_tries = num_tries

        self._indexes =  {'c':0, 'g':0}
        self._improved = {
            'cH': 0, # c higher
            'cL': 0, # c lower
            'gH': 0, # g higher
            'gL': 0} # g lower

        self._model = None

        self.data_X: Optional[NPL] = None
        self.data_y: Optional[NPL] = None

        np.random.seed(seed)

    # builds SVR RBF model from given indexes of parameters
    def _build_model(self, cix:Optional[int]=None, gix:Optional[int]=None) -> SVR:
        if cix is None: cix = self._indexes['c']
        if gix is None: gix = self._indexes['g']
        return SVR(
            kernel=     "rbf",                      # Radial Basis Function kernel
            C=          RBFRegressor.VAL['c'][cix], # regularization, controls error with margin, lower C -> larger margin, more support vectors, longer fitting time
            gamma=      RBFRegressor.VAL['g'][gix], # controls shape of decision boundary, larger value for more complex
            epsilon=    self._epsilon)              # threshold for what is considered an acceptable error rate in the training data

    # fits model, returns test loss
    @staticmethod
    def _fit(
            model,
            X_train: NPL,
            y_train: NPL,
            X_test: NPL,
            y_test: NPL,
    ) -> float:
        model.fit(X=X_train, y=y_train)
        return loss(model=model, X_test=X_test, y_test=y_test)

    # tries to update model params, adds data, then fits, returns loss of new model for given data
    def update(self, X_new:NPL, y_new:NPL) -> float:

        # add new data
        self.data_X = np.concatenate([self.data_X, X_new]) if self.data_X is not None else X_new
        self.data_y = np.concatenate([self.data_y, y_new]) if self.data_y is not None else y_new

        ### prepare data split

        data_size = len(self.data_X)
        train_size = data_size // 2 # TODO <- parametrize?
        choice = np.random.choice(range(data_size), size=train_size, replace=False)
        tr_sel = np.zeros(data_size, dtype=bool)
        tr_sel[choice] = True
        ts_sel = ~tr_sel

        X_train = self.data_X[tr_sel]
        y_train = self.data_y[tr_sel]
        X_test = self.data_X[ts_sel]
        y_test = self.data_y[ts_sel]

        m_configs = {
            'current': {'cix': self._indexes['c'],   'gix': self._indexes['g']},   # current
            'cH':      {'cix': self._indexes['c']+1, 'gix': self._indexes['g']},   # c higher
            'cL':      {'cix': self._indexes['c']-1, 'gix': self._indexes['g']},   # c lower
            'gH':      {'cix': self._indexes['c'],   'gix': self._indexes['g']+1}, # g higher
            'gL':      {'cix': self._indexes['c'],   'gix': self._indexes['g']-1}} # g lower

        # filter out not valid
        not_valid = []
        for k in m_configs:
            for p in m_configs[k]:
                if m_configs[k][p] in [-1,len(RBFRegressor.VAL[p[0]])]:
                    not_valid.append(k)
        not_valid = list(set(not_valid))
        for k in not_valid:
            m_configs.pop(k)

        models = {config: self._build_model(**m_configs[config]) for config in m_configs}

        losses = {config: RBFRegressor._fit(
            model=      models[config],
            X_train=    X_train,
            y_train=    y_train,
            X_test=     X_test,
            y_test=     y_test) for config in models}

        loss_current = losses.pop('current')

        for k in losses:
            if losses[k] < loss_current: self._improved[k] += 1
            else:                        self._improved[k] = 0

        # search for k to update, priority for lower
        update_k = None
        losses_sorted = sorted([(k,l) for k,l in losses.items()], key=lambda x:x[1])
        for e in losses_sorted:
            k = e[0]
            if self._improved[k] == self._num_tries:
                update_k = k
                break

        if update_k is not None:
            self._indexes[update_k[0]] += 1 if update_k[1] == 'H' else -1
            self._model = models[update_k]
        else:
            self._model = models['current']

        # finally fit with all data
        return RBFRegressor._fit(
            model=      self._model,
            X_train=    self.data_X,
            y_train=    self.data_y,
            X_test=     self.data_X,
            y_test=     self.data_y)


    def predict(self, x:NPL) -> np.ndarray:
        if not self.fitted:
            raise Exception('RBFRegressor needs to be fitted before predict')
        return self._model.predict(X=x)

    @property
    def fitted(self) -> bool:
        return self._model is not None

    # returns object state
    @property
    def state(self) -> Dict:
        return {
            'indexes':      self._indexes,
            'epsilon':      self._epsilon,
            'num_tries':    self._num_tries,
            'improved':     self._improved}

    # builds object from a state
    @classmethod
    def from_state(cls, state:Dict):
        reg = cls(epsilon=state['epsilon'], num_tries=state['num_tries'])
        reg._indexes = state['indexes']
        reg._improved = state['improved']
        return reg

    def __str__(self):
        c = RBFRegressor.VAL['c'][self._indexes['c']]
        g = RBFRegressor.VAL['g'][self._indexes['g']]
        return f'SVR RBF, C:{c}, gamma:{g}, data:{len(self.data_X)}'