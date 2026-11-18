import numpy as np
from .elm import ELM
from .pso import PSO2D
from .config import (PSO_VAL_RATIO, ELM_FIXED_SEED, PSO_NH_BOUNDS,
                     PSO_LOG10C_BOUNDS, PSO_PARTICLES, PSO_ITERATIONS)

class PSO_ELM_Hyper:
    def __init__(self, val_ratio=PSO_VAL_RATIO, fixed_seed=ELM_FIXED_SEED,
                 nh_bounds=PSO_NH_BOUNDS, lc_bounds=PSO_LOG10C_BOUNDS):
        self.val_ratio = val_ratio
        self.fixed_seed = fixed_seed
        self.nh_min, self.nh_max = nh_bounds
        self.lc_min, self.lc_max = lc_bounds
        self.best_nh = self.best_C = self.elm_ = self.conv_hist = None

    def _decode(self, p):
        nh = int(np.clip(round(p[0]), self.nh_min, self.nh_max))
        C = 10 ** np.clip(p[1], self.lc_min, self.lc_max)
        return nh, C

    def fit(self, X_train, y_train, N_pso=PSO_PARTICLES, I_max=PSO_ITERATIONS):
        n_val = max(10, int(len(X_train) * self.val_ratio))
        n_in = len(X_train) - n_val
        Xin, yin = X_train[:n_in], y_train[:n_in]
        Xval, yval = X_train[n_in:], y_train[n_in:]

        def fitness(p):
            nh, C = self._decode(p)
            elm = ELM(n_hidden=nh, C=C, random_state=self.fixed_seed).fit(Xin, yin)
            yp = elm.predict(Xval)
            m = (yval != 0).flatten()
            if not np.any(m):
                return float("inf")
            return float(np.mean(np.abs(
                (yval.flatten()[m] - yp.flatten()[m]) / yval.flatten()[m]
            )) * 100)

        pso = PSO2D(fitness, N=N_pso, I_max=I_max,
                    bounds_min=(self.nh_min, self.lc_min),
                    bounds_max=(self.nh_max, self.lc_max))
        best_p, _, self.conv_hist = pso.run()
        self.best_nh, self.best_C = self._decode(best_p)
        self.elm_ = ELM(self.best_nh, self.best_C, self.fixed_seed).fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.elm_.predict(X)
