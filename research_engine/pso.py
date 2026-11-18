import numpy as np
from .config import (PSO_PARTICLES, PSO_ITERATIONS, PSO_C1, PSO_C2,
                     PSO_WMAX, PSO_WMIN, PSO_NH_BOUNDS, PSO_LOG10C_BOUNDS)

class PSO2D:
    def __init__(self, fn, N=PSO_PARTICLES, I_max=PSO_ITERATIONS,
                 c1=PSO_C1, c2=PSO_C2, wmax=PSO_WMAX, wmin=PSO_WMIN,
                 bounds_min=(PSO_NH_BOUNDS[0], PSO_LOG10C_BOUNDS[0]),
                 bounds_max=(PSO_NH_BOUNDS[1], PSO_LOG10C_BOUNDS[1])):
        self.fn = fn
        self.N, self.I = int(N), int(I_max)
        self.c1, self.c2 = c1, c2
        self.wmax, self.wmin = wmax, wmin
        self.xmin = np.array(bounds_min, dtype=float)
        self.xmax = np.array(bounds_max, dtype=float)
        self.dim = len(bounds_min)

    def run(self):
        X = self.xmin + np.random.rand(self.N, self.dim) * (self.xmax - self.xmin)
        V = np.zeros_like(X)
        f = np.array([self.fn(X[i]) for i in range(self.N)])
        pb, pf = X.copy(), f.copy()
        gi = np.argmin(pf)
        gp, gf = pb[gi].copy(), pf[gi]
        hist = [float(gf)]
        for t in range(self.I):
            w = self.wmax - (self.wmax - self.wmin) * t / max(self.I - 1, 1)
            r1 = np.random.rand(self.N, self.dim)
            r2 = np.random.rand(self.N, self.dim)
            V = w * V + self.c1 * r1 * (pb - X) + self.c2 * r2 * (gp - X)
            X = np.clip(X + V, self.xmin, self.xmax)
            f = np.array([self.fn(X[i]) for i in range(self.N)])
            m = f < pf
            pb[m], pf[m] = X[m].copy(), f[m]
            ni = np.argmin(pf)
            if pf[ni] < gf:
                gp, gf = pb[ni].copy(), pf[ni]
            hist.append(float(gf))
        return gp, float(gf), hist
