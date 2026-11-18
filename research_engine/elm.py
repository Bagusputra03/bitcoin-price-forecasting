import numpy as np

class ELM:
    """ELM + regularisasi Tikhonov sesuai skrip penelitian."""
    def __init__(self, n_hidden, C=1e3, random_state=None):
        self.n_hidden = int(n_hidden)
        self.C = float(C)
        self.rng = np.random.RandomState(random_state)
        self.W = self.b = self.beta = None

    def _H(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(X @ self.W.T + self.b, -500, 500)))

    def _ridge(self, H, y):
        n, L = H.shape
        if L <= n:
            return np.linalg.solve(H.T @ H + np.eye(L) / self.C, H.T @ y)
        return H.T @ np.linalg.solve(H @ H.T + np.eye(n) / self.C, y)

    def fit(self, X, y):
        nf = X.shape[1]
        self.W = self.rng.uniform(-1.0, 1.0, (self.n_hidden, nf))
        self.b = self.rng.uniform(-1.0, 1.0, (self.n_hidden,))
        self.beta = self._ridge(self._H(X), y)
        return self

    def predict(self, X):
        return self._H(X) @ self.beta
