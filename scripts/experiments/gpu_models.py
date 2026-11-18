#!/usr/bin/env python3
"""GPU (ROCm/CUDA torch) drop-ins for the linear model and the MLP of exp_l1_ladder.

Enabled by GPU=1 in the environment when torch sees a device (run with .venv-rocm/bin/python).

TorchLinear: one-vs-rest logistic regression with the objective of SGDClassifier(loss="log_loss",
alpha=a), per class mean log-loss + (a / 2) ||w||^2 (intercept not penalized), solved to convergence
by full-batch L-BFGS (all classes jointly; the objective is separable). Binary problems have one
coefficient row and multiclass probabilities are the normalized per-class sigmoids, as in sklearn.
SGD only approximates this optimum, so results agree with the CPU path up to SGD's optimization error
(checked by gpu_check.py). SGDClassifierGPU swaps this solver into sklearn's SGDClassifier for the
log-loss / L2 case; sitecustomize in .venv-rocm installs it when GPU=1.
TorchMLP: one hidden layer of 512 ReLU units, L2 1e-4, Adam, early stopping on a 10% validation
split (patience 10 epochs), as MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-4, early_stopping=True).
Both expose classes_, coef_ / intercept_ (linear), predict, predict_proba and decision_function.
"""
import os

import numpy as np

try:
    import torch
    AVAILABLE = torch.cuda.is_available()
except ImportError:  # system python without torch: CPU path only
    torch = None
    AVAILABLE = False

ENABLED = AVAILABLE and os.environ.get("GPU") == "1"
DEV = "cuda" if AVAILABLE else "cpu"


def _forked_child():  # CUDA cannot be re-initialized after fork: worker processes use the CPU path
    global ENABLED
    ENABLED = False


os.register_at_fork(after_in_child=_forked_child)


def _t(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = np.asarray(X)
    return torch.as_tensor(X, dtype=torch.float32, device=DEV)


class _Base:
    def predict_proba(self, X):
        z = torch.as_tensor(self.decision_function(X))
        return torch.softmax(z, 1).numpy()

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), 1)]


class TorchLinear(_Base):
    def __init__(self, alpha=1e-4, max_iter=500, tol=1e-7):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, Xt=None):
        self.classes_, yi = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if Xt is None:  # sparse input stays sparse (CSR on the device): no dense host copy for very wide X
            if hasattr(X, "tocsr"):
                Xt = TorchLogReg._sp(X)
            else:
                Xt = _t(X)
        yt = torch.as_tensor(yi, device=DEV)
        # float32: this GPU's float64 rate is low, and float32 reproduced SGD's predictions exactly
        if K == 2:
            Y = (yt == 1).float()[:, None]
        else:
            Y = torch.nn.functional.one_hot(yt, K).float()
        if Xt.layout == torch.sparse_csr:
            Xd = Xt
        else:
            Xd = Xt.float()
        R = Y.shape[1]
        W = torch.zeros(R, Xd.shape[1], dtype=torch.float32, device=DEV, requires_grad=True)
        b = torch.zeros(R, dtype=torch.float32, device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=self.max_iter, history_size=20,
                                tolerance_grad=self.tol, tolerance_change=1e-10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            Z = Xd @ W.T + b
            per_elem = torch.nn.functional.binary_cross_entropy_with_logits(Z, Y, reduction="none")
            L = per_elem.mean(0).sum()
            L = L + 0.5 * self.alpha * (W * W).sum()
            L.backward()
            return L

        opt.step(closure)
        self.coef_ = W.detach().cpu().numpy().astype(np.float64)
        self.intercept_ = b.detach().cpu().numpy().astype(np.float64)
        return self

    def decision_function(self, X):
        z = np.asarray(X @ self.coef_.T) + self.intercept_
        if z.shape[1] == 1:
            return z[:, 0]
        return z

    def predict(self, X):
        if len(self.classes_) == 1:  # degenerate single-class fit: predict that class
            return np.full(X.shape[0], self.classes_[0])
        z = self.decision_function(X)
        if z.ndim == 1:
            return self.classes_[(z > 0).astype(int)]
        return self.classes_[np.argmax(z, 1)]

    def predict_proba(self, X):
        z = self.decision_function(X)
        p = 1 / (1 + np.exp(-z))
        if z.ndim == 1:
            return np.c_[1 - p, p]
        s_ = p.sum(1, keepdims=True)
        return np.divide(p, s_, out=np.full_like(p, 1 / p.shape[1]), where=s_ > 0)


class TorchMLP(_Base):
    def __init__(self, hidden=512, alpha=1e-4, max_iter=300, seed=42, patience=10, batch=200, lr=1e-3):
        self.hidden = hidden
        self.alpha = alpha
        self.max_iter = max_iter
        self.seed = seed
        self.patience = patience
        self.batch = batch
        self.lr = lr

    def fit(self, X, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        Xt = _t(X)
        yt = torch.as_tensor(yi, device=DEV)
        n = len(yi)
        perm = torch.randperm(n, generator=g).to(DEV)
        nv = max(1, n // 10)
        va = perm[:nv]
        tr = perm[nv:]
        net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], self.hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(self.hidden, K)).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        best = -1.0
        best_state = None
        bad = 0
        for _ in range(self.max_iter):
            net.train()
            order = tr[torch.randperm(len(tr), generator=g).to(DEV)]
            for i in range(0, len(order), self.batch):
                idx = order[i:i + self.batch]
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(net(Xt[idx]), yt[idx])
                # L2 on weight matrices only (not biases), summed in parameter order
                l2 = 0
                for n_, p in net.named_parameters():
                    if n_.endswith("weight"):
                        l2 = l2 + (p * p).sum()
                loss = loss + 0.5 * self.alpha * l2 / len(idx)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                acc = (net(Xt[va]).argmax(1) == yt[va]).float().mean().item()
            if acc > best + 1e-4:
                best = acc
                bad = 0
                best_state = {}
                for k, v in net.state_dict().items():
                    best_state[k] = v.clone()
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net = net.eval()
        return self

    def decision_function(self, X):
        with torch.no_grad():
            out = []
            if hasattr(X, "toarray"):
                Xn = X.toarray()
            else:
                Xn = np.asarray(X)
            for i in range(0, len(Xn), 4096):
                chunk = self.net(_t(Xn[i:i + 4096]))
                out.append(chunk.cpu().numpy())
        return np.concatenate(out).astype(np.float64)


def _as_sklearn_sgd(state):
    import sklearn.linear_model._stochastic_gradient as sg
    cls = sg.SGDClassifier
    if getattr(cls, "_gpu_wrapper", False):  # installed in this process too: use the original class
        cls = cls.__bases__[0]
    m = cls.__new__(cls)
    m.__setstate__(state)
    return m


def _sgd_gpu_class():
    from sklearn.linear_model import SGDClassifier as _SGD

    class SGDClassifierGPU(_SGD):
        """SGDClassifier whose plain log-loss/L2 fit runs exactly on the GPU (TorchLinear); else sklearn's SGD."""
        def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
            if (self.loss != "log_loss" or self.penalty != "l2" or sample_weight is not None
                    or self.class_weight is not None or coef_init is not None or not self.fit_intercept
                    or not ENABLED):
                return super().fit(X, y, coef_init=coef_init, intercept_init=intercept_init,
                                   sample_weight=sample_weight)
            m = TorchLinear(self.alpha).fit(X, y)
            self.classes_ = m.classes_
            self.coef_ = m.coef_
            self.intercept_ = m.intercept_
            self.n_features_in_ = self.coef_.shape[1]
            self.n_iter_ = 1
            self.t_ = 1.0
            self._gpu = True
            return self

        def __reduce_ex__(self, protocol):  # pickles as sklearn's SGDClassifier (fitted state kept), so
            return _as_sklearn_sgd, (self.__getstate__(),)  # models can be sent to worker processes

    SGDClassifierGPU._gpu_wrapper = True
    SGDClassifierGPU.__name__ = "SGDClassifier"
    SGDClassifierGPU.__qualname__ = "SGDClassifier"
    return SGDClassifierGPU


def install():
    """Replace sklearn's SGDClassifier by the GPU-backed subclass; no-op unless GPU=1 and a device is present."""
    if not ENABLED:
        return
    import sklearn.linear_model as lm
    import sklearn.linear_model._stochastic_gradient as sg
    cls = _sgd_gpu_class()
    lm.SGDClassifier = cls
    sg.SGDClassifier = cls


class TorchLogReg(_Base):
    """Multinomial L2 LogisticRegression(C) by full-batch float32 L-BFGS on the GPU, for sparse or dense X."""
    def __init__(self, C=1.0, max_iter=3000, tol=1e-6):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _sp(X):
        import scipy.sparse as sp
        X = sp.csr_matrix(X, dtype=np.float32)
        crow = torch.as_tensor(X.indptr, dtype=torch.int64)
        col = torch.as_tensor(X.indices, dtype=torch.int64)
        val = torch.as_tensor(X.data)
        return torch.sparse_csr_tensor(crow, col, val, size=X.shape, device=DEV)

    def fit(self, X, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if hasattr(X, "tocsr"):
            Xt = self._sp(X)
        else:
            Xt = _t(X)
        yt = torch.as_tensor(yi, device=DEV)
        W = torch.zeros(Xt.shape[1], K, device=DEV, requires_grad=True)
        b = torch.zeros(K, device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=self.max_iter, history_size=10,
                                tolerance_grad=self.tol, tolerance_change=1e-12, line_search_fn="strong_wolfe")
        n = len(yi)

        def closure():
            opt.zero_grad()
            Z = (Xt @ W) + b
            # scaled by 1/(C n) for conditioning; same minimizer as 0.5||W||^2 + C * sum CE
            L = torch.nn.functional.cross_entropy(Z, yt, reduction="sum") / n + 0.5 * (W * W).sum() / (self.C * n)
            L.backward()
            return L

        opt.step(closure)
        self.coef_ = W.detach().T.cpu().numpy().astype(np.float64)
        self.intercept_ = b.detach().cpu().numpy().astype(np.float64)
        del W, b, opt, Xt
        torch.cuda.empty_cache()
        return self

    def decision_function(self, X):
        return np.asarray(X @ self.coef_.T) + self.intercept_


def fit_linear_gpu(X, y, groups, tr_idx, seed, upsample):
    """fit_linear with the same inner 3-fold grouped alpha search, on the GPU."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    Xg = _t(X)
    best = -1
    best_a = None
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    splits = list(inner.split(X[tr_idx], y[tr_idx], groups[tr_idx]))
    for a in (1e-5, 1e-4, 1e-3):
        sc = []
        for itr, ite in splits:
            itr = tr_idx[itr]
            ite = tr_idx[ite]
            up = upsample(itr, y, seed)
            m = TorchLinear(a).fit(None, y[up], Xt=Xg[torch.as_tensor(up, device=DEV)])
            sc.append(f1_score(y[ite], m.predict(X[ite]), average="macro"))
        if np.mean(sc) > best:
            best = np.mean(sc)
            best_a = a
    up = upsample(tr_idx, y, seed)
    m = TorchLinear(best_a).fit(None, y[up], Xt=Xg[torch.as_tensor(up, device=DEV)])
    return m, {"alpha": best_a, "backend": "gpu"}
