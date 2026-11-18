#!/usr/bin/env python3
"""Batched GPU versions of the greedy token-addition attacks of exp_p4_cfg_cert / exp_open_world.

All applications of a fold are attacked at once, in float64, with the same per-step choices as
the CPU loops: the coordinate with the largest |gradient| of the margin over all blocks (first
block and first coordinate on ties), a step of -sign(gradient) (+1 at a zero gradient), and
termination at the first step after which the prediction leaves y (attack_blocks) or equals the
target class (attack_to). Finished applications are masked out. Returns, per application, the
number of steps or None, and for attack_blocks the class the prediction moved to.
"""
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _prep(phis_list, W, b):
    apps = []
    for p in phis_list:
        apps.append(np.stack(p))  # one application: K blocks x d
    P = torch.as_tensor(np.stack(apps), dtype=torch.float64, device=DEV)
    return P.clone(), torch.as_tensor(W, dtype=torch.float64, device=DEV), torch.as_tensor(b, dtype=torch.float64, device=DEV)


def _scores(P, Wt, bt):
    n, K, d = P.shape
    x = (P / P.norm(dim=2, keepdim=True)).reshape(n, K * d)
    return x @ Wt.T + bt


def _step(P, u, active):
    """One greedy step for the active rows; u is n x (K*d)."""
    n, K, d = P.shape
    a = P.norm(dim=2, keepdim=True)
    uk = u.reshape(n, K, d)
    proj = (uk * P).sum(2, keepdim=True) / a
    g = (uk - proj * P / a) / a
    flat = g.reshape(n, K * d)
    j = flat.abs().argmax(1)
    gj = flat.gather(1, j[:, None])[:, 0]
    sg = torch.where(gj == 0, torch.ones_like(gj), -torch.sign(gj))
    rows = torch.nonzero(active)[:, 0]
    P.view(n, K * d)[rows, j[rows]] += sg[rows]


def attack_blocks(phis_list, W, b, ys, max_t):
    P, Wt, bt = _prep(phis_list, W, b)
    n = P.shape[0]
    y = torch.as_tensor(ys, device=DEV)
    ar = torch.arange(n, device=DEV)
    t_out = torch.full((n,), -1, dtype=torch.long, device=DEV)
    land = torch.full((n,), -1, dtype=torch.long, device=DEV)
    active = torch.ones(n, dtype=torch.bool, device=DEV)
    for t in range(1, max_t + 1):
        z = _scores(P, Wt, bt)
        s = z[ar, y][:, None] - z
        s[ar, y] = float("inf")
        c = s.argmin(1)
        u = Wt[y] - Wt[c]
        _step(P, u, active)
        pred = _scores(P, Wt, bt).argmax(1)
        hit = active & (pred != y)
        t_out[hit] = t
        land[hit] = pred[hit]
        active &= ~hit
        if not active.any():
            break
    t_out = t_out.cpu().numpy()
    land = land.cpu().numpy()
    # -1 marks an application the attack never flipped
    steps = []
    for v in t_out:
        if v < 0:
            steps.append(None)
        else:
            steps.append(int(v))
    landed = []
    for v in land:
        if v < 0:
            landed.append(None)
        else:
            landed.append(int(v))
    return steps, landed


def attack_to(phis_list, W, b, ys, target, max_t):
    P, Wt, bt = _prep(phis_list, W, b)
    n = P.shape[0]
    y = torch.as_tensor(ys, device=DEV)
    u = Wt[y] - Wt[target]
    t_out = torch.full((n,), -1, dtype=torch.long, device=DEV)
    active = torch.ones(n, dtype=torch.bool, device=DEV)
    for t in range(1, max_t + 1):
        _step(P, u, active)
        hit = active & (_scores(P, Wt, bt).argmax(1) == target)
        t_out[hit] = t
        active &= ~hit
        if not active.any():
            break
    steps = []
    for v in t_out.cpu().numpy():
        if v < 0:
            steps.append(None)
        else:
            steps.append(int(v))
    return steps
