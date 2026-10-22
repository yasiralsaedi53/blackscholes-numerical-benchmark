"""Pricing methods for Black–Scholes-type contracts.

Contracts: European call, cash-or-nothing digital call, down-and-out call (continuous barrier),
American put. Methods: finite differences (theta-schemes), CRR/BBSR trees, Monte Carlo, COS.
All functions return (price, delta, gamma) at S0 unless stated.
"""
import numpy as np
from scipy.stats import norm
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------- exact formulas
def bs_call(S, K, r, s, T):
    S = np.asarray(S, float)
    d1 = (np.log(S / K) + (r + 0.5 * s * s) * T) / (s * np.sqrt(T)); d2 = d1 - s * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def call_exact(S, K, r, s, T):
    d1 = (np.log(S / K) + (r + 0.5 * s * s) * T) / (s * np.sqrt(T))
    return bs_call(S, K, r, s, T), norm.cdf(d1), norm.pdf(d1) / (S * s * np.sqrt(T))

def digital_exact(S, K, r, s, T):
    sq = s * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * s * s) * T) / sq; d2 = d1 - sq
    df = np.exp(-r * T)
    return df * norm.cdf(d2), df * norm.pdf(d2) / (S * sq), -df * norm.pdf(d2) * d1 / (S * S * sq * sq)

def dao_price(S, K, B, r, s, T):
    """Down-and-out call, B < K, continuous monitoring, no dividends."""
    S = np.asarray(S, float)
    p = 2 * r / (s * s) - 1
    v = bs_call(S, K, r, s, T) - (B / S) ** p * bs_call(B * B / S, K, r, s, T)
    return np.where(S <= B, 0.0, v)

def dao_exact(S, K, B, r, s, T, h=1e-3):
    f = lambda x: float(dao_price(x, K, B, r, s, T))
    return f(S), (f(S + h) - f(S - h)) / (2 * h), (f(S + h) - 2 * f(S) + f(S - h)) / (h * h)

# ---------------------------------------------------------------- finite differences
def _greeks_from_grid(Sg, V, S0):
    i = int(round((S0 - Sg[0]) / (Sg[1] - Sg[0])))
    h = Sg[1] - Sg[0]
    return V[i], (V[i + 1] - V[i - 1]) / (2 * h), (V[i + 1] - 2 * V[i] + V[i - 1]) / h ** 2

def fd_price(contract, scheme, N, M, S0=100., K=100., r=.05, s=.2, T=1., B=90., Smax_mult=4., n_rannacher=2,
             return_grid=False):
    """Uniform S grid, N intervals, M time steps.
    scheme: 'BE' (backward Euler), 'CN' (Crank–Nicolson), 'CNR' (CN with Rannacher start-up).
    contract: 'call', 'digital', 'dao' (down-and-out call), 'amput' (American put, Brennan–Schwartz)."""
    Smin = B if contract == 'dao' else 0.0
    Smax = Smax_mult * K
    Sg = np.linspace(Smin, Smax, N + 1)
    h = Sg[1] - Sg[0]; dt = T / M
    if contract in ('call', 'dao'):
        V = np.maximum(Sg - K, 0.0)
    elif contract == 'digital':
        V = (Sg > K).astype(float); V[np.isclose(Sg, K)] = 0.5   # value at the discontinuity = average
    elif contract == 'amput':
        V = np.maximum(K - Sg, 0.0)
    payoff = V.copy()
    x = Sg[1:-1]
    a = 0.5 * s * s * x * x / h ** 2 - 0.5 * r * x / h
    b = -s * s * x * x / h ** 2 - r
    c = 0.5 * s * s * x * x / h ** 2 + 0.5 * r * x / h
    L = diags([a[1:], b, c[:-1]], [-1, 0, 1], format='csc')
    I = identity(len(x), format='csc')

    def bc(tau):
        if contract == 'call':    return 0.0, Smax - K * np.exp(-r * tau)
        if contract == 'digital': return 0.0, np.exp(-r * tau)
        if contract == 'dao':     return 0.0, Smax - K * np.exp(-r * tau)
        if contract == 'amput':   return K, 0.0

    def step(Vin, tau0, dtl, theta, lu):
        lo0, hi0 = bc(tau0); lo1, hi1 = bc(tau0 + dtl)
        rhs = Vin[1:-1] + (1 - theta) * dtl * (L @ Vin[1:-1])
        rhs[0] += dtl * a[0] * ((1 - theta) * lo0 + theta * lo1)
        rhs[-1] += dtl * c[-1] * ((1 - theta) * hi0 + theta * hi1)
        if contract == 'amput':
            inner = penalty_lcp(-theta * dtl * a, 1 - theta * dtl * b, -theta * dtl * c, rhs, payoff[1:-1],
                                Vin[1:-1])
        else:
            inner = lu.solve(rhs)
        out = np.empty_like(Vin); out[1:-1] = inner; out[0] = lo1; out[-1] = hi1
        return out

    cache = {}
    def getlu(theta, dtl):
        key = (theta, dtl)
        if key not in cache:
            cache[key] = splu((I - theta * dtl * L).tocsc()) if contract != 'amput' else None
        return cache[key]

    tau = 0.0
    nR = n_rannacher if scheme == 'CNR' else 0
    for m in range(M):
        if scheme == 'BE' or m < nR:
            for _ in range(2 if m < nR else 1):
                dtl = dt / 2 if m < nR else dt
                V = step(V, tau, dtl, 1.0, getlu(1.0, dtl)); tau += dtl
        else:
            V = step(V, tau, dt, 0.5, getlu(0.5, dt)); tau += dt
    if return_grid:
        return Sg, V
    return _greeks_from_grid(Sg, V, S0)

def penalty_lcp(a, b, c, d, g, u0, rho=1e8, maxit=50):
    """Tridiagonal linear complementarity problem by the penalty / policy-iteration method
    (Forsyth & Vetzal, 2002), solved with LAPACK banded solves. Row i: a_i u_{i-1} + b_i u_i + c_i u_{i+1} = d_i."""
    from scipy.linalg import solve_banded
    n = len(d)
    ab = np.zeros((3, n)); ab[0, 1:] = c[:-1]; ab[2, :-1] = a[1:]
    active = u0 <= g
    for _ in range(maxit):
        ab[1] = b + rho * active
        u = solve_banded((1, 1), ab, d + rho * active * g)
        new = u <= g
        if np.array_equal(new, active): break
        active = new
    return np.maximum(u, g)

def brennan_schwartz(a, b, c, d, g):
    """Solve tridiagonal LCP (American put: exercise region at low S).
    Row i: a_i u_{i-1} + b_i u_i + c_i u_{i+1} = d_i with u >= g. a[0] and c[-1] unused."""
    n = len(d); bp = b.copy().astype(float); dp = d.copy().astype(float)
    for i in range(n - 2, -1, -1):
        m = c[i] / bp[i + 1]
        bp[i] = bp[i] - m * a[i + 1]
        dp[i] = dp[i] - m * dp[i + 1]
    u = np.empty(n)
    u[0] = max(dp[0] / bp[0], g[0])
    for i in range(1, n):
        u[i] = max((dp[i] - a[i] * u[i - 1]) / bp[i], g[i])
    return u

# ---------------------------------------------------------------- binomial trees
def crr(contract, n, S0=100., K=100., r=.05, s=.2, T=1., B=90., bbs=False):
    """Cox–Ross–Rubinstein tree. Greeks from nodes at steps 1 and 2 (Delta, Gamma).
    bbs=True: binomial Black–Scholes (last step replaced by the exact European value)."""
    dt = T / n; u = np.exp(s * np.sqrt(dt)); d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d); disc = np.exp(-r * dt)
    american = contract == 'amput'
    def pay(S):
        if contract in ('call', 'dao'): return np.maximum(S - K, 0.)
        if contract == 'digital': return (S > K).astype(float)
        return np.maximum(K - S, 0.)
    j = np.arange(n + 1)
    if bbs:
        S = S0 * u ** (2 * np.arange(n) - (n - 1))      # nodes at step n-1
        if contract == 'amput':
            Sp = np.maximum(S, 1e-12)
            eu = bs_call(Sp, K, r, s, dt) - Sp + K * np.exp(-r * dt)   # put by parity
            V = np.maximum(eu, pay(S))
        elif contract == 'call':
            V = bs_call(np.maximum(S, 1e-12), K, r, s, dt)
        else:
            raise ValueError('BBS only for call/amput here')
        start = n - 1
    else:
        S = S0 * u ** (2 * j - n)
        V = pay(S)
        if contract == 'dao': V[S <= B] = 0.
        start = n
    levels = {}
    for step in range(start - 1, -1, -1):
        S = S0 * u ** (2 * np.arange(step + 1) - step)
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if american: V = np.maximum(V, pay(S))
        if contract == 'dao': V[S <= B] = 0.
        if step <= 2: levels[step] = (S.copy(), V.copy())
    (S1, V1), (S2, V2) = levels[1], levels[2]
    delta = (V1[1] - V1[0]) / (S1[1] - S1[0])
    gamma = ((V2[2] - V2[1]) / (S2[2] - S2[1]) - (V2[1] - V2[0]) / (S2[1] - S2[0])) / (0.5 * (S2[2] - S2[0]))
    return levels[0][1][0] if 0 in levels else V[0], delta, gamma

def bbsr_amput(n, **kw):
    """Binomial Black–Scholes with Richardson extrapolation (Broadie & Detemple, 1996)."""
    p1 = crr('amput', n, bbs=True, **kw); p2 = crr('amput', n // 2, bbs=True, **kw)
    return tuple(2 * x - y for x, y in zip(p1, p2))

# ---------------------------------------------------------------- Monte Carlo
def mc(contract, npaths, S0=100., K=100., r=.05, s=.2, T=1., B=90., nsteps=50, seed=0):
    """Antithetic Monte Carlo. Greeks: pathwise Delta for the call, likelihood-ratio (LRM) otherwise.
    Barrier: continuous monitoring via Brownian-bridge crossing probability on a discrete grid.
    Returns (price, delta, gamma, se_price)."""
    rng = np.random.default_rng(seed)
    half = npaths // 2
    df = np.exp(-r * T); sq = s * np.sqrt(T)
    if contract in ('call', 'digital'):
        Z = rng.standard_normal(half); Z = np.concatenate([Z, -Z])
        ST = S0 * np.exp((r - 0.5 * s * s) * T + sq * Z)
        if contract == 'call':
            P = np.maximum(ST - K, 0.)
            delta = df * np.mean((ST > K) * ST / S0)
        else:
            P = (ST > K).astype(float)
            delta = df * np.mean(P * Z / (S0 * sq))
        gamma = df * np.mean(P * (Z * Z - 1 - Z * sq) / (S0 * S0 * sq * sq))
        pv = df * P
        pairs = 0.5 * (pv[:half] + pv[half:])
        return pv.mean(), delta, gamma, pairs.std(ddof=1) / np.sqrt(half)
    if contract == 'dao':
        dt = T / nsteps
        Z = rng.standard_normal((half, nsteps)); Z = np.concatenate([Z, -Z])
        lnS = np.log(S0) + np.cumsum((r - 0.5 * s * s) * dt + s * np.sqrt(dt) * Z, axis=1)
        lnS = np.concatenate([np.full((npaths, 1), np.log(S0)), lnS], axis=1)
        lb = np.log(B)
        x0, x1 = lnS[:, :-1] - lb, lnS[:, 1:] - lb
        pcross = np.where((x0 > 0) & (x1 > 0), np.exp(-2 * x0 * x1 / (s * s * dt)), 1.0)
        survive = np.prod(1 - pcross, axis=1)
        P = np.maximum(np.exp(lnS[:, -1]) - K, 0.) * survive
        # LRM on the first increment. The first-interval survival factor
        # q1 = 1 - exp(-a x0 x1), a = 2/(s^2 dt), x0 = ln(S0/B), also depends on S0
        # explicitly; its derivatives (holding S1 fixed) are added to the estimators.
        Z1 = Z[:, 0]; sq1 = s * np.sqrt(dt); a = 2.0 / (s * s * dt)
        x0c, x1c = x0[:, 0], x1[:, 0]
        alive = (x0c > 0) & (x1c > 0)
        e1 = np.where(alive, np.exp(-a * x0c * x1c), 0.0)
        q1 = np.where(alive, 1.0 - e1, 0.0)
        rest = np.where(q1 > 0, P / np.where(q1 > 0, q1, 1.0), 0.0)   # payoff x survival after step 1
        dq1 = a * x1c * e1 / S0                                        # dq1/dS0
        d2q1 = -a * x1c * e1 * (a * x1c + 1.0) / S0 ** 2               # d2q1/dS0^2
        w1 = Z1 / (S0 * sq1)                                           # LRM score
        w2 = (Z1 * Z1 - 1 - Z1 * sq1) / (S0 * S0 * sq1 * sq1)          # LRM second-order weight
        delta = df * np.mean(P * w1 + rest * dq1)
        gamma = df * np.mean(P * w2 + 2.0 * rest * dq1 * w1 + rest * d2q1)
        pv = df * P; pairs = 0.5 * (pv[:half] + pv[half:])
        return pv.mean(), delta, gamma, pairs.std(ddof=1) / np.sqrt(half)
    raise ValueError(contract)

# ---------------------------------------------------------------- COS method
def cos_price(contract, Nterms, S0=100., K=100., r=.05, s=.2, T=1., L=10.):
    """Fang & Oosterlee (2008) COS method for European call and cash-or-nothing digital call.
    Greeks from analytic differentiation of the series in x = ln(S0/K)."""
    x = np.log(S0 / K)
    c1 = (r - 0.5 * s * s) * T; c2 = s * s * T
    a = c1 - L * np.sqrt(c2); bnd = c1 + L * np.sqrt(c2)
    k = np.arange(Nterms); w = k * np.pi / (bnd - a)
    phi = np.exp(1j * w * (r - 0.5 * s * s) * T - 0.5 * s * s * w * w * T)
    def chi(c, d):
        return (np.cos(w * (d - a)) * np.exp(d) - np.cos(w * (c - a)) * np.exp(c)
                + w * np.sin(w * (d - a)) * np.exp(d) - w * np.sin(w * (c - a)) * np.exp(c)) / (1 + w * w)
    def psi(c, d):
        out = np.empty(Nterms); out[0] = d - c
        out[1:] = (np.sin(w[1:] * (d - a)) - np.sin(w[1:] * (c - a))) / w[1:]
        return out
    if contract == 'call':
        Vk = 2 / (bnd - a) * K * (chi(0, bnd) - psi(0, bnd)); scale = 1.0
    elif contract == 'digital':
        Vk = 2 / (bnd - a) * psi(0, bnd); scale = 1.0
    else:
        raise ValueError(contract)
    Vk = Vk.astype(complex); Vk[0] *= 0.5
    e = np.exp(1j * w * (x - a))
    base = phi * e * Vk
    df = np.exp(-r * T)
    V = df * np.real(base.sum())
    dVdx = df * np.real((1j * w * base).sum())
    d2Vdx2 = df * np.real((-(w * w) * base).sum())
    return V * scale, dVdx / S0, (d2Vdx2 - dVdx) / S0 ** 2
