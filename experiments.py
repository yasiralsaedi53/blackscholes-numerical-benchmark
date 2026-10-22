"""Experiments for the Black–Scholes numerical-methods benchmark. Writes results.json and figures."""
import json, time
import numpy as np
from methods import *

P = dict(S0=100., K=100., r=.05, s=.2, T=1.)
B = 90.
EX = {'call': call_exact(100., 100., .05, .2, 1.),
      'digital': digital_exact(100., 100., .05, .2, 1.),
      'dao': dao_exact(100., 100., B, .05, .2, 1.)}
res = {'params': dict(P, B=B), 'exact': {k: [float(x) for x in v] for k, v in EX.items()}}

def err(v, ex):  # absolute errors in price, delta, gamma
    return [abs(float(v[i]) - ex[i]) for i in range(3)]

def timeit(f, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = f(); ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))

# ---------------- E1: finite differences, call and digital, T = 1
Ns = [100, 200, 400, 800, 1600, 3200]
res['fd'] = {}
for c in ['call', 'digital']:
    for sch in ['BE', 'CN', 'CNR']:
        rows = []
        for N in Ns:
            v, t = timeit(lambda: fd_price(c, sch, N, N, **P), reps=3 if N <= 800 else 1)
            rows.append([N] + err(v, EX[c]) + [t])
        res['fd'][f'{c}_{sch}'] = rows
print('E1 done')

# ---------------- E1b: coarse time stepping (M = 25 fixed), spatial refinement; Gamma robustness
res['fixedM'] = {}
for c, exf in [('digital', digital_exact), ('call', call_exact)]:
    for sch in ['BE', 'CN', 'CNR']:
        rows = []
        for N in [200, 400, 800, 1600, 3200]:
            Sg, V = fd_price(c, sch, N, 25, return_grid=True, **P)
            h = Sg[1] - Sg[0]
            G = (V[2:] - 2 * V[1:-1] + V[:-2]) / h ** 2
            m = (Sg[1:-1] >= 80) & (Sg[1:-1] <= 120)
            exg = np.array([exf(x, 100., .05, .2, 1.)[2] for x in Sg[1:-1][m]])
            v = fd_price(c, sch, N, 25, **P)
            rows.append([N] + err(v, EX[c]) + [float(np.max(np.abs(G[m] - exg)))])
        res['fixedM'][f'{c}_{sch}'] = rows
res['gamma_profile'] = {}
for sch in ['CN', 'CNR']:
    Sg, V = fd_price('digital', sch, 1600, 25, return_grid=True, **P)
    h = Sg[1] - Sg[0]
    G = (V[2:] - 2 * V[1:-1] + V[:-2]) / h ** 2
    m = (Sg[1:-1] >= 85) & (Sg[1:-1] <= 115)
    res['gamma_profile'][sch] = [Sg[1:-1][m].tolist(), G[m].tolist()]
Sx = np.linspace(85, 115, 301)
res['gamma_profile']['exact'] = [Sx.tolist(), [float(digital_exact(x, 100., .05, .2, 1.)[2]) for x in Sx]]
print('E1b done')

# ---------------- E2: COS
res['cos'] = {}
for c in ['call', 'digital']:
    rows = []
    for Nt in [8, 16, 32, 64, 128, 256]:
        v, t = timeit(lambda: cos_price(c, Nt, **P), reps=20)
        rows.append([Nt] + err(v, EX[c]) + [t])
    res['cos'][c] = rows
print('E2 done')

# ---------------- E3: CRR trees
res['crr'] = {}
for c in ['call', 'digital', 'dao']:
    rows = []
    for n in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
        v, t = timeit(lambda: crr(c, n, B=B, **P), reps=3 if n <= 1600 else 1)
        rows.append([n] + err(v, EX[c]) + [t])
    res['crr'][c] = rows
# oscillation: price error for consecutive n
res['crr_osc'] = {}
for c in ['call', 'dao']:
    ns = list(range(100, 301))
    res['crr_osc'][c] = [ns, [float(crr(c, n, B=B, **P)[0] - EX[c][0]) for n in ns]]
print('E3 done')

# ---------------- E4: Monte Carlo (RMSE over 20 seeds)
res['mc'] = {}
for c in ['call', 'digital', 'dao']:
    rows = []
    sizes = [1_000, 10_000, 100_000, 1_000_000] if c != 'dao' else [1_000, 10_000, 100_000]
    for npth in sizes:
        outs, ts = [], []
        for sd in range(20):
            t0 = time.perf_counter()
            o = mc(c, npth, B=B, nsteps=100, seed=sd, **P); ts.append(time.perf_counter() - t0)
            outs.append(o)
        outs = np.array(outs)
        rmse = [float(np.sqrt(np.mean((outs[:, i] - EX[c][i]) ** 2))) for i in range(3)]
        rows.append([npth] + rmse + [float(np.median(ts))])
    res['mc'][c] = rows
print('E4 done')

# ---------------- E5: barrier with FD (barrier on node) -- already have dao exact
res['fd_dao'] = {}
for sch in ['CN', 'CNR']:
    rows = []
    for N in [31, 62, 124, 248, 496, 992]:     # h = 310/N -> S0 = 100 on a node when N % 31 == 0
        v, t = timeit(lambda: fd_price('dao', sch, N, N, B=B, **P), reps=1)
        rows.append([N] + err(v, EX['dao']) + [t])
    res['fd_dao'][sch] = rows
print('E5 done')

# ---------------- E6: American put
# Primary reference: independent fine-grid FD (CN-Rannacher + penalty, N = M = 6400 and 12800,
# Richardson-extrapolated; produced by amref_fd.py). Cross-check: BBSR with 20,000 steps.
_fd = json.load(open('amref_fd.json'))
AM = [float(x) for x in _fd['fd_rich'][20]]          # S0 = 100 is index 20 of S = 80..120
bb = bbsr_amput(20000, **P)
res['amput_ref'] = AM
res['amput_ref_bbsr'] = [float(x) for x in bb]
res['amput_ref_diff'] = [abs(float(x) - y) for x, y in zip(bb, AM)]
res['amput'] = {}
for sch in ['CN', 'CNR']:
    rows = []
    for N in [100, 200, 400, 800, 1600]:
        v, t = timeit(lambda: fd_price('amput', sch, N, N, **P), reps=1)
        rows.append([N] + err(v, AM) + [t])
    res['amput']['FD_' + sch] = rows
for name, f in [('CRR', lambda n: crr('amput', n, **P)), ('BBSR', lambda n: bbsr_amput(n, **P))]:
    rows = []
    for n in [50, 100, 200, 400, 800, 1600, 3200]:
        v, t = timeit(lambda: f(n), reps=1)
        rows.append([n] + err(v, AM) + [t])
    res['amput'][name] = rows
print('E6 done')


# ---------------- E7: robustness of the Gamma result across volatility and maturity (N = 1600, M = 25)
out = []
for sg in [0.1, 0.2, 0.4]:
    for TT in [0.25, 1.0]:
        row = [sg, TT]
        for c, exf in [('call', call_exact), ('digital', digital_exact)]:
            for sch in ['CN', 'CNR']:
                Sg, V = fd_price(c, sch, 1600, 25, s=sg, T=TT, return_grid=True)
                h = Sg[1] - Sg[0]; G = (V[2:] - 2 * V[1:-1] + V[:-2]) / h ** 2
                m = (Sg[1:-1] >= 80) & (Sg[1:-1] <= 120)
                ex = np.array([exf(x, 100., .05, sg, TT)[2] for x in Sg[1:-1][m]])
                row.append(float(np.max(np.abs(G[m] - ex))))
            row.append(float(np.max(np.abs(ex))))
        out.append(row)
res['robust'] = out

json.dump(res, open('results.json', 'w'), indent=1)
print('saved')
