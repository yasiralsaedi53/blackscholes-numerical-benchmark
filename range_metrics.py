"""Extra experiment: errors across a range of spot prices (MAE, RMSE, MAPE, MaxAE).
Same computational settings as Table 2 / Tables 4-5 of the manuscript."""
import json, time
import numpy as np
from methods import *

K, r, s, T, B = 100., .05, .2, 1., 90.
P = dict(K=K, r=r, s=s, T=T)

def metrics(num, ex, pct=True):
    num, ex = np.asarray(num), np.asarray(ex)
    e = num - ex
    out = {'MAE': float(np.mean(np.abs(e))), 'RMSE': float(np.sqrt(np.mean(e * e))), 'MaxAE': float(np.max(np.abs(e)))}
    out['MAPE'] = float(100 * np.mean(np.abs(e / ex))) if pct else None
    return out

def fd_grid(contract, N, M, S_eval):
    Sg, V = fd_price(contract, 'CNR', N, M, B=B, return_grid=True, **P)
    h = Sg[1] - Sg[0]
    idx = np.rint((S_eval - Sg[0]) / h).astype(int)
    assert np.allclose(Sg[idx], S_eval)
    return V[idx], (V[idx + 1] - V[idx - 1]) / (2 * h), (V[idx + 1] - 2 * V[idx] + V[idx - 1]) / h ** 2

def pointwise(f, S_eval):
    out = np.array([f(x)[:3] for x in S_eval]); return out[:, 0], out[:, 1], out[:, 2]

res = {}
S_main = np.arange(80., 121.)          # 41 spots
S_dao = np.arange(91., 121.)           # 30 spots above the barrier

def run(contract, S_eval, exact_f, methods):
    ex = np.array([exact_f(x) for x in S_eval]); exP, exD, exG = ex[:, 0], ex[:, 1], ex[:, 2]
    res[contract] = {}
    for name, fn in methods:
        t0 = time.perf_counter(); p, d, g = fn(S_eval); t = time.perf_counter() - t0
        res[contract][name] = {'price': metrics(p, exP), 'delta': metrics(d, exD, pct=np.all(np.abs(exD) > 1e-3)),
                               'gamma': metrics(g, exG, pct=False), 'time_total_s': t, 'n_points': len(S_eval)}
    print(contract, 'done')

cos_vec = lambda c, S_eval: pointwise(lambda x: cos_price(c, 64, S0=x, **P), S_eval)
mc_vec = lambda c, S_eval, **kw: pointwise(lambda x: mc(c, 100_000, S0=x, seed=int(x), B=B, **kw, **P), S_eval)

run('call', S_main, lambda x: call_exact(x, K, r, s, T), [
    ('FD CN + Rannacher (N=M=400)', lambda S: fd_grid('call', 400, 400, S)),
    ('CRR (n=800)', lambda S: pointwise(lambda x: crr('call', 800, S0=x, **P), S)),
    ('COS (64 terms)', lambda S: cos_vec('call', S)),
    ('Monte Carlo (1e5 paths)', lambda S: mc_vec('call', S))])
run('digital', S_main, lambda x: digital_exact(x, K, r, s, T), [
    ('FD CN + Rannacher (N=M=400)', lambda S: fd_grid('digital', 400, 400, S)),
    ('CRR (n=800)', lambda S: pointwise(lambda x: crr('digital', 800, S0=x, **P), S)),
    ('COS (64 terms)', lambda S: cos_vec('digital', S)),
    ('Monte Carlo (1e5 paths)', lambda S: mc_vec('digital', S))])
run('dao', S_dao, lambda x: dao_exact(x, K, B, r, s, T), [
    ('FD CN + Rannacher (N=M=620)', lambda S: fd_grid('dao', 620, 620, S)),
    ('CRR (n=800)', lambda S: pointwise(lambda x: crr('dao', 800, S0=x, B=B, **P), S)),
    ('Monte Carlo + bridge (1e5 paths)', lambda S: mc_vec('dao', S, nsteps=100))])
ref = np.array(json.load(open('amref_fd.json'))['fd_rich'])   # independent FD reference, S = 80..120
res['amput_ref'] = ref.tolist()
res['amput_ref_bbsr'] = [list(bbsr_amput(20000, S0=x, **P)[:3]) for x in S_main]
run('amput', S_main, lambda x: ref[int(x - 80)], [
    ('FD CN + penalty (N=M=400)', lambda S: fd_grid('amput', 400, 400, S)),
    ('CRR (n=3200)', lambda S: pointwise(lambda x: crr('amput', 3200, S0=x, **P), S)),
    ('BBSR (n=800)', lambda S: pointwise(lambda x: bbsr_amput(800, S0=x, **P), S))])
json.dump(res, open('results_range.json', 'w'), indent=1)
