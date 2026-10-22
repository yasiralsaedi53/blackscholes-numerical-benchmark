"""Recompute American-put errors against the independent FD reference (amref_fd.json)."""
import json, numpy as np
from methods import fd_price, crr, bbsr_amput
P = dict(K=100., r=.05, s=.2, T=1.)
a = json.load(open('amref_fd.json')); FD = np.array(a['fd_rich']); S = np.array(a['S'])
BB = np.array(json.load(open('results_range.json'))['amput_ref'])
out = {'ref_S0': {'fd_richardson': FD[20].tolist(), 'bbsr20000': BB[20].tolist()}}
# single point
vals = {}
for N in [100, 200, 400, 800, 1600]:
    vals[('FD_CN', N)] = fd_price('amput', 'CN', N, N, **P)
for n in [50, 100, 200, 400, 800, 1600, 3200]:
    vals[('CRR', n)] = crr('amput', n, **P); vals[('BBSR', n)] = bbsr_amput(n, **P)
out['single'] = {f'{k[0]}_{k[1]}': {'vs_fd': np.abs(np.array(v[:3]) - FD[20]).tolist(),
                                    'vs_bbsr': np.abs(np.array(v[:3]) - BB[20]).tolist()} for k, v in vals.items()}
# range
Sg, V = fd_price('amput', 'CNR', 400, 400, return_grid=True, **P); h = Sg[1]-Sg[0]; i = np.rint(S/h).astype(int)
fdv = np.stack([V[i], (V[i+1]-V[i-1])/(2*h), (V[i+1]-2*V[i]+V[i-1])/h**2], 1)
crrv = np.array([crr('amput', 3200, S0=x, **P)[:3] for x in S]); bbv = np.array([bbsr_amput(800, S0=x, **P)[:3] for x in S])
def met(v, ref):
    e = v - ref
    return {'price_MAE': float(np.mean(abs(e[:, 0]))), 'price_MAPE': float(100*np.mean(abs(e[:, 0]/ref[:, 0]))),
            'price_Max': float(np.max(abs(e[:, 0]))), 'delta_MAE': float(np.mean(abs(e[:, 1]))),
            'gamma_MAE': float(np.mean(abs(e[:, 2]))), 'gamma_Max': float(np.max(abs(e[:, 2]))),
            'gamma_Max_S>=82': float(np.max(abs(e[2:, 2])))}
out['range'] = {nm: {'vs_fd': met(v, FD), 'vs_bbsr': met(v, BB)} for nm, v in [('FD', fdv), ('CRR', crrv), ('BBSR', bbv)]}
json.dump(out, open('amref_check.json', 'w'), indent=1)
for k, v in out['single'].items():
    if any(k.endswith(s) for s in ['_400', '_800', '_1600', '_3200']):
        print(k, ' '.join(f'{x:.2e}' for x in v['vs_fd']), '|', ' '.join(f'{x:.2e}' for x in v['vs_bbsr']))
for k, v in out['range'].items():
    print(k, {kk: f'{vv:.2e}' for kk, vv in v['vs_fd'].items()})
    print('  bb', {kk: f'{vv:.2e}' for kk, vv in v['vs_bbsr'].items()})
