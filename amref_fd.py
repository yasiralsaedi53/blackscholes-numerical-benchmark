"""Independent American-put reference: CN-Rannacher + penalty on fine grids, Richardson-extrapolated.
Grid S in [0, 400]; with N a multiple of 400 every integer spot 80..120 is a node."""
import json, time, numpy as np
from methods import fd_price
P = dict(K=100., r=.05, s=.2, T=1.)
S = np.arange(80., 121.)
out = {}
for N in [3200, 6400, 12800]:
    t0 = time.perf_counter()
    Sg, V = fd_price('amput', 'CNR', N, N, return_grid=True, **P)
    h = Sg[1] - Sg[0]; i = np.rint(S / h).astype(int); assert np.allclose(Sg[i], S)
    out[N] = np.stack([V[i], (V[i+1]-V[i-1])/(2*h), (V[i+1]-2*V[i]+V[i-1])/h**2], 1)
    print(N, round(time.perf_counter()-t0, 1), 's', out[N][20])
rich = (4 * out[12800] - out[6400]) / 3
json.dump({'S': S.tolist(), 'fd3200': out[3200].tolist(), 'fd6400': out[6400].tolist(),
           'fd12800': out[12800].tolist(), 'fd_rich': rich.tolist()}, open('amref_fd.json', 'w'), indent=1)
