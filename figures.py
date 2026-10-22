import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = json.load(open('results.json'))
C = {'CNR': '#2a78d6', 'CN': '#eb6834', 'BE': '#1baf7a', 'CRR': '#eda100', 'COS': '#e87ba4',
     'MC': '#008300', 'BBSR': '#4a3aa7', 'exact': '#0b0b0b'}
MK = {'CNR': 'o', 'CN': 's', 'BE': '^', 'CRR': 'D', 'COS': 'v', 'MC': 'P', 'BBSR': 'X'}
LBL = {'CNR': 'FD Crank–Nicolson + Rannacher', 'CN': 'FD Crank–Nicolson', 'BE': 'FD backward Euler',
       'CRR': 'CRR binomial tree', 'COS': 'COS (Fourier cosine)', 'MC': 'Monte Carlo (RMSE)',
       'BBSR': 'Binomial BS + Richardson'}
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'axes.edgecolor': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'lines.markersize': 5, 'legend.frameon': False, 'grid.color': '#e4e3df'})

def style(ax):
    ax.grid(True, which='major', lw=0.6)

# Figure 1: work–precision, call and digital
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=220)
for ax, c, title in zip(axs, ['call', 'digital'], ['European call', 'Cash-or-nothing digital call']):
    series = [('CNR', r['fd'][f'{c}_CNR']), ('CRR', r['crr'][c]), ('COS', r['cos'][c]), ('MC', r['mc'][c])]
    for key, rows in series:
        rows = np.array(rows)
        e = np.maximum(rows[:, 1], 1e-16)
        ax.loglog(rows[:, 4], e, marker=MK[key], color=C[key], label=LBL[key])
    # reference slopes in error vs CPU time: FD with N = M costs O(N^2), so order 2 in N gives slope -1;
    # Monte Carlo (and CRR on smooth payoffs) give slope -1/2
    t0 = np.array([1e-3, 1e-1]); y0 = 1e-8
    ax.loglog(t0, y0 * (t0 / t0[0]) ** -1, ls='--', lw=0.9, color='#8a8984')
    ax.loglog(t0, y0 * (t0 / t0[0]) ** -0.5, ls=':', lw=0.9, color='#8a8984')
    ax.text(t0[1] * 1.15, y0 * 1e-2, 'Slope −1', fontsize=7, color='#52514e', va='center')
    ax.text(t0[1] * 1.15, y0 * 10 ** -1, 'Slope −½', fontsize=7, color='#52514e', va='center')
    ax.text(t0[0], y0 * 4, 'Reference slopes', fontsize=6.5, color='#52514e', va='bottom')
    ax.axhspan(1e-17, 1e-14, color='#e4e3df', alpha=0.6, lw=0)
    ax.text(1.1e-3, 3e-15, 'Double-precision round-off', fontsize=6.5, color='#52514e', va='center')
    ax.set_ylim(1e-17, 1e2)
    ax.set_title(title, fontsize=9.5); ax.set_xlabel('CPU time per valuation (s)'); style(ax)
axs[0].set_ylabel('Absolute price error')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig('fig1_work_precision.png'); plt.close(fig)

# Figure 2: Gamma robustness with coarse time steps (M = 25)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=220)
for ax, c, title in zip(axs, ['call', 'digital'], ['European call', 'Digital call']):
    for key in ['BE', 'CN', 'CNR']:
        rows = np.array(r['fixedM'][f'{c}_{key}'])
        ax.loglog(rows[:, 0], rows[:, 4], marker=MK[key], color=C[key], label=LBL[key])
    ax.set_title(title, fontsize=9.5); ax.set_xlabel('Spatial intervals N (M = 25 time steps)'); style(ax)
axs[0].set_ylabel('Max |Gamma error| on S ∈ [80, 120]')
axs[0].legend(fontsize=7.5, loc='upper left')
fig.tight_layout(); fig.savefig('fig2_gamma_robustness.png'); plt.close(fig)

# Figure 3: Gamma profile of the digital, N = 1600, M = 25
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=220)
gp = r['gamma_profile']
ax.plot(gp['CN'][0], gp['CN'][1], color=C['CN'], lw=1.0, label=LBL['CN'])
ax.plot(gp['CNR'][0], gp['CNR'][1], color=C['CNR'], lw=2.0, label=LBL['CNR'])
ax.plot(gp['exact'][0], gp['exact'][1], color=C['exact'], lw=1.0, ls='--', label='Exact')
lim = 10 ** np.ceil(np.log10(np.max(np.abs(gp['CN'][1]))))
ax.set_yscale('symlog', linthresh=1e-3, linscale=1.0); ax.set_ylim(-lim, lim)
ax.axhspan(-1e-3, 1e-3, color='#e4e3df', alpha=0.5, lw=0)
ax.set_xlabel('Underlying price S'); ax.set_ylabel('Gamma (symmetric log scale)')
ax.set_title('Digital call Gamma, N = 1600, M = 25', fontsize=9.5)
style(ax); ax.legend(fontsize=7.5, loc='lower left')
fig.tight_layout(); fig.savefig('fig3_gamma_profile.png'); plt.close(fig)

# Figure 4: CRR price error by number of steps (odd-even and barrier oscillation)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=220)
for ax, c, title in zip(axs, ['call', 'dao'], ['European call', 'Down-and-out call (B = 90)']):
    ns, e = np.array(r['crr_osc'][c][0]), np.array(r['crr_osc'][c][1])
    if c == 'call':
        ev, od = ns % 2 == 0, ns % 2 == 1
        ax.plot(ns[ev], e[ev], color=C['CRR'], lw=1.4, label='Even n')
        ax.plot(ns[od], e[od], color='#4a3aa7', lw=1.4, ls='--', label='Odd n')
        ax.legend(fontsize=7.5, loc='lower right')
    else:
        ax.plot(ns, e, color=C['CRR'], lw=1.0, marker='.', ms=2.5)
    ax.axhline(0, color='#52514e', lw=0.6)
    ax.set_title(title, fontsize=9.5); ax.set_xlabel('Tree steps n'); style(ax)
axs[0].set_ylabel('Signed price error')
fig.tight_layout(); fig.savefig('fig4_crr_oscillation.png'); plt.close(fig)

# Figure 5: American put, error vs CPU time
fig, ax = plt.subplots(figsize=(5.4, 3.3), dpi=220)
for key, name in [('CN', 'FD_CN'), ('CRR', 'CRR'), ('BBSR', 'BBSR')]:
    rows = np.array(r['amput'][name])
    lab = 'FD Crank–Nicolson + penalty iteration' if key == 'CN' else LBL[key]
    ax.loglog(rows[:, 4], rows[:, 1], marker=MK[key], color=C[key], label=lab)
ax.set_xlabel('CPU time per valuation (s)'); ax.set_ylabel('Absolute price error')
ax.set_title('American put', fontsize=9.5); style(ax); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig('fig5_american.png'); plt.close(fig)
print('figures done')
