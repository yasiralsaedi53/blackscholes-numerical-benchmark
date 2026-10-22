Black–Scholes numerical benchmark: code and results
====================================================

Requirements
  Python 3.11 or later; NumPy, SciPy, Matplotlib (all preinstalled in Google Colab).
  Results in this package were produced with Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1,
  Matplotlib 3.10.9 on one core of an Intel Xeon 2.1 GHz processor.

Run in this order (total about 3 minutes)
  1. python3 amref_fd.py        # independent American-put reference (fine-grid FD + Richardson) -> amref_fd.json
  2. python3 experiments.py     # Tables 1-5 (Table 3 under 'robust')                           -> results.json
  3. python3 figures.py         # Figures 1-5                                                      -> fig*.png
  4. python3 range_metrics.py   # Table 6: MAE, RMSE, MAPE, MaxAE across spot prices               -> results_range.json
  5. python3 amref_check.py     # optional: American errors against both references               -> amref_check.json

Where each table and figure comes from
  Table 1  results.json: fd, crr, mc, fd_dao, amput (observed orders)
  Table 2  results.json: fd, crr, cos, mc at about 10 ms per valuation
  Table 3  results.json: robust
  Table 4  results.json: fd_dao, crr, mc['dao']
  Table 5  results.json: amput (reference: amref_fd.json; cross-check: amput_ref_bbsr)
  Table 6  results_range.json
  Figures  figures.py, reading results.json

Experimental settings
  S0 = K = 100, r = 0.05, sigma = 0.2, T = 1, barrier B = 90 (continuous monitoring).
  Monte Carlo: antithetic variates; RMSE over 20 seeds at S0 = 100 (Tables 1, 2, 4);
  one independent seed per spot price (seed = spot value) for Table 6.
  Barrier Monte Carlo: 100 time steps with Brownian-bridge survival weights.

Reproducibility
  Errors, convergence orders and Greeks are deterministic (fixed Monte Carlo seeds).
  Computing times depend on the machine; rankings and slopes should not.

Change log
  v2: Barrier Monte Carlo Greeks (methods.py, function mc, contract 'dao') now include the
      explicit derivatives of the first-interval Brownian-bridge survival factor with respect
      to S0, in addition to the likelihood-ratio weights of the first increment. The earlier
      estimator omitted these terms and was biased near the barrier (at S0 = 91: Delta 0.565
      vs exact 0.920; now 0.916). Affected output: Table 6, barrier Monte Carlo row
      (Delta MAE 2.85e-2, Gamma MAE 1.70e-2, Gamma MaxAE 5.39e-2). Table 4 (S0 = 100)
      changes only beyond the third significant figure.
      Figure scripts updated (reference slopes, symmetric-log Gamma axis, even/odd n).
