# Black–Scholes Numerical Benchmark

Code and results for the paper:

**Pricing Accuracy and Greek Stability in the Black–Scholes Model: A Reproducible Benchmark of Finite-Difference, Lattice, Monte Carlo, and Fourier Cosine Methods**

The benchmark compares finite-difference (Crank–Nicolson, Rannacher, backward Euler), binomial lattice (CRR, BBSR), Monte Carlo, and Fourier cosine (COS) methods for European, digital, down-and-out barrier, and American options, reporting errors in price, Delta, and Gamma together with computing time.

## Requirements

Python 3.11 or later with NumPy, SciPy, and Matplotlib (all preinstalled in Google Colab).

```bash
pip install -r requirements.txt
```

## How to reproduce the results

Run the scripts in this order (about 3 minutes in total):

| Step | Command | Produces |
|---|---|---|
| 1 | `python amref_fd.py` | Independent American-put reference → `amref_fd.json` |
| 2 | `python experiments.py` | Tables 1–5 → `results.json` |
| 3 | `python figures.py` | Figures 1–5 → `fig*.png` |
| 4 | `python range_metrics.py` | Table 6 (MAE, RMSE, MAPE, MaxAE across spot prices) → `results_range.json` |
| 5 | `python amref_check.py` | Optional: American errors against both references → `amref_check.json` |

### Google Colab

```python
!git clone https://github.com/yasiralsaedi53/blackscholes-numerical-benchmark
%cd blackscholes-numerical-benchmark
!python amref_fd.py
!python experiments.py
!python figures.py
!python range_metrics.py
```

## Files

| File | Content |
|---|---|
| `methods.py` | All pricing methods and closed-form references |
| `experiments.py` | Convergence, cost, coarse-time-step, barrier, and American experiments |
| `range_metrics.py` | Evaluation across 41 spot prices |
| `figures.py` | All figures in the paper |
| `amref_fd.py`, `amref_check.py` | American-put reference and cross-check |
| `*.json` | Saved results used in the paper |
| `README.txt` | Detailed notes, settings, and change log |

## Reproducibility

Errors, convergence orders, and Greeks are deterministic (fixed Monte Carlo seeds). Computing times depend on the machine; rankings and slopes should not.

## License

MIT License. See `LICENSE`.
