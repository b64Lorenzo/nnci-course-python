# NNCI Course — Python

A minimal Python version of Assignment I: the Rosenblatt perceptron, capacity/strength studies, and plots.

## Run

```sh
python main.py perceptron --help
python main.py perceptron --dimensions 20 --patterns 15 --seed 2026
python main.py perceptron --capacity --dimensions 20 --datasets 50 --max-epochs 100
python main.py perceptron --strength --dimensions 20 --patterns 15 --datasets 50
```

Capacity and strength plots require Matplotlib: `python -m pip install matplotlib`.
Capacity CSVs include `Q_ls_variance_est`, exact and CLT `P_ls` values, and 95% Wilson bounds. Plots mark the `α=2` transition and exact `P=2N` value `P_ls=1/2`.

## Files

- `A1assignment/main.py` — Assignment I parser and entry point.
- `A1assignment/runs.py` — single run, studies, and CSV output.
- `A1assignment/perceptron_rosenblatt.py` — data generation and Rosenblatt training.
- `A1assignment/plots.py` — capacity and embedding-strength plots.
- `logger.py` — shared timestamped console/file logger.
- `main.py` — compact top-level entry point.

Results are written to `results/`; run logs go to `logs/`. No package-build setup is needed. Run the numerical validation with `python -m unittest discover -s tests -v`.
