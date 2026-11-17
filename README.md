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

## Files

- `A1assignment/main.py` — Assignment I parser and entry point.
- `A1assignment/runs.py` — single run, studies, and CSV output.
- `A1assignment/perceptron_rosenblatt.py` — data generation and Rosenblatt training.
- `A1assignment/plots.py` — capacity and embedding-strength plots.
- `logger.py` — shared timestamped console/file logger.
- `main.py` — compact top-level entry point.

Results are written to `results/`; run logs go to `logs/`. This project intentionally has no package-build setup or test suite.
