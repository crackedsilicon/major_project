# Adaptive Federated Aggregation — Setup & Run Instructions

This file contains reproducible setup and run commands so another person or an AI can set up and run the project locally on Windows (PowerShell). Adapt paths and Python executable as needed.

Prerequisites
- Python 3.9+ installed. (Project tested with Python 3.14 on this machine.)
- Optional: CUDA-enabled GPU and matching PyTorch build.

1) Create and activate a virtual environment (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Notes on PyTorch:
- On a machine without GPU, install the CPU-only wheel. Example:
	```powershell
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	```
- On a CUDA machine, follow https://pytorch.org for the matching CUDA version and run the recommended `pip` or `conda` command.

3) Prepare datasets

- Place raw dataset files (TON_IoT, IoT-NID, Edge-IIoTset) under `data/raw/`.
- Use the data preparation script (stub) to convert datasets into `data/processed/`:

```powershell
python scripts/prepare_data.py --dataset TON_IoT --out data/processed
```

Note: `scripts/prepare_data.py` is currently a placeholder. Replace with dataset-specific parsing logic.

Detailed dataset guidance:
- For TON_IoT: extract the downloaded archive and put CSVs under `data/raw/TON_IoT/` (you may keep nested folders; the prep script searches recursively).
- If CSVs contain mixed types or missing labels, the prep script will drop rows with missing labels, numeric-only features are selected automatically, and string labels are factorized to integer class ids; mapping is saved to `data/processed/<dataset>/meta.npz`.
- Use `--n_clients`, `--split iid|dirichlet`, and `--alpha` to control client partitioning; defaults are sensible for quick experiments.

4) Run a quick baseline simulation

```powershell
python scripts/run_baseline.py --rounds 3 --clients 3 --epochs 1
```

This runs a small federated simulation on synthetic data to verify the pipeline.

6) Run unit tests

```powershell
python -m pytest -q
```

7) Run experiments

- Example (no compression):

```powershell
python experiments/runner.py --dataset synthetic --rounds 5 --epochs 1 --exp synthetic_exp
```

- Example (estimate compressed communication using float16):

```powershell
python experiments/runner.py --dataset synthetic --rounds 5 --epochs 1 --exp synthetic_exp_compressed --compress
```

Experiment CLI (key args):
- `--dataset`: dataset folder name under `data/processed/` (required).
- `--rounds`: number of federated rounds to simulate.
- `--epochs`: local epochs per client per round.
- `--exp`: results folder name under `results/`.
- `--aggregator`: aggregator strategy: `adaptive`, `fedavg`, `weighted`, `fedprox`.
- `--mu`: FedProx proximal coefficient (used when `--aggregator fedprox`).
- `--compress`: estimate communication after float16 quantization.

Examples:
```powershell
python experiments/runner.py --dataset TON_IoT --rounds 10 --epochs 1 --exp TON_IoT_adaptive --aggregator adaptive
python experiments/runner.py --dataset TON_IoT --rounds 10 --epochs 1 --exp TON_IoT_fedprox --aggregator fedprox --mu 0.1
python experiments/runner.py --dataset TON_IoT --rounds 10 --epochs 1 --exp TON_IoT_compressed --compress
```

6) Running experiments

- Implement experiment configurations under `experiments/` and call `experiments/runner.py`.

7) Notes for sharing with AI or collaborators
- Provide the full repository and specify which dataset files are required.
- If an AI is instructed to run experiments, point it to `SETUP.md` then `IMPLEMENTATION_PLAN.md` for objectives and milestones.
- For reproducible results, include `configs/*.yaml` files with seeds, dataset splits, and hyperparameters.

Troubleshooting
- If you encounter `ValueError: Object arrays cannot be loaded when allow_pickle=False` when loading `.npz` files, ensure `scripts/prepare_data.py` produced numeric `y` arrays (categorical labels are factorized). Re-run the prepare step if needed.
- If `IndexError: Target -1 is out of bounds` occurs during training, it indicates missing labels or an encoding issue—re-check the CSV label column and re-run prepare.
- For PowerShell quoting issues, prefer using helper scripts under `scripts/` (e.g., `scripts/run_toniot_smoke.py`) instead of passing complex `-c` strings.

Next steps (recommended):
- Run the provided small TON_IoT smoke run to validate your environment:
	```powershell
	python scripts/run_toniot_smoke.py
	```
- Run the class-incremental simulation demo:
	```powershell
	python scripts/run_cil_demo.py
	```
- Run the multi-seed baseline comparison:
	```powershell
	python scripts/run_multi_seed_comparison.py --dataset synthetic --rounds 3 --seeds 0 1 2
	```
- After validation, run the full comparative experiments on TON_IoT (10+ rounds) and generate plots using `scripts/plot_results.py` or the helper `scripts/plot_toniot_small.py`.

Contact / Next steps
- If you want me to run experiments here, confirm which experiment settings to use (aggregators, rounds, compression). I can run and produce comparison plots and a draft report.

Contact / Next steps
- To continue implementation: add dataset parsers, integrate Class-Incremental Learning methods (MEMENTO+, iCaRL+), and run federated experiments using `server/aggregator.aggregate`.

Running from a zipped package (venv + dataset included)

Important portability note
- A bundled virtual environment is operating-system and Python-version specific. A `.venv` created on Windows will usually not work on Linux/macOS, and even different Python minor versions can break it.
- For maximum reliability, recreate the venv on the target machine using the commands below. Only use the included `.venv` if the recipient is on the exact same OS and Python version.

1) Extract the zip

```powershell
Expand-Archive -Path major_project.zip -DestinationPath C:\path\to\major_project
Set-Location C:\path\to\major_project
```

2) Option A — Use the included virtual environment (only if same OS and Python)

```powershell
.\.venv\Scripts\Activate.ps1
python --version
pip --version
```

Then run the smoke test:

```powershell
python scripts/run_toniot_small_smoke.py
```

3) Recommended — Recreate the environment on the target machine

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_toniot_small_smoke.py
```

4) If the included venv does not activate, call the Python interpreter directly

```powershell
.\.venv\Scripts\python.exe scripts\run_toniot_small_smoke.py
```

5) Dataset path notes
- The project expects processed datasets to be under `data/processed/<dataset>/clients/` and metadata under `data/processed/<dataset>/meta.npz`.
- If your ZIP includes raw data and prepared data, no extra setup is needed. If it contains only raw CSVs, run the preparation script first:

```powershell
python scripts/prepare_data.py --dataset TON_IoT --out data/processed
```

6) Sharing recommendations
- Keep the zip small by excluding large `.venv` folders and raw datasets when possible. Host large files separately on Drive/S3/OneDrive and include download instructions in `SETUP.md`.
- Include `requirements.txt`, `SETUP.md`, `docs/implementation_details.md`, and the helper scripts in `scripts/`.
- For a very large project, prefer GitHub or a release archive instead of a raw zip with everything included.

7) Risk warning
- Do not run a zip received from an unknown source without checking the contents. A venv may contain binaries or scripts that behave unexpectedly on another machine.

This section is intended for distributing the project in a zip package while keeping it easy to run on another Windows PC.
