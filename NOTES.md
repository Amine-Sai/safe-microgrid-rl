# Project notes and decision log

**Status:** prototype snapshot, tagged `v0-prototype`. The study is complete enough
to report its negative result; the code is a research prototype, not a maintained
library. This file records the decisions that are not obvious from the README and
states exactly what has (and has not) been reproduced.

The working copy used a longer decision log during the project. This public file
indexes the entries that are referenced from code and comments.

---

## D1 — Reproduction scope

Verified in a fresh Python 3.14.7 virtual environment built from
`requirements.txt`:

- The dataset was downloaded (OPSD 124.3 MB, PVGIS 1.2 MB, Open-Meteo 0.8 MB)
  and `python data/prepare.py` built `data/processed/{train,val,test}.parquet`
  (17,472 / 4,368 / 4,032 rows).
- **Full test suite: 32/32 pass** with `python -m pytest tests/ -v`. This
  includes `test_ppo.py` and every `test_config.py` case; no tests were skipped.
- A short PPO smoke run (`experiments/train.py --config bench10k --seed 101`)
  completed end to end, wrote the result JSON and step checkpoints, and
  evaluated on Val. The published 300k-step results were not re-run.
- The report pipeline (`tools/report/export_csv.py` then `plots.py`) was run
  against the committed run JSONs: the CSVs reproduce cleanly and the four
  README figures are pixel-identical. The committed results remain the artefacts
  of the original runs.

The CUDA-linked wheel named by `requirements.txt` (`torch==2.14.1`) was not
installed here; verification used the CPU-only build `torch==2.14.1+cpu` from
`https://download.pytorch.org/whl/cpu` (187 MB), which needs no NVIDIA runtime.
The upstream version is the same.

## D2 — Dependency pins regenerated from the producing environment

The original pins (`numpy 1.26.4`, `torch 2.3.1`, `pytest 8.2.2`) did **not**
match the environment that produced the results. `requirements.txt` now records
the versions that did. See the header of that file for the verification caveat
above.

---

## Design decisions (evidence in this repo)

- **Peak-reduction target, not a hardware rating.** Grid import is aimed at
  **368.153918 kW**, derived as the P90 of Grid-Only import on the Train split
  (`max(0, load − PV)` with the battery idle) in `data/prepare.py` and persisted to
  `data/processed/split_report.json`; `env/smartgrid_env.py::resolve_grid_import_cap`
  reads it and there is no hardcoded fallback. It is a soft constraint with a
  violation budget (`cost_limit`), not a physical connection limit: 10.0 % of Train
  hours and 10.7 % of Val hours exceed it with the battery idle, and 6 of 104 Train
  weeks need more peak-shaving energy than the battery can deliver at one full
  cycle per day. Economic sizing was considered as an alternative derivation and
  rejected: at ~10 EUR/kW-month, capping at 368 kW removes only ~3.6 kEUR/year of
  demand charge (721.9 kW-month over the two Train years) against a 500 kWh battery,
  so an economically optimal cap would sit *above* P90 rather than justify it.
- **Strong comparator.** The reference is a tuned peak-shaving rule (PSM'), not
  an untuned baseline. The negative result is largely about the strength of that
  comparator and the presence of the grid-import cap; see the README takeaway.
- **Reward scale.** Learning targets use `c = 1/mean|r| = 0.096879` computed on
  Grid-Only Train (`experiments/train_core.py::grid_only_reward_scale`). Metrics
  stay in unscaled EUR/kW.
- **Data splits by whole weeks.** Train 2017–2018, Val 2019 H1, Test 2019 H2.
  The Test split is closed and is not read by any code in this release.
- **Evaluation protocol.** Deterministic evaluation on all 26 Val weeks,
  seed-frozen normaliser, 5 seeds per method. Wall-clock and per-eval metrics are
  in `results/runs/` and `results/tables/runs.csv`.
- **Tracked results.** `results/` is committed on purpose: 4 figures, 20 seed
  run JSONs, 4 batch roll-ups and 4 CSV tables (~2.4 MB). See
  `results/README.md`; the raw datasets are **not** committed.

## Compute

The per-run JSONs record wall-clock time for each 300k-step PPO run. Across the
20 published runs: **256–571 s per run, mean 471 s**, with one CPU thread per run
(`torch_threads: 1`). The batch runner uses one process per run, so the full
20-run grid is roughly 2.6 CPU-hours of work before parallelism. No GPU was used.
The figures and tables were regenerated from the committed JSON/CSV artefacts.

## Known gaps / next steps

- No evaluation on the closed Test split; the reported table is Val only.
- Raw data is downloaded at run time and is not vendored. Checksums for the
  expected files live in `data/checksums.json`. The Open-Meteo response includes
  a per-request `generationtime_ms`; `download.py` strips it before hashing, so
  the `openmeteo` checksum was re-recorded against the current archive snapshot.
  Re-downloads of that live archive can still differ if upstream revises it.
- The README figures are committed PNGs; `tools/report/plots.py` rebuilds them
  from `results/`.
- Data download, the full test suite, and a short training run were re-verified
  with the CPU-only torch wheel; the 300k-step grid was not re-run. See D1.
