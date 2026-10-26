# Can reinforcement learning stay under a grid-import cap and still keep the bill low?

A simulated study on real public data (France, 2017–2019).

---

## The problem

Picture a small microgrid: a **battery**, a **solar array**, and **five zones of electrical load**, all connected to the main grid.

That grid connection has a **368 kW import target**. The microgrid should draw no more than that from the grid in any hour: exceeding it is charged as a constraint violation, and the controller is judged on how rarely and how little it overruns.

Every hour, a controller has to decide:

- how hard to **charge or discharge the battery**, and
- how much load to **shed** (switch off) in each zone,

so that the total cost stays as low as possible:

```
cost = electricity bought + penalty for unserved energy + battery wear
```

Shedding load keeps the site under the cap, but it is penalised (1 EUR per kWh not served). Using the battery helps, but it wears out. Doing nothing is cheapest on paper, but it breaks the cap. The controller has to balance all three.

## The question

Most RL-for-energy papers ask: *"can RL beat a hand-written rule?"*

This project asks something harder:

> If the controller **has to** respect a constraint on grid import, and the rule it is compared against is **well tuned**, can RL find a policy that is both **feasible** and **cheaper**?

## Where this comes from

This repo extends two papers by Salaorni et al. (Politecnico di Milano):

- **ErNESTO-DT**, a battery simulator (*Journal of Energy Storage*, 2025, [doi:10.1016/j.est.2025.117745](https://doi.org/10.1016/j.est.2025.117745)).
- **RL for optimal control in microgrids**, PPO on top of that simulator (*IJCNN 2025*, [arXiv:2506.22995](https://arxiv.org/abs/2506.22995)).

That second paper shows RL beating simple rule-based strategies, but it has **no grid-import cap of any kind** and compares against **untuned** rules. This repo changes exactly those two things:

| | IJCNN 2025 paper | this repo |
|---|---|---|
| Constraint | soft penalty on battery limits | **grid-import cap as a learned penalty**, hard battery limits |
| Rule-based comparator | simple rules | **tuned** peak-shaving heuristics |
| Objective | maximise profit | minimise cost |
| Load shedding | not modelled | in the objective, per zone |
| Battery model | SoC, temperature, ageing | SoC only, linear wear (simpler) |

## Approaches compared

Three ways of making RL respect the cap, against two reference controllers:

| Method | Idea |
|---|---|
| **PPO** | plain PPO, no special treatment of the cap |
| **PPO-Lagrangian** | add a learned penalty multiplier on cap violations |
| **PPO-Residual** | learn a small correction on top of the tuned heuristic |
| **Shielded PPO** | a wrapper clips any action that would break the cap |
| *Grid-Only* | do nothing, just buy from the grid (reference) |
| *PSM'* | tuned peak-shaving rule, the strong comparator |

## The environment

`env/smartgrid_env.py` is a [Gymnasium](https://gymnasium.farama.org/) environment with hourly steps and weekly episodes.

- **Observation (17 values):** battery charge, solar power, the five zone loads, price, hour and weekday (sin/cos), weather, and event flags.
- **Action (6 values):** one battery command in [-1, 1] and five shed fractions in [0, 1].
- **Battery:** 500 kWh, 250 kW, charge kept between 10 % and 90 %.
- **Solar:** about 250 kW peak.
- **Cap:** 368.15 kW grid import, checked every hour. It is the 90th percentile of
  train-split grid-only import (`max(0, load − PV)` with the battery idle), derived by
  `data/prepare.py` and stored in `data/processed/split_report.json`. It is a
  peak-reduction **target**, not a hardware rating: 10.0 % of train hours and 10.7 %
  of val hours exceed it before any control, and 6 of the 104 train weeks need more
  peak-shaving energy than the battery can deliver at one full cycle per day. Runs
  record the resolved value under `resolved_config.grid_import_cap_kw`.

`env/cap_shield.py` is the shield wrapper. It clips the battery command so the cap cannot be exceeded and turns any remaining violation into shed load.

## The data

Datasets are fetched at run time and are **not** committed.

| Data | Source |
|---|---|
| Hourly load and price | Open Power System Data (French load, German day-ahead price used as a proxy) |
| Solar irradiance | PVGIS (EU Joint Research Centre) |
| Temperature, humidity, pressure | Open-Meteo |

The zone split of the load and the stadium-event calendar are **synthetic**. Data is split by whole weeks into Train (2017–2018), Val (2019 H1) and Test (2019 H2). The Test split is kept closed and is not used by any code here.

## Project layout

```
configs/config.yaml      all experiment settings
data/                    download and preparation scripts
env/                     environment, cap shield, observation normaliser
agents/                  PPO, heuristics, reward shaping
experiments/             training and batch runners
tools/                   table, figure and status scripts
tests/                   unit tests
results/                 run outputs, tables, figures (see results/README.md)
```

## Reproduction

From a clean checkout:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/download.py
python data/prepare.py
python -m pytest tests/ -v

python experiments/train.py --config s3_ppo --seed 101
python tools/report/export_csv.py
python tools/report/plots.py
```

`--config` takes an experiment name from `configs/config.yaml` (for example
`s3_ppo`, `s3_ppo_lag`, `s3_ppo_res`, `s3_sh_ppo`). Use `--overwrite` to replace an
existing result. A full 300k-step run writes
`results/runs/<agent>/<run-id>_seed<seed>.json` and a checkpoint under
`checkpoints/`.

### Tests

`tests/` covers the environment (physics, cap, SoC bounds, determinism), the PPO
agent (GAE, truncation bootstrapping, seed determinism, λ update), the config
loader, and the reward-shaping helper. Tests marked `needs_real_data` skip until
`data/processed/*.parquet` exists.

### Compute

Everything runs CPU-only, one thread per run (`torch_threads: 1`). The 20 published
300k-step runs took **256–571 s each (mean 471 s)**, about **2.6 CPU-hours** for the
full grid before parallelism; six workers finish it in roughly half an hour.

## Limitations

- The setup is a simulation: the topology, zone split, events, battery and PV sizes, and the price proxy are assumptions.
- The comparison with the IJCNN paper is analytical, not experimental: different data, objective and battery model.
- The reported table is Val only. The Test split (2019 H2) is closed and was not evaluated.

---

## Results

Evaluated on the Val split, 300k training steps, 5 seeds per method.

<table> <tr> <td align="center" valign="top" width="50%"> <img src="results/figures/learning_curves.png" alt="Learning curves" width="360"><br> <sub><b>Learning curves</b></sub> </td> <td align="center" valign="top" width="50%"> <img src="results/figures/cost_vs_exceedance.png" alt="Cost vs exceedance" width="360"><br> <sub><b>Cost vs exceedance</b></sub> </td> </tr> <tr> <td align="center" valign="top" width="50%"> <img src="results/figures/per_seed_results.png" alt="Per-seed results" width="360"><br> <sub><b>Per-seed results</b></sub> </td> <td align="center" valign="top" width="50%"> <img src="results/figures/cost_breakdown.png" alt="Cost breakdown" width="360"><br> <sub><b>Cost breakdown</b></sub> </td> </tr> </table>

All four figures and the per-seed JSONs behind them are committed under
[`results/`](results/README.md). That folder also carries a λ-trajectory
diagnostic (`lambda_trajectory.png`) regenerated from the same runs.

| Method | Val cost (EUR) | Within the cap? |
|---|---:|---:|
| PPO | 45,513 | no |
| PPO-Lagrangian | 45,539 | no |
| PPO-Residual | 54,241 | no (close) |
| Shielded PPO | 58,937 | **yes** |
| *Grid-Only* | *45,338* | *no* |
| *PSM' (tuned rule)* | *55,532* | *yes* |

"Within the cap" means mean exceedance at most 0.05 kW per hourly step.

**Takeaways**

- **No learned policy beats the tuned rule while staying within the cap.** The cheap ones (PPO, PPO-Lagrangian) are cheap because they ignore the cap and behave like Grid-Only.
- **The residual comes closest** (about 2 % cheaper than PSM'), but it still breaks the cap slightly on every seed.
- **The shield is the only method that is always within the cap, but it works by construction**: it clips the actions, so the policy is effectively removed from the loop. Passing a feasibility test that way says nothing about learning.

So the gap to the original paper comes mostly from the **strength of the comparator** and the **presence of a grid-import constraint**, not from RL being unable to learn.

## Citations

```bibtex
@article{salaorni2025ernesto,
  title   = {A novel digital twin for battery energy storage systems in micro-grids},
  author  = {Salaorni, Davide and Bianchi, Federico and Colnago, Silvia
             and Barisione, Andrea and Trov{\`o}, Francesco and Restelli, Marcello},
  journal = {Journal of Energy Storage},
  volume  = {132},
  pages   = {117745},
  year    = {2025},
  doi     = {10.1016/j.est.2025.117745}
}

@inproceedings{salaorni2025reinforcement,
  title     = {A Reinforcement Learning Approach for Optimal Control in Microgrids},
  author    = {Salaorni, Davide and Bianchi, Federico and Trov{\`o}, Francesco
               and Restelli, Marcello},
  booktitle = {2025 International Joint Conference on Neural Networks (IJCNN)},
  year      = {2025},
  eprint    = {2506.22995},
  archivePrefix = {arXiv},
  doi       = {10.1109/IJCNN64981.2025.11227643}
}
```

Data: Open Power System Data (CC BY 4.0), PVGIS / EU Joint Research Centre (EC reuse policy), Open-Meteo (CC BY 4.0).
