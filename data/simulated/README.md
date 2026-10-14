# Simulated data

Each scenario's replicate datasets live in their own subfolder:

- `scenario_A/rep_0.csv` ... `rep_9.csv` — 10 independent simulated epidemics for Scenario A
- `scenario_B/rep_0.csv` ... `rep_9.csv` — Scenario B
- `scenario_C/rep_0.csv` ... `rep_9.csv` — Scenario C

These are generated automatically the first time you run the corresponding
script in `simulation_study/` (e.g. `python -m simulation_study.scenario_A`).
If the CSVs already exist, the script loads them instead of re-simulating,
so re-running an analysis (e.g. to tweak a plot) doesn't require regenerating
the data. Delete a scenario's folder (or an individual `rep_N.csv`) to force
regeneration with a fresh random draw.
