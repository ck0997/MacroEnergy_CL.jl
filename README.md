# MacroEnergy_CL.jl
# MacroEnergy_CL.jl

`MacroEnergy_CL.jl` is a China-focused low-carbon energy and industrial systems modeling repository built on top of [`MacroEnergy.jl`](https://github.com/macroenergy/MacroEnergy.jl). It contains MacroEnergy model extensions, case data, and analysis workflows for studying electricity, steel, cement, aluminum, CO2 transport, and CO2 storage options across Chinese provinces.

The main case in this repository is a 31-province, one-period model with 24 representative hours and high renewable availability assumptions:

```text
31_provinces_1_period_updatedelec_steel_cement_nonprovincial_aluminum_max_renewables_24hours/
```

## Repository Layout

```text
.
├── src/                 # MacroEnergy package source and model components
├── test/                # Unit and balance tests
├── docs/                # Upstream MacroEnergy documentation source
├── electricity_3zone/   # Small electricity-only example case
├── improved_co2_pipelines/
│   └── chinny_co2_pipeline_distance/
│       ├── candidate_pipelines_N2_K1.csv
│       ├── all_routes_N2.csv
│       ├── routes_export.csv
│       ├── basin_centroids.csv
│       └── methodology_anna2.ipynb
└── 31_provinces_1_period_updatedelec_steel_cement_nonprovincial_aluminum_max_renewables_24hours/
    ├── run.jl
    ├── run_cases.jl
    ├── settings/
    ├── system/
    ├── user_additions/
    ├── plot_inputs/
    └── plot_results*.ipynb
```

## Requirements

- Julia 1.9 or newer
- A Julia optimizer supported by MacroEnergy
- HiGHS for the small `electricity_3zone` example
- Gurobi for the large 31-province case scripts as currently written

The package dependencies are declared in `Project.toml` and pinned in `Manifest.toml`.

## Setup

Clone the repository and instantiate the Julia environment:

```bash
git clone https://github.com/ck0997/MacroEnergy_CL.jl.git
cd MacroEnergy_CL.jl
julia --project=.
```

Inside Julia:

```julia
using Pkg
Pkg.instantiate()
```

If you are using Gurobi, make sure your Gurobi installation and license are available before running the 31-province case.

## Running Examples

### Small Electricity Example

The `electricity_3zone` case is a compact electricity-only example that can be run with HiGHS:

```bash
julia --project=. electricity_3zone/run.jl
```

### 31-Province China Case

The main China case is located at:

```text
31_provinces_1_period_updatedelec_steel_cement_nonprovincial_aluminum_max_renewables_24hours/
```

Run the base case with:

```bash
julia --project=. 31_provinces_1_period_updatedelec_steel_cement_nonprovincial_aluminum_max_renewables_24hours/run.jl
```

Run the scenario sweep with:

```bash
julia --project=. 31_provinces_1_period_updatedelec_steel_cement_nonprovincial_aluminum_max_renewables_24hours/run_cases.jl
```

`run_cases.jl` sweeps across CO2 injection assumptions using:

- `system/nodes_min_co2_injection.json`
- `system/nodes_mean_co2_injection.json`
- `system/nodes_max_co2_injection.json`

It also runs emissions-cap cases based on fractions of the uncapped baseline.

## CO2 Pipeline Data

The `improved_co2_pipelines/chinny_co2_pipeline_distance/` directory contains candidate CO2 pipeline routes, route exports, basin centroids, provincial capital data, and notebooks used to develop the pipeline distance inputs. These files support the CO2 transport and storage representation used by the China case.

## Plotting and Outputs

The main case directory includes Python scripts and notebooks for inspecting inputs and plotting results:

- `extract_and_plot_inputs.py`
- `electricity_plotting.py`
- `plot_co2_capture_transport_storage.py`
- `plot_results.ipynb`
- `plot_results_min.ipynb`
- `plot_results_mean.ipynb`
- `plot_results_max.ipynb`

Scenario outputs are written under `results_*` folders or the `results/` directory, depending on the run script.

## Development Notes

This repository is a research fork of MacroEnergy. New assets and model behavior should be added under `src/`, with corresponding tests under `test/` when possible. For package development, activate the repository environment:

```bash
julia --project=.
```

Then run tests with:

```julia
using Pkg
Pkg.test()
```

## Citation

This work builds on MacroEnergy.jl. If you use this repository, cite this repository as appropriate and also cite the upstream MacroEnergy.jl software paper:

```bibtex
@article{macdonald2025macroenergy,
  title={MacroEnergy. jl: A large-scale multi-sector energy system framework},
  author={Macdonald, Ruaridh and Pecci, Filippo and Bonaldo, Luca and Law, Jun Wen and Weng, Yu and Mallapragada, Dharik and Jenkins, Jesse},
  journal={arXiv preprint arXiv:2510.21943},
  year={2025}
}
```

## License

See `LICENSE` for license information.
