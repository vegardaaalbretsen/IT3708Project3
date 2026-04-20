# IT3708Project3

Small Julia project for parsing feature-selection landscapes, generating a synthetic triangle landscape, and visualizing landscape structure.

## Structure

- `src/datasets.jl`: dataset paths and default output paths
- `src/types.jl`: shared `Landscape` and `HBMNode` types
- `src/parser.jl`: parse HDF5 datasets and write CSV lookup tables
- `src/landscape.jl`: load CSV lookup tables and evaluate fitness/penalties
- `src/general_ga.jl`: reusable single-objective GA core
- `src/nsga2_core.jl`: NSGA-II non-domination, crowding, and environmental selection
- `src/feature_main.jl`: adapts the GA core and NSGA-II core to the subset-index landscape
- `src/swarm_evolp.jl`: EvoLP-based PSO adapter for subset selection
- `src/triangle.jl`: triangle fitness function and synthetic landscape generator
- `src/hbm.jl`: HBM mapping, one-flip neighbors, and optima detection
- `src/visualization.jl`: HBM, feature-count, EA, and swarm plotting/animation

## Commands

Available landscape keys:

- real datasets: `breast-w`, `credit-a`, `letter-r`
- synthetic dataset: `triangle`

Parse a real HDF5 dataset to CSV:

```bash
julia --project=. parse_landscape.jl breast-w
```

Create an HBM plot for a real dataset:

```bash
julia --project=. plot_hbm.jl breast-w
julia --project=. plot_hbm.jl breast-w 0.01
```

Create an HBM plot for the synthetic triangle landscape:

```bash
julia --project=. plot_hbm.jl triangle
```

Create a fitness-by-feature-count plot:

```bash
julia --project=. plot_feature_count.jl breast-w
julia --project=. plot_feature_count.jl breast-w 0.01
julia --project=. plot_feature_count.jl triangle
```

Run the single-objective GA on a landscape:

```bash
julia --project=. run_ea.jl breast-w
julia --project=. run_ea.jl breast-w 10000 0.01
julia --project=. run_ea.jl triangle 5000 0.0 42 0
julia --project=. run_ea.jl breast-w 10000 0.01 --plot trace --seed 42
julia --project=. run_ea.jl breast-w 10000 0.01 --plot feature-count --seed 42
julia --project=. run_ea.jl triangle 5000 0.1 --plot both --seed 42 --initial-index 0
julia --project=. run_ea.jl breast-w 500 0.01 --popsize 150 --pc 0.9 --pm 0.02 --tournament-size 5 --survivor-mode generational --elite 2
```

`run_ea.jl` supports:

- `--plot none|trace|feature-count|both`
- `--popsize N`
- `--pc V`
- `--pm V`
- `--tournament-size N`
- `--survivor-mode elitist|generational`
- `--elite N`

The GA runner uses `run_single_objective_ea(...)`, implemented in `src/feature_main.jl` on top of the reusable core in `src/general_ga.jl`. Plotting automatically enables history collection for that run.

If you call the API directly, `keep_history=false` avoids storing per-generation traces, while `keep_history=true` enables trace and feature-count path plots.

Example:

```julia
using IT3708Project3
using Random

landscape = load_landscape_key("breast-w")
result = run_single_objective_ea(
    landscape;
    iterations=500,
    epsilon=0.01,
    population_size=150,
    crossover_probability=0.9,
    mutation_probability=0.02,
    tournament_size=5,
    survivor_mode=:generational,
    elite=2,
    rng=MersenneTwister(42),
    keep_history=true,
)

save_ea_trace_plot(result, "exports/plots/ea/breast-w_ga_trace.png")
save_fitness_by_feature_count_with_ea_plot(landscape, result, "exports/plots/ea/breast-w_ga_feature_count.png")
```

Run NSGA-II on a landscape:

```bash
julia --project=. run_nsga2.jl breast-w
julia --project=. run_nsga2.jl breast-w 1000 0.01
julia --project=. run_nsga2.jl triangle 500 0.0 42 0
julia --project=. run_nsga2.jl breast-w 500 0.01 --popsize 150 --pc 0.9 --pm 0.02
julia --project=. run_nsga2.jl breast-w 500 0.01 --plot front
julia --project=. run_nsga2.jl breast-w 500 0.01 --plot stn
julia --project=. run_nsga2.jl breast-w 500 0.01 --plot both
julia --project=. run_nsga2.jl breast-w 500 0.01 --plot all
```

`run_nsga2.jl` supports:

- `--seed N`
- `--initial-index I`
- `--popsize N`
- `--pc V`
- `--pm V`
- `--log-every N`
- `--plot none|front|trace|stn|both|all`
- `--output path`
- `--plot-output path`

By default, the Pareto-front CSV written by `run_nsga2.jl` is stored under `exports/csv/results/` so NSGA-II outputs stay separate from the parsed landscape CSV files.

The NSGA-II runner uses `run_nsga2_feature_ea(...)`, implemented in `src/feature_main.jl` on top of the non-dominated sorting and crowding operators in `src/nsga2_core.jl`.

NSGA-II uses two objectives during selection:

- maximize accuracy
- minimize number of selected features

`epsilon` is not used for Pareto selection. It is only used when reporting `pareto_penalized_fitness` and the single `best_penalized_*` summary fields.

NSGA-II visualizations:

- `plot_nsga2_pareto_front` and `save_nsga2_pareto_front_plot`
- `plot_nsga2_trace` and `save_nsga2_trace_plot`
- `plot_nsga2_search_trajectory_network` and `save_nsga2_search_trajectory_network_plot`

The Pareto-front plot shows all subsets in accuracy-vs-feature-count space and highlights the final non-dominated set.

The trace plot summarizes the search over time using:

- maximum accuracy on the current front
- minimum selected features on the current front
- number of unique Pareto points

The search trajectory network (STN) projects visited subsets into HBM space and overlays:

- parent-to-child transitions aggregated across generations
- visit frequency per subset
- initial population
- final population
- final Pareto front
- best penalized summary point

The trace plot and STN require `keep_history=true`, which `run_nsga2.jl` enables automatically for `--plot trace`, `--plot stn`, `--plot both`, and `--plot all`.

Example:

```julia
using IT3708Project3
using Random

landscape = load_landscape_key("breast-w")
result = run_nsga2_feature_ea(
    landscape;
    iterations=500,
    epsilon=0.01,
    population_size=150,
    crossover_probability=0.9,
    mutation_probability=0.02,
    rng=MersenneTwister(42),
    keep_history=true,
)

result.pareto_indices
result.pareto_accuracy
result.pareto_num_selected
result.best_penalized_index

save_nsga2_pareto_front_plot(landscape, result, "exports/plots/ea/breast-w_nsga2_pareto.png")
save_nsga2_trace_plot(result, "exports/plots/ea/breast-w_nsga2_trace.png")
save_nsga2_search_trajectory_network_plot(landscape, result, "exports/plots/stn/breast-w_nsga2_stn.png")
```

Run the swarm EA on a landscape:

```bash
julia --project=. run_swarm.jl breast-w
julia --project=. run_swarm.jl breast-w 500 0.01
julia --project=. run_swarm.jl triangle 300 0.0 --seed 42
julia --project=. run_swarm.jl breast-w 500 0.01 --swarm-size 50 --w 0.7 --c1 1.4 --c2 1.4
julia --project=. run_swarm.jl breast-w 500 0.01 --plot feature-count --seed 42
julia --project=. run_swarm.jl breast-w 500 0.01 --plot trace --seed 42
julia --project=. run_swarm.jl triangle 300 0.0 --plot hbm --seed 42
julia --project=. run_swarm.jl triangle 300 0.0 --plot all --seed 42
```

`run_swarm.jl` supports `--plot none|trace|feature-count|hbm|all`. The trace plot automatically enables swarm history collection.

## Swarm Visualizations

The swarm implementation treats each particle as a continuous vector in `[0,1]^n` and decodes it into a subset bitmask using `x[i] >= 0.5`.

Available swarm visualization helpers:

- `plot_fitness_by_feature_count_with_swarm` and `save_fitness_by_feature_count_with_swarm_plot`
- `plot_hbm_with_swarm` and `save_hbm_with_swarm_plot`
- `plot_swarm_trace` and `save_swarm_trace_plot`
- `save_fitness_by_feature_count_swarm_animation`
- `save_hbm_swarm_animation`

Static swarm overlays work with any `run_swarm_ea(...)` result. If history is available, the feature-count and HBM overlays also show the best-so-far path from the first generation to the current one. Swarm trace plots and animations require `keep_history=true`.

Example:

```julia
using IT3708Project3
using Random

landscape = load_landscape_key("breast-w")
result = run_swarm_ea(
    landscape;
    iterations=200,
    epsilon=0.01,
    swarm_size=40,
    rng=MersenneTwister(42),
    keep_history=true,
)

save_fitness_by_feature_count_with_swarm_plot(landscape, result, "exports/plots/ea/breast-w_swarm_feature_count.png")
save_hbm_with_swarm_plot(landscape, result, "exports/plots/hbm/breast-w_swarm_hbm.png")
save_swarm_trace_plot(landscape, result, "exports/plots/ea/breast-w_swarm_trace.png")
save_fitness_by_feature_count_swarm_animation(landscape, result, "exports/plots/ea/breast-w_swarm_feature_count.gif")
save_hbm_swarm_animation(landscape, result, "exports/plots/hbm/breast-w_swarm_hbm.gif")
```

The swarm trace plot includes:

- best fitness per iteration
- mean fitness per iteration
- median fitness per iteration
- number of unique decoded subsets
- mean pairwise Hamming distance
- fraction of particles equal to the global best

Run tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
