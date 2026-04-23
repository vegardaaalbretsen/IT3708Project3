# IT3708Project3

Small Julia project for parsing feature-selection landscapes, generating a synthetic triangle landscape, and visualizing landscape structure.

For threaded evaluation in the GA, NSGA-II, and swarm runners, start Julia with multiple threads, for example `julia --threads auto --project=.`.

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
- `run_experiments.jl`: batch experiment runner and summary CSV generator
- `plot_experiment_fitness.jl`: plot experiment fitness/diversity curves from `generation_stats.csv`
- `plot_population_snapshots.jl`: plot selected population snapshots from experiment runs
- `benchmark_threaded_eval.jl`: serial vs threaded evaluation benchmark for GA, NSGA-II, and swarm

## Commands

Available landscape keys:

- training datasets: `breast-w`, `credit-a`, `letter-r`
- Step 6 test datasets: `zoo`, `hepatitis`
- synthetic dataset: `triangle`

Parse a real HDF5 dataset to CSV:

```bash
julia --project=. parse_landscape.jl breast-w
julia --project=. parse_landscape.jl zoo
julia --project=. parse_landscape.jl hepatitis
```

Create an HBM plot for a real dataset:

```bash
julia --project=. plot_hbm.jl breast-w
julia --project=. plot_hbm.jl breast-w 0.01
julia --project=. plot_hbm.jl zoo 0.0
```

Create an HBM plot for the synthetic triangle landscape:

```bash
julia --project=. plot_hbm.jl triangle
```

Create a fitness-by-feature-count plot:

```bash
julia --project=. plot_feature_count.jl breast-w
julia --project=. plot_feature_count.jl breast-w 0.01
julia --project=. plot_feature_count.jl hepatitis 0.0
julia --project=. plot_feature_count.jl triangle
```

Use the Step 6 test datasets manually with `epsilon=0.0`; they are loadable through the same dataset-key flow but are intentionally not part of the default `run_experiments.jl` training suite.

Run batch experiments for the report:

```bash
julia --threads auto --project=. run_experiments.jl
julia --threads auto --project=. run_experiments.jl --seeds 10 --epsilon 0.01
julia --threads auto --project=. run_experiments.jl --datasets breast-w,triangle --algorithms ga,swarm
```

`run_experiments.jl` writes four CSV files under `exports/csv/experiments/`:

- `raw_runs.csv`: one row per algorithm, landscape, seed, and epsilon, including how many unique global optima were seen during the run
- `generation_stats.csv`: min, average, max, best-so-far fitness, normalized diversity entropy, and cumulative global-optima coverage per generation
- `population_snapshots.csv`: selected population snapshots grouped by unique bitstring and duplicate count
- `summary.csv`: average and standard deviation of best fitness across runs, plus aggregated global-optima coverage

`global_optima_seen` is cumulative: it counts the number of distinct global optima that appeared anywhere in the population/front up to that generation. `global_optima_fraction` is that count divided by the total number of global optima in the landscape for the chosen `epsilon`.

The default batch experiment budget is 100 iterations/generations for each algorithm. This keeps the comparison focused on convergence behavior instead of only showing that all algorithms eventually find the same solution.

Create plots from the experiment generation statistics:

```bash
julia --project=. plot_experiment_fitness.jl
julia --project=. plot_experiment_fitness.jl --metric mean_fitness
julia --project=. plot_experiment_fitness.jl --metric diversity_entropy
julia --project=. plot_experiment_fitness.jl --metric global_optima_seen
julia --project=. plot_experiment_fitness.jl --metric global_optima_fraction
```

By default, `plot_experiment_fitness.jl` plots `best_so_far_fitness`. To plot diversity/entropy, explicitly pass `--metric diversity_entropy`.

The plotting script writes one PNG per landscape and epsilon under `exports/plots/experiments/`. Useful metrics are:

- `best_so_far_fitness`: main convergence plot
- `mean_fitness`: average population quality over time
- `diversity_entropy`: normalized population diversity over time, from `0.0` to `1.0`
- `global_optima_seen`: cumulative number of unique global optima observed
- `global_optima_fraction`: cumulative fraction of the landscape's global optima that were observed

If you want diversity plots, rerun `run_experiments.jl` first so `generation_stats.csv` includes the `diversity_entropy` column.

Create HBM and feature-count population snapshot plots for one run:

```bash
julia --project=. plot_population_snapshots.jl --landscape breast-w --algorithm ga --seed 1
julia --project=. plot_population_snapshots.jl --landscape breast-w --algorithm nsga2 --seed 1
julia --project=. plot_population_snapshots.jl --landscape breast-w --algorithm swarm --seed 1
```

The snapshot plotting script reads `exports/csv/experiments/population_snapshots.csv` and writes PNGs under `exports/plots/experiments/snapshots/<landscape>/<algorithm>/<plot_type>/`. It plots the selected snapshot generations from each run: start, two intermediate points, the first generation where peak best-so-far fitness was reached, and the final generation. Duplicate snapshot generations are skipped. Use `--plot feature-count` or `--plot hbm` to create only one plot type.

Run the single-objective GA on a landscape:

```bash
julia --project=. run_ea.jl breast-w
julia --project=. run_ea.jl breast-w 10000 0.01
julia --project=. run_ea.jl zoo 500 0.0 --seed 42
julia --project=. run_ea.jl triangle 5000 0.0 42 0
julia --project=. run_ea.jl breast-w 10000 0.01 --plot trace --seed 42
julia --project=. run_ea.jl breast-w 10000 0.01 --plot feature-count --seed 42
julia --project=. run_ea.jl triangle 5000 0.01 --plot both --seed 42 --initial-index 0
julia --project=. run_ea.jl breast-w 500 0.01 --popsize 150 --pc 0.9 --pm 0.02 --tournament-size 5 --survivor-mode generational --elite 2
```

To enable threaded evaluation, run for example:

```bash
julia --threads auto --project=. run_ea.jl breast-w 500 0.01
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
julia --project=. run_nsga2.jl hepatitis 500 0.0 --seed 42
julia --project=. run_nsga2.jl triangle 500 0.0 42 0
julia --project=. run_nsga2.jl breast-w 500 0.01 --popsize 150 --pc 0.9 --pm 0.02
julia --project=. run_nsga2.jl breast-w 500 0.01 --plot front
julia --project=. run_nsga2.jl breast-w 500 0.01 --plot stn
julia --project=. run_nsga2.jl breast-w 500 0.01 --plot both
julia --project=. run_nsga2.jl breast-w 500 0.01 --plot all
```

To enable threaded evaluation, run for example:

```bash
julia --threads auto --project=. run_nsga2.jl breast-w 500 0.01 --plot all
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

NSGA-II uses three objectives during selection:

- maximize accuracy
- minimize number of selected features
- minimize evaluation time

`epsilon` is not used for Pareto selection. It is only used when reporting `pareto_penalized_fitness` and the single `best_penalized_*` summary fields.

NSGA-II visualizations:

- `plot_nsga2_pareto_front` and `save_nsga2_pareto_front_plot`
- `plot_nsga2_trace` and `save_nsga2_trace_plot`
- `plot_nsga2_search_trajectory_network` and `save_nsga2_search_trajectory_network_plot`

The Pareto-front plot shows all subsets in accuracy-vs-feature-count space and highlights the final non-dominated set.

The trace plot summarizes the search over time using:

- maximum accuracy on the current front
- minimum selected features on the current front
- minimum evaluation time on the current front
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
result.pareto_time
result.best_penalized_index

save_nsga2_pareto_front_plot(landscape, result, "exports/plots/ea/breast-w_nsga2_pareto.png")
save_nsga2_trace_plot(result, "exports/plots/ea/breast-w_nsga2_trace.png")
save_nsga2_search_trajectory_network_plot(landscape, result, "exports/plots/stn/breast-w_nsga2_stn.png")
```

The default NSGA-II Pareto CSV includes `index`, `accuracy`, `num_selected`, `time`, and `penalized_fitness`.

## Threading Benchmark

Use `benchmark_threaded_eval.jl` to compare serial and threaded evaluation on this machine.

```bash
julia --threads auto --project=. benchmark_threaded_eval.jl all 4000 3
```

Arguments:

- mode: `ga|nsga2|swarm|all`
- work factor: heavier synthetic objective workload per evaluation
- repeats: number of timed runs per mode

Run the swarm EA on a landscape:

```bash
julia --project=. run_swarm.jl breast-w
julia --project=. run_swarm.jl breast-w 500 0.01
julia --project=. run_swarm.jl zoo 300 0.0 --seed 42
julia --project=. run_swarm.jl triangle 300 0.0 --seed 42
julia --project=. run_swarm.jl breast-w 500 0.01 --swarm-size 50 --w 0.7 --c1 1.4 --c2 1.4
julia --project=. run_swarm.jl breast-w 500 0.01 --plot feature-count --seed 42
julia --project=. run_swarm.jl breast-w 500 0.01 --plot trace --seed 42
julia --project=. run_swarm.jl triangle 300 0.0 --plot hbm --seed 42
julia --project=. run_swarm.jl triangle 300 0.0 --plot all --seed 42
```

To enable threaded particle evaluation, run for example:

```bash
julia --threads auto --project=. run_swarm.jl breast-w 500 0.01 --plot all
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

For presentation-friendly population snapshot GIFs with every generation and a short pause on the final frame, use:

```bash
julia --project=. animate_population_snapshots.jl breast-w 150 0.01 --algorithm ga --seed 1 --plot feature-count --fps 8 --final-hold 12
julia --project=. animate_population_snapshots.jl breast-w 150 0.01 --algorithm nsga2 --seed 1 --plot feature-count --fps 8 --final-hold 12
julia --project=. animate_population_snapshots.jl breast-w 150 0.01 --algorithm swarm --seed 1 --plot feature-count --fps 8 --final-hold 12
```

The `--final-hold` value is the number of extra copies of the last frame. At `--fps 8`, `--final-hold 12` gives roughly a 1.5 second pause at the end of the GIF. This workflow animates the grouped population snapshot for every generation, so it is usually the clearest option for presentations.

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
