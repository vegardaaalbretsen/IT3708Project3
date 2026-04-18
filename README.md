# IT3708Project3

Small Julia project for parsing feature-selection landscapes, generating a synthetic triangle landscape, and visualizing landscape structure.

## Structure

- `src/datasets.jl`: dataset paths and default output paths
- `src/types.jl`: shared `Landscape` and `HBMNode` types
- `src/parser.jl`: parse HDF5 datasets and write CSV lookup tables
- `src/landscape.jl`: load CSV lookup tables and evaluate fitness/penalties
- `src/ea.jl`: standard `(1+1)` EA with `1/n` bit-flip mutation
- `src/triangle.jl`: triangle fitness function and synthetic landscape generator
- `src/hbm.jl`: HBM mapping, one-flip neighbors, and optima detection
- `src/visualization.jl`: HBM and feature-count plotting

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

Run the standard EA on a landscape:

```bash
julia --project=. run_ea.jl breast-w
julia --project=. run_ea.jl breast-w 10000 0.01
julia --project=. run_ea.jl triangle 5000 0.0 42 0
```

Run tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
