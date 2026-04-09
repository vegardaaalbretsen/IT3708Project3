# IT3708Project3

Minimal project for parsing feature-selection HDF5 data and exporting CSV metrics.

Structure:

- `src/datasets.jl`: dataset paths and output locations
- `src/parser.jl`: parse one HDF5 dataset and write CSV
- `src/landscape.jl`: load CSV lookup tables and apply penalty
- `main.jl`: project entrypoint

Run:

```bash
julia --project=. main.jl credit-a
```
