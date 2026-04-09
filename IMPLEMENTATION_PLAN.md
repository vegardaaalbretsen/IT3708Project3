# Implementation Plan

## Goal

Add three optimization methods to the current Julia project:

1. A single-objective evolutionary algorithm
2. A multi-objective evolutionary algorithm
3. A swarm intelligence algorithm

The project already has the most important prerequisite: exact lookup-table fitness evaluation for both real feature-selection landscapes and the synthetic triangle landscape.

## Recommended Choices

### 1. Single-objective EA
Use a **binary Genetic Algorithm (GA)**.

Why:
- Standard and easy to explain in a report
- Natural for bitstring / subset search
- Reusable operators for the multi-objective algorithm
- Fits the current integer bitmask representation well

### 2. Multi-objective EA
Use **NSGA-II**.

Why:
- Classic and widely accepted
- Good match for feature selection
- Easy to define meaningful objectives:
  - maximize accuracy
  - minimize number of selected features

### 3. Swarm Intelligence
Use **PSO with `EvoLP.jl` support**.

Why:
- Standard swarm method that can be adapted to bitstring search spaces
- `EvoLP.jl` already includes PSO building blocks and minimisers
- reduces implementation risk for the swarm part
- still gives a clear contrast to GA / NSGA-II

Recommended approach:
- keep **GA** and **NSGA-II** custom and minimal
- use **`EvoLP.jl`** for the swarm algorithm, ideally as a thin wrapper around PSO
- if `EvoLP.jl` does not fit the binary representation cleanly, fall back to a custom Binary PSO only for that part

## Core Design Principle

Keep all candidate solutions as **integer subset indices** internally, not `Vector{Bool}`.

Reason:
- The current code already stores landscapes as lookup tables indexed by subset ID
- Fitness lookup is then O(1)
- Mutation and bit flips are cheap with bit operations
- Decoding to features is only needed for reporting

## Existing Code To Reuse

Current project already provides:

- real-data lookup table loading
- synthetic triangle landscape generation
- bitstring helpers
- HBM and LON visualization helpers
- exact best-subset summaries

This means the new work should **not** reimplement evaluation logic.
Instead, add a thin optimization layer on top of the current landscape API.

For implementation style:
- keep the evolutionary algorithms mostly custom for transparency and reportability
- allow the swarm algorithm to reuse `EvoLP.jl`

## Objective Definitions

### Real feature-selection landscapes

#### Single-objective default
Maximize:

- `penalized_table`

This matches the current regularized feature-selection formulation.

#### Multi-objective default
Optimize:

1. maximize raw accuracy
2. minimize number of selected features

Optional alternative:
1. maximize raw accuracy
2. minimize mean training time

Recommended default for the report:
- use `raw accuracy` vs `number of features`
- it is easier to explain than using time

### Synthetic triangle landscape

#### Single-objective
Maximize:
- `triangle_fitness`

#### Multi-objective
Use:
1. maximize `triangle_fitness`
2. minimize Hamming weight

#### Swarm
Use the same scalar objective as the single-objective case.

## Architecture Plan

Do not keep expanding `src/IT3708Project3.jl` into one giant file.

Refactor toward:

- `src/IT3708Project3.jl`
- `src/landscapes.jl`
- `src/objectives.jl`
- `src/algorithms/ga.jl`
- `src/algorithms/nsga2.jl`
- `src/algorithms/swarm.jl`
- `src/experiments.jl`

### Responsibilities

#### `src/landscapes.jl`
Move or keep:
- real-data landscape loading
- triangle landscape generation
- lookup-table helpers

#### `src/objectives.jl`
Add:
- `evaluate(landscape, index; objective=:penalized)`
- `evaluate_multi(landscape, index; objectives=...)`
- `is_valid_subset(landscape, index)`
- `repair_subset(landscape, index)`

Important:
- real landscapes exclude subset `0`
- synthetic landscapes may include `0`

So validity must come from the landscape, not from the algorithm.

#### `src/algorithms/ga.jl`
Add:
- GA result struct
- initialization
- tournament selection
- crossover
- mutation
- elitism
- run loop

#### `src/algorithms/nsga2.jl`
Add:
- nondominated sorting
- crowding distance
- tournament selection by rank/crowding
- crossover/mutation reuse from GA
- Pareto front extraction

#### `src/algorithms/swarm.jl`
Add:
- `EvoLP.jl` integration or wrapper code
- conversion between project subset indices and the particle representation expected by `EvoLP.jl`
- objective adapter from lookup-table landscapes to the fitness callback used by the swarm solver
- repair of invalid zero subset for real landscapes
- result extraction into the same format used by the custom algorithms

#### `src/experiments.jl`
Add:
- repeated-run helpers
- fixed-seed execution
- history collection
- result summaries

## Algorithm Plan

## 1. Single-objective GA

### Representation
- one individual = one subset index `Int`

### Operators
- selection: tournament selection
- crossover: one-point or uniform crossover
- mutation: bit-flip with rate about `1 / n_features`
- elitism: keep top 1 or 2 individuals

### Output
Return:
- best subset
- best score
- history of best-so-far per generation
- optional population summary

### Default evaluation
- real landscapes: penalized fitness
- triangle: triangle fitness

## 2. Multi-objective NSGA-II

### Representation
- same subset index `Int`

### Objectives
Default:
- objective 1: maximize raw accuracy
- objective 2: minimize selected features

### Main components
- fast nondominated sorting
- crowding distance
- binary tournament based on rank then crowding
- same crossover and mutation as GA

### Output
Return:
- final Pareto set
- rank and crowding values
- generation summaries
- hypervolume optional, not required initially

## 3. Swarm Algorithm

### Representation
- preferred: use the particle representation expected by `EvoLP.jl`
- keep a conversion layer to and from subset indices
- internally, still report all final answers as subset indices and active columns

### Updates
- prefer `EvoLP.jl`'s PSO components rather than writing the whole swarm loop from scratch
- adapt the objective callback so each particle can be evaluated on:
  - penalized real-data fitness
  - raw real-data fitness if needed
  - synthetic triangle fitness
- if needed, add a thin binary repair layer after each particle update

### Important constraint
For real landscapes:
- repair or reject subset `0`

### Output
Return:
- best subset found
- best score
- iteration history
- enough metadata to compare fairly against GA and NSGA-II

## Experiment Plan

Use a common experimental structure across all three algorithms.

### Datasets
Test on:
- `breast-w`
- `credit-a`
- `letter-r`
- synthetic triangle

### Budgets
Use equal total evaluation budgets where possible.

Examples:
- GA: generations x population size
- NSGA-II: generations x population size
- PSO: iterations x swarm size

Compare algorithms using:
- total fitness evaluations

### Repetitions
Run:
- 20 to 30 seeds per setup

### Metrics

#### Single-objective algorithms
Record:
- best fitness
- best raw accuracy
- best penalized fitness
- number of selected features
- subset index
- time per run

#### Multi-objective algorithm
Record:
- Pareto front size
- best raw accuracy on front
- smallest subset on front
- representative tradeoff points

### Strong advantage of this repo
Because the full landscape is known, compare found solutions against the **exact optimum** or the **exact Pareto front** where feasible.

This is much stronger than a normal black-box benchmark.

## Result File Plan

Suggested output structure:

- `results/ga/<problem>/<seed>/`
- `results/nsga2/<problem>/<seed>/`
- `results/swarm/<problem>/<seed>/`

Per run store:

### GA / Swarm
- `config.toml`
- `history.tsv`
- `best.tsv`

### NSGA-II
- `config.toml`
- `history.tsv`
- `pareto.tsv`

Useful columns:
- seed
- evaluations
- generation or iteration
- subset_index
- raw_accuracy
- penalized_fitness
- mean_time
- n_active

## Script Plan

Add root-level run scripts to match the current repo style:

- `run_ga.jl`
- `run_nsga2.jl`
- `run_swarm.jl`

Optional later:
- `plot_algorithm_results.jl`

Each run script should:
- parse CLI arguments
- load a chosen landscape
- run the algorithm
- print a compact summary
- optionally save results

## Test Plan

Keep tests deterministic and small.

### Objective / evaluation tests
- `evaluate` matches lookup-table values
- real landscapes reject subset `0`
- triangle landscape handles `0` correctly

### GA tests
- initialization only produces valid subsets
- mutation stays in valid range
- crossover stays in valid range
- finds known optimum on a tiny triangle instance with fixed seed

### NSGA-II tests
- nondominated sorting works on a toy objective set
- crowding distance behaves correctly on front extremes
- final returned set is actually nondominated

### Swarm tests
- the `EvoLP.jl` wrapper evaluates the correct landscape values
- particle outputs are converted back to valid subset indices
- real landscapes never return subset `0`
- reaches a known optimum on a tiny triangle instance with fixed seed and budget

### Integration tests
- one short run per algorithm on `breast-w`
- check returned result structure is consistent

## Risks To Watch

### 1. Empty subset handling
Real landscapes start at index `1`, not `0`.
This must be enforced centrally.

### 2. Bit ordering confusion
Current code uses:
- one convention for feature decoding
- another for HBM plotting

Algorithms should avoid decoding whenever possible and stay on integer bitmasks.

### 3. Multi-objective story drift
Too many objective variants will make the report messy.
Use one default:
- accuracy vs feature count

### 4. Overcomplication
Do not add:
- advanced adaptive operators
- heavy experiment frameworks
- extra dependencies unless necessary

Keep it student-project sized.

## Recommended Build Order

1. Add objective wrapper functions
2. Implement GA first
3. Add one run script for GA
4. Implement NSGA-II using GA operators
5. Add one run script for NSGA-II
6. Add `EvoLP.jl` and implement the swarm wrapper
7. Add one run script for the swarm algorithm
8. Add repeated-run experiment helpers
9. Add tests
10. Add comparison plots or summary tables last

## Suggested Deliverable Story

A clean report structure could be:

1. Landscape construction from lookup tables
2. Single-objective optimization with GA
3. Multi-objective optimization with NSGA-II
4. Swarm optimization with PSO via `EvoLP.jl`
5. Comparison against exact optima / Pareto fronts
6. Discussion of regularization, feature count, and algorithm behavior

## Final Recommendation

Use:

- **GA** for single-objective
- **NSGA-II** for multi-objective
- **PSO using `EvoLP.jl`** for swarm intelligence

This is the cleanest combination for the current codebase and gives a coherent implementation story.
