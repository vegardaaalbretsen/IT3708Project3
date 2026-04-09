module IT3708Project3

include("datasets.jl")
include("parser.jl")
include("landscape.jl")

export DATASETS,
       default_output_path,
       parse_dataset,
       write_csv,
       load_landscape,
       penalty,
       penalized_fitness,
       apply_penalty

end # module IT3708Project3
