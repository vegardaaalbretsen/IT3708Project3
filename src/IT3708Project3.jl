module IT3708Project3

include("datasets.jl")
include("types.jl")
include("parser.jl")
include("landscape.jl")
include("ea.jl")
include("triangle.jl")
include("hbm.jl")
include("visualization.jl")

export DATASETS,
       Landscape,
       default_output_path,
       default_hbm_plot_path,
       default_feature_count_plot_path,
       default_ea_plot_path,
       parse_dataset,
       write_csv,
       load_landscape,
       load_landscape_key,
       fitness_values,
       penalized_fitness_values,
       fitness,
       penalty,
       penalized_fitness,
       standard_bit_mutation,
       run_standard_ea,
       triangle_fitness,
       triangle_landscape,
       HBMNode,
       build_hbm,
       one_flip_neighbors,
       local_optima,
       global_optima,
       hbm_plot_data,
       plot_hbm,
       save_hbm_plot,
       feature_count_plot_data,
       plot_fitness_by_feature_count,
       save_fitness_by_feature_count_plot,
       plot_fitness_by_feature_count_with_ea,
       save_fitness_by_feature_count_with_ea_plot,
       ea_trace_plot_data,
       plot_ea_trace,
       save_ea_trace_plot

end # module IT3708Project3
