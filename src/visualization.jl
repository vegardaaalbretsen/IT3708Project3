using Plots
using Statistics

function hbm_marker_sizes(x_cells::Int, y_cells::Int)
    base_size = clamp(min(520 / x_cells, 340 / y_cells), 1.5, 18.0)
    optimum_size = clamp(base_size * 1.45, 3.0, 26.0)
    return (base = base_size, optimum = optimum_size)
end

function top_local_optima(nodes::AbstractVector{HBMNode},
                          local_indices::AbstractVector{<:Integer},
                          global_indices::AbstractVector{<:Integer};
                          max_count::Int = 50)
    node_lookup = Dict(node.index => node for node in nodes)
    global_set = Set(global_indices)
    local_only = [index for index in local_indices if !(index in global_set)]

    sort!(
        local_only;
        by = index -> (-node_lookup[index].fitness, index),
    )

    return local_only[1:min(max_count, length(local_only))]
end

function hbm_plot_data(nodes::AbstractVector{HBMNode}, n_features::Int; allow_zero::Bool = false)
    x = Float64[]
    y = Float64[]
    fitness = Float64[]
    labels = String[]
    local_indices = local_optima(nodes, n_features; allow_zero=allow_zero)
    global_indices = global_optima(nodes)

    for node in nodes
        push!(x, Float64(node.x))
        push!(y, Float64(node.y))
        push!(fitness, node.fitness)
        push!(labels, string(node.index))
    end

    return (
        x = x,
        y = y,
        fitness = fitness,
        labels = labels,
        local_optima = local_indices,
        global_optima = global_indices,
    )
end

function hbm_plot_data(landscape::Landscape; values = fitness_values(landscape))
    return hbm_plot_data(
        build_hbm(landscape; values=values),
        landscape.num_features;
        allow_zero=landscape.allow_zero,
    )
end

function hbm_plot_data(landscape::TriangleByteLandscape;
                       max_local_optima::Int = 49)
    sample = triangle_asym_hbm_nodes(landscape; max_local_optima=max_local_optima)
    return hbm_plot_data(
        sample.nodes,
        landscape.num_features;
        allow_zero=landscape.allow_zero,
    )
end

function plot_hbm(nodes::AbstractVector{HBMNode},
                  n_features::Int;
                  title::AbstractString = "HBM Plot",
                  fitness_label::AbstractString = "Fitness",
                  max_local_optima::Int = 50,
                  size::Tuple{Int, Int} = (2200, 1400),
                  dpi::Int = 300,
                  allow_zero::Bool = false)
    plot_data = hbm_plot_data(nodes, n_features; allow_zero=allow_zero)
    isempty(nodes) && throw(ArgumentError("nodes must not be empty"))

    node_lookup = Dict(node.index => node for node in nodes)
    x_bits = ceil(Int, n_features / 2)
    y_bits = fld(n_features, 2)
    x_max = (1 << x_bits) - 1
    y_max = (1 << y_bits) - 1
    marker_sizes = hbm_marker_sizes(x_max + 1, y_max + 1)
    x_ticks = ([0.0, Float64(x_max)], ["2^0 - 1", "2^$(x_bits) - 1"])
    y_ticks = ([0.0, Float64(y_max)], ["2^0 - 1", "2^$(y_bits) - 1"])
    local_only = top_local_optima(
        nodes,
        plot_data.local_optima,
        plot_data.global_optima;
        max_count = max_local_optima,
    )

    base_plt = scatter(
        plot_data.x,
        plot_data.y;
        marker_z = plot_data.fitness,
        color = cgrad([:darkgreen, :ivory, :purple]),
        ms = marker_sizes.base,
        markerstrokewidth = 0,
        xlabel = "First half of bitstring",
        ylabel = "Second half of bitstring",
        title = title,
        label = "",
        legend = false,
        colorbar = :right,
        colorbar_title = "",
        colorbar_tickfontsize = 22,
        aspect_ratio = :equal,
        size = size,
        dpi = dpi,
        xticks = x_ticks,
        yticks = y_ticks,
        xlims = (-0.8, x_max + 0.8),
        ylims = (-0.8, y_max + 0.8),
        grid = true,
        gridalpha = 0.18,
        framestyle = :box,
        background_color = :white,
        tickfontsize = 24,
        guidefontsize = 34,
        titlefontsize = 54,
        left_margin = 14Plots.mm,
        right_margin = 12Plots.mm,
        top_margin = 10Plots.mm,
        bottom_margin = 18Plots.mm,
    )

    if !isempty(local_only)
        local_nodes = [node_lookup[index] for index in local_only]
        scatter!(
            base_plt,
            Float64[node.x for node in local_nodes],
            Float64[node.y for node in local_nodes];
            ms = marker_sizes.optimum,
            markershape = :circle,
            markercolor = :dodgerblue3,
            markeralpha = 1.0,
            markerstrokewidth = 0,
            label = "",
        )
    end

    if !isempty(plot_data.global_optima)
        global_nodes = [node_lookup[index] for index in plot_data.global_optima]
        scatter!(
            base_plt,
            Float64[node.x for node in global_nodes],
            Float64[node.y for node in global_nodes];
            ms = marker_sizes.optimum,
            markershape = :circle,
            markercolor = :red2,
            markeralpha = 0.98,
            markerstrokewidth = 0,
            label = "",
        )
    end

    info_plt = plot(
        xlim = (0, 1),
        ylim = (0, 1);
        legend = :best,
        legend_column = 2,
        framestyle = :none,
        grid = false,
        showaxis = false,
        ticks = nothing,
        background_color = :white,
        legend_background_color = :white,
        legend_font_pointsize = 24,
        left_margin = 0Plots.mm,
        right_margin = 0Plots.mm,
        top_margin = 0Plots.mm,
        bottom_margin = 0Plots.mm,
    )

        scatter!(
            info_plt,
            [2.0],
            [2.0];
            ms = 26,
            markershape = :circle,
            markercolor = :dodgerblue3,
            markerstrokewidth = 0,
            label = "Local optima",
        )

        scatter!(
            info_plt,
            [2.0],
            [2.0];
            ms = 26,
            markershape = :circle,
            markercolor = :red2,
            markerstrokewidth = 0,
            label = "Global optima",
        )

    label_plt = plot(
        xlim = (0, 1),
        ylim = (0, 1);
        framestyle = :none,
        grid = false,
        showaxis = false,
        ticks = nothing,
        background_color = :white,
        left_margin = 0Plots.mm,
        right_margin = 0Plots.mm,
        top_margin = 0Plots.mm,
        bottom_margin = 0Plots.mm,
        annotations = [(0.5, 0.5, text(fitness_label, 30, :black, rotation = 90))],
    )

    lower_plt = plot(
        base_plt,
        label_plt;
        layout = grid(1, 2, widths = [0.95, 0.05]),
        background_color = :white,
    )

    return plot(
        info_plt,
        lower_plt;
        layout = grid(2, 1, heights = [0.12, 0.88]),
        size = size,
        dpi = dpi,
        background_color = :white,
    )
end

function plot_hbm(landscape::Landscape;
                  values = fitness_values(landscape),
                  title::AbstractString = "$(landscape.name) HBM Plot",
                  fitness_label::AbstractString = "Fitness",
                  max_local_optima::Int = 50,
                  size::Tuple{Int, Int} = (2200, 1400),
                  dpi::Int = 300)
    return plot_hbm(
        build_hbm(landscape; values=values),
        landscape.num_features;
        title=title,
        fitness_label=fitness_label,
        max_local_optima=max_local_optima,
        size=size,
        dpi=dpi,
        allow_zero=landscape.allow_zero,
    )
end

function plot_hbm(landscape::TriangleByteLandscape;
                  title::AbstractString = "$(landscape.name) HBM Plot",
                  fitness_label::AbstractString = "Fitness",
                  max_local_optima::Int = 49,
                  size::Tuple{Int, Int} = (2200, 1400),
                  dpi::Int = 300)
    sample = triangle_asym_hbm_nodes(landscape; max_local_optima=max_local_optima)
    return plot_hbm(
        sample.nodes,
        landscape.num_features;
        title=title,
        fitness_label=fitness_label,
        max_local_optima=max_local_optima,
        size=size,
        dpi=dpi,
        allow_zero=landscape.allow_zero,
    )
end

function save_hbm_plot(nodes::AbstractVector{HBMNode},
                       n_features::Int,
                       output_path::AbstractString;
                       title::AbstractString = "HBM Plot",
                       fitness_label::AbstractString = "Fitness",
                       max_local_optima::Int = 50,
                       size::Tuple{Int, Int} = (2200, 1400),
                       dpi::Int = 300,
                       allow_zero::Bool = false)
    plt = plot_hbm(
        nodes,
        n_features;
        title=title,
        fitness_label=fitness_label,
        max_local_optima=max_local_optima,
        size=size,
        dpi=dpi,
        allow_zero=allow_zero,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function save_hbm_plot(landscape::Landscape,
                       output_path::AbstractString;
                       values = fitness_values(landscape),
                       title::AbstractString = "$(landscape.name) HBM Plot",
                       fitness_label::AbstractString = "Fitness",
                       max_local_optima::Int = 50,
                       size::Tuple{Int, Int} = (2200, 1400),
                       dpi::Int = 300)
    plt = plot_hbm(
        landscape;
        values=values,
        title=title,
        fitness_label=fitness_label,
        max_local_optima=max_local_optima,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function save_hbm_plot(landscape::TriangleByteLandscape,
                       output_path::AbstractString;
                       title::AbstractString = "$(landscape.name) HBM Plot",
                       fitness_label::AbstractString = "Fitness",
                       max_local_optima::Int = 49,
                       size::Tuple{Int, Int} = (2200, 1400),
                       dpi::Int = 300)
    plt = plot_hbm(
        landscape;
        title=title,
        fitness_label=fitness_label,
        max_local_optima=max_local_optima,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function feature_count_plot_data(landscape::Landscape; values = fitness_values(landscape))
    length(values) == length(landscape.indices) || throw(ArgumentError("values must match landscape size"))

    counts = sort(unique(landscape.num_selected))
    means = Float64[]
    maximums = Float64[]

    for count in counts
        positions = findall(==(count), landscape.num_selected)
        count_values = Float64[values[position] for position in positions]
        push!(means, mean(count_values))
        push!(maximums, maximum(count_values))
    end

    return (
        feature_counts = counts,
        mean_fitness = means,
        max_fitness = maximums,
        local_optima = local_optima(landscape; values=values),
        global_optima = global_optima(landscape; values=values),
    )
end

function feature_count_plot_data(landscape::TriangleByteLandscape)
    counts = collect(0:landscape.num_features)
    values = triangle_asym_fitness_by_count(landscape.num_features)
    optimum_counts = triangle_asym_local_optimum_counts(landscape.num_features)

    return (
        feature_counts = counts,
        mean_fitness = values,
        max_fitness = values,
        local_optima_counts = Int[count for count in optimum_counts if count < landscape.num_features],
        global_optima_counts = Int[count for count in optimum_counts if count == landscape.num_features],
    )
end

function plot_fitness_by_feature_count(landscape::Landscape;
                                       values = fitness_values(landscape),
                                       title::AbstractString = "$(landscape.name) Fitness by Feature Count",
                                       fitness_label::AbstractString = "Fitness",
                                       size::Tuple{Int, Int} = (1400, 900),
                                       dpi::Int = 200)
    plot_data = feature_count_plot_data(landscape; values=values)
    value_lookup = Dict(landscape.indices[i] => Float64(values[i]) for i in eachindex(landscape.indices))
    count_lookup = Dict(landscape.indices[i] => landscape.num_selected[i] for i in eachindex(landscape.indices))

    plt = scatter(
        landscape.num_selected,
        values;
        ms = 3.0,
        markeralpha = 0.28,
        markercolor = :gray35,
        markerstrokewidth = 0,
        label = "Subsets",
        xlabel = "Number of selected features",
        ylabel = fitness_label,
        title = title,
        legend = :bottomright,
        xlims = (0, landscape.num_features + 0.5),
        xticks = 0:landscape.num_features,
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        size = size,
        dpi = dpi,
    )

    plot!(
        plt,
        plot_data.feature_counts,
        plot_data.mean_fitness;
        linewidth = 3,
        color = :dodgerblue3,
        label = "Mean",
    )

    plot!(
        plt,
        plot_data.feature_counts,
        plot_data.max_fitness;
        linewidth = 3,
        color = :darkorange3,
        label = "Best per count",
    )

    if !isempty(plot_data.local_optima)
        scatter!(
            plt,
            [count_lookup[index] for index in plot_data.local_optima],
            [value_lookup[index] for index in plot_data.local_optima];
            ms = 5.5,
            markercolor = :purple3,
            markeralpha = 0.9,
            markerstrokewidth = 0,
            label = "Local optima",
        )
    end

    if !isempty(plot_data.global_optima)
        scatter!(
            plt,
            [count_lookup[index] for index in plot_data.global_optima],
            [value_lookup[index] for index in plot_data.global_optima];
            ms = 8.0,
            markercolor = :red2,
            markeralpha = 1.0,
            markerstrokewidth = 0,
            label = "Global optima",
        )
    end

    return plt
end

function plot_fitness_by_feature_count(landscape::TriangleByteLandscape;
                                       title::AbstractString = "$(landscape.name) Fitness by Feature Count",
                                       fitness_label::AbstractString = "Fitness",
                                       size::Tuple{Int, Int} = (1400, 900),
                                       dpi::Int = 200)
    plot_data = feature_count_plot_data(landscape)

    plt = scatter(
        plot_data.feature_counts,
        plot_data.max_fitness;
        ms = 7.5,
        markercolor = :gray35,
        markerstrokewidth = 0,
        label = "Counts",
        xlabel = "Number of selected features",
        ylabel = fitness_label,
        title = title,
        legend = :bottomright,
        xlims = (0, landscape.num_features + 0.5),
        xticks = 0:landscape.num_features,
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        framestyle = :box,
        size = size,
        dpi = dpi,
    )

    plot!(
        plt,
        plot_data.feature_counts,
        plot_data.max_fitness;
        linewidth = 3,
        color = :dodgerblue3,
        label = "Triangle fitness",
    )

    if !isempty(plot_data.local_optima_counts)
        scatter!(
            plt,
            plot_data.local_optima_counts,
            [triangle_asym_fitness(count) for count in plot_data.local_optima_counts];
            ms = 8.5,
            markercolor = :purple3,
            markeralpha = 0.9,
            markerstrokewidth = 0,
            label = "Local optima counts",
        )
    end

    scatter!(
        plt,
        plot_data.global_optima_counts,
        [triangle_asym_fitness(count) for count in plot_data.global_optima_counts];
        ms = 11.0,
        markercolor = :red2,
        markeralpha = 1.0,
        markerstrokewidth = 0,
        label = "Global optimum count",
    )

    return plt
end

function save_fitness_by_feature_count_plot(landscape::Landscape,
                                            output_path::AbstractString;
                                            values = fitness_values(landscape),
                                            title::AbstractString = "$(landscape.name) Fitness by Feature Count",
                                            fitness_label::AbstractString = "Fitness",
                                            size::Tuple{Int, Int} = (1400, 900),
                                            dpi::Int = 200)
    plt = plot_fitness_by_feature_count(
        landscape;
        values=values,
        title=title,
        fitness_label=fitness_label,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function save_fitness_by_feature_count_plot(landscape::TriangleByteLandscape,
                                            output_path::AbstractString;
                                            title::AbstractString = "$(landscape.name) Fitness by Feature Count",
                                            fitness_label::AbstractString = "Fitness",
                                            size::Tuple{Int, Int} = (1400, 900),
                                            dpi::Int = 200)
    plt = plot_fitness_by_feature_count(
        landscape;
        title=title,
        fitness_label=fitness_label,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function ea_trace_plot_data(result)
    current_history = getproperty(result, :current_history)
    best_history = getproperty(result, :best_history)
    current_num_selected_history = getproperty(result, :current_num_selected_history)
    best_num_selected_history = getproperty(result, :best_num_selected_history)

    any(isnothing, (current_history, best_history, current_num_selected_history, best_num_selected_history)) &&
        throw(ArgumentError("EA history is missing. Run run_single_objective_ea(...; keep_history=true) before plotting."))

    return (
        iterations = collect(0:(length(current_history) - 1)),
        current_fitness = current_history,
        best_fitness = best_history,
        current_num_selected = current_num_selected_history,
        best_num_selected = best_num_selected_history,
    )
end

function plot_ea_trace(result;
                       title::AbstractString = "EA Trace",
                       fitness_label::AbstractString = "Penalized fitness",
                       size::Tuple{Int, Int} = (1400, 900),
                       dpi::Int = 200)
    plot_data = ea_trace_plot_data(result)
    max_count = max(maximum(plot_data.current_num_selected), maximum(plot_data.best_num_selected))

    fitness_plt = plot(
        plot_data.iterations,
        plot_data.current_fitness;
        linewidth = 2.5,
        color = :dodgerblue3,
        label = "Current",
        ylabel = fitness_label,
        title = title,
        legend = :bottomright,
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        framestyle = :box,
        tickfontsize = 10,
        guidefontsize = 12,
        titlefontsize = 16,
    )

    plot!(
        fitness_plt,
        plot_data.iterations,
        plot_data.best_fitness;
        linewidth = 3.0,
        color = :darkorange3,
        label = "Best so far",
    )

    count_plt = plot(
        plot_data.iterations,
        plot_data.current_num_selected;
        linewidth = 2.5,
        color = :darkgreen,
        label = "Current",
        xlabel = "Iteration",
        ylabel = "Selected features",
        legend = :topright,
        ylims = (-0.2, max_count + 0.2),
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        framestyle = :box,
        tickfontsize = 10,
        guidefontsize = 12,
    )

    plot!(
        count_plt,
        plot_data.iterations,
        plot_data.best_num_selected;
        linewidth = 3.0,
        color = :purple3,
        label = "Best so far",
    )

    return plot(
        fitness_plt,
        count_plt;
        layout = grid(2, 1, heights = [0.62, 0.38]),
        size = size,
        dpi = dpi,
        background_color = :white,
        link = :x,
    )
end

function save_ea_trace_plot(result,
                            output_path::AbstractString;
                            title::AbstractString = "EA Trace",
                            fitness_label::AbstractString = "Penalized fitness",
                            size::Tuple{Int, Int} = (1400, 900),
                            dpi::Int = 200)
    plt = plot_ea_trace(
        result;
        title=title,
        fitness_label=fitness_label,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function nsga2_pareto_plot_data(result)
    pareto_indices = getproperty(result, :pareto_indices)
    pareto_accuracy = getproperty(result, :pareto_accuracy)
    pareto_num_selected = getproperty(result, :pareto_num_selected)
    pareto_time = getproperty(result, :pareto_time)
    pareto_penalized_fitness = getproperty(result, :pareto_penalized_fitness)

    lengths = (
        length(pareto_indices),
        length(pareto_accuracy),
        length(pareto_num_selected),
        length(pareto_time),
        length(pareto_penalized_fitness),
    )
    all(==(first(lengths)), lengths) ||
        throw(ArgumentError("NSGA-II Pareto result vectors must have the same length"))

    return (
        indices = pareto_indices,
        accuracy = pareto_accuracy,
        num_selected = pareto_num_selected,
        time = pareto_time,
        penalized_fitness = pareto_penalized_fitness,
        best_penalized_index = getproperty(result, :best_penalized_index),
        best_penalized_accuracy = getproperty(result, :best_penalized_accuracy),
        best_penalized_num_selected = getproperty(result, :best_penalized_num_selected),
        best_penalized_time = getproperty(result, :best_penalized_time),
        best_penalized_fitness = getproperty(result, :best_penalized_fitness),
        epsilon = getproperty(result, :epsilon),
    )
end

function plot_nsga2_pareto_front(landscape::Landscape,
                                 result;
                                 title::AbstractString = "$(landscape.name) NSGA-II Pareto Front",
                                 size::Tuple{Int, Int} = (1400, 900),
                                 dpi::Int = 200)
    plot_data = nsga2_pareto_plot_data(result)

    plt = scatter(
        landscape.num_selected,
        landscape.accuracy;
        ms = 3.0,
        markeralpha = 0.24,
        markercolor = :gray35,
        markerstrokewidth = 0,
        label = "Subsets",
        xlabel = "Number of selected features",
        ylabel = "Accuracy",
        title = title,
        legend = :bottomright,
        xlims = (0, landscape.num_features + 0.5),
        xticks = 0:landscape.num_features,
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        framestyle = :box,
        size = size,
        dpi = dpi,
    )

    if !isempty(plot_data.indices)
        plot!(
            plt,
            plot_data.num_selected,
            plot_data.accuracy;
            linewidth = 2.8,
            color = :black,
            alpha = 0.7,
            label = "Pareto front",
        )

        scatter!(
            plt,
            plot_data.num_selected,
            plot_data.accuracy;
            ms = 8.5,
            markercolor = :dodgerblue3,
            markerstrokecolor = :black,
            markerstrokewidth = 0.5,
            label = "Pareto points",
        )

        scatter!(
            plt,
            [plot_data.best_penalized_num_selected],
            [plot_data.best_penalized_accuracy];
            ms = 12.0,
            markershape = :star5,
            markercolor = :gold3,
            markerstrokecolor = :black,
            markerstrokewidth = 0.8,
            label = plot_data.epsilon == 0 ? "Best accuracy summary" : "Best penalized summary",
        )
    end

    return plt
end

function save_nsga2_pareto_front_plot(landscape::Landscape,
                                      result,
                                      output_path::AbstractString;
                                      title::AbstractString = "$(landscape.name) NSGA-II Pareto Front",
                                      size::Tuple{Int, Int} = (1400, 900),
                                      dpi::Int = 200)
    plt = plot_nsga2_pareto_front(
        landscape,
        result;
        title=title,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function nsga2_trace_plot_data(result)
    pareto_accuracy_history = getproperty(result, :pareto_accuracy_history)
    pareto_num_selected_history = getproperty(result, :pareto_num_selected_history)
    pareto_time_history = getproperty(result, :pareto_time_history)
    pareto_penalized_fitness_history = getproperty(result, :pareto_penalized_fitness_history)
    front_size_history = getproperty(result, :front_size_history)

    any(isnothing, (pareto_accuracy_history, pareto_num_selected_history, pareto_time_history, pareto_penalized_fitness_history, front_size_history)) &&
        throw(ArgumentError("NSGA-II history is missing. Run run_nsga2_feature_ea(...; keep_history=true) before plotting."))

    return (
        iterations = collect(0:(length(front_size_history) - 1)),
        best_accuracy = [maximum(history) for history in pareto_accuracy_history],
        min_num_selected = [minimum(history) for history in pareto_num_selected_history],
        min_time = [minimum(history) for history in pareto_time_history],
        best_penalized_fitness = [maximum(history) for history in pareto_penalized_fitness_history],
        front_size = front_size_history,
    )
end

function plot_nsga2_trace(result;
                          title::AbstractString = "NSGA-II Trace",
                          size::Tuple{Int, Int} = (1400, 1000),
                          dpi::Int = 200)
    plot_data = nsga2_trace_plot_data(result)
    show_penalized = any(
        !isapprox(accuracy, penalized; atol=1e-12)
        for (accuracy, penalized) in zip(plot_data.best_accuracy, plot_data.best_penalized_fitness)
    )
    max_front = maximum(plot_data.front_size)
    max_selected = maximum(plot_data.min_num_selected)

    accuracy_plt = plot(
        plot_data.iterations,
        plot_data.best_accuracy;
        linewidth = 3.0,
        color = :dodgerblue3,
        label = "Max front accuracy",
        ylabel = "Accuracy",
        title = title,
        legend = :bottomright,
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        framestyle = :box,
        tickfontsize = 10,
        guidefontsize = 12,
        titlefontsize = 16,
    )

    if show_penalized
        plot!(
            accuracy_plt,
            plot_data.iterations,
            plot_data.best_penalized_fitness;
            linewidth = 2.4,
            linestyle = :dash,
            color = :purple3,
            label = "Best penalized on front",
        )
    end

    features_plt = plot(
        plot_data.iterations,
        plot_data.min_num_selected;
        linewidth = 3.0,
        color = :darkgreen,
        label = "Min selected features",
        ylabel = "Selected features",
        legend = :topright,
        ylims = (-0.2, max_selected + 0.2),
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        framestyle = :box,
        tickfontsize = 10,
        guidefontsize = 12,
    )

    time_plt = plot(
        plot_data.iterations,
        plot_data.min_time;
        linewidth = 3.0,
        color = :firebrick3,
        label = "Min evaluation time",
        ylabel = "Time",
        legend = :topright,
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        framestyle = :box,
        tickfontsize = 10,
        guidefontsize = 12,
    )

    front_plt = plot(
        plot_data.iterations,
        plot_data.front_size;
        linewidth = 3.0,
        color = :black,
        label = "Unique Pareto points",
        xlabel = "Iteration",
        ylabel = "Front size",
        legend = :topright,
        ylims = (0.8, max_front + 0.2),
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        framestyle = :box,
        tickfontsize = 10,
        guidefontsize = 12,
    )

    return plot(
        accuracy_plt,
        features_plt,
        time_plt,
        front_plt;
        layout = grid(4, 1, heights = [0.34, 0.22, 0.22, 0.22]),
        size = size,
        dpi = dpi,
        background_color = :white,
        link = :x,
    )
end

function save_nsga2_trace_plot(result,
                               output_path::AbstractString;
                               title::AbstractString = "NSGA-II Trace",
                               size::Tuple{Int, Int} = (1400, 1000),
                               dpi::Int = 200)
    plt = plot_nsga2_trace(
        result;
        title=title,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function nsga2_search_trajectory_network_data(landscape::Landscape,
                                             result;
                                             values = penalized_fitness_values(landscape, getproperty(result, :epsilon)))
    population_indices_history = getproperty(result, :population_indices_history)
    transition_edges_history = getproperty(result, :transition_edges_history)

    any(isnothing, (population_indices_history, transition_edges_history)) &&
        throw(ArgumentError("NSGA-II history is missing. Run run_nsga2_feature_ea(...; keep_history=true) before plotting the search trajectory network."))

    isempty(population_indices_history) &&
        throw(ArgumentError("NSGA-II population history must not be empty"))

    nodes = build_hbm(landscape; values=values)
    node_lookup = Dict(node.index => node for node in nodes)
    visit_counts = Dict{Int, Int}()
    start_counts = Dict{Int, Int}()
    end_counts = Dict{Int, Int}()
    edge_counts = Dict{Tuple{Int, Int}, Int}()

    for index in first(population_indices_history)
        start_counts[index] = get(start_counts, index, 0) + 1
    end

    for snapshot in population_indices_history
        for index in snapshot
            visit_counts[index] = get(visit_counts, index, 0) + 1
        end
    end

    for index in last(population_indices_history)
        end_counts[index] = get(end_counts, index, 0) + 1
    end

    for generation_edges in transition_edges_history
        for (from, to) in generation_edges
            from == to && continue
            edge = from < to ? (from, to) : (to, from)
            edge_counts[edge] = get(edge_counts, edge, 0) + 1
        end
    end

    visited_indices = sort!(collect(keys(visit_counts)); by = index -> (-visit_counts[index], index))
    start_indices = sort!(collect(keys(start_counts)); by = index -> (-start_counts[index], index))
    end_indices = sort!(collect(keys(end_counts)); by = index -> (-end_counts[index], index))

    return (
        nodes = nodes,
        node_lookup = node_lookup,
        visited_indices = visited_indices,
        visit_counts = visit_counts,
        start_indices = start_indices,
        start_counts = start_counts,
        end_indices = end_indices,
        end_counts = end_counts,
        edge_counts = edge_counts,
        pareto_indices = getproperty(result, :pareto_indices),
        best_penalized_index = getproperty(result, :best_penalized_index),
    )
end

function plot_nsga2_search_trajectory_network(landscape::Landscape,
                                              result;
                                              values = penalized_fitness_values(landscape, getproperty(result, :epsilon)),
                                              title::AbstractString = "$(landscape.name) NSGA-II Search Trajectory Network",
                                              fitness_label::AbstractString = "Penalized fitness",
                                              size::Tuple{Int, Int} = (2200, 1400),
                                              dpi::Int = 300)
    network_data = nsga2_search_trajectory_network_data(landscape, result; values=values)
    nodes = network_data.nodes
    plot_data = hbm_plot_data(nodes, landscape.num_features; allow_zero=landscape.allow_zero)
    isempty(nodes) && throw(ArgumentError("nodes must not be empty"))

    node_lookup = network_data.node_lookup
    x_bits = ceil(Int, landscape.num_features / 2)
    y_bits = fld(landscape.num_features, 2)
    x_max = (1 << x_bits) - 1
    y_max = (1 << y_bits) - 1
    marker_sizes = hbm_marker_sizes(x_max + 1, y_max + 1)
    x_ticks = ([0.0, Float64(x_max)], ["2^0 - 1", "2^$(x_bits) - 1"])
    y_ticks = ([0.0, Float64(y_max)], ["2^0 - 1", "2^$(y_bits) - 1"])

    plt = scatter(
        plot_data.x,
        plot_data.y;
        marker_z = plot_data.fitness,
        color = cgrad([:darkgreen, :ivory, :purple]),
        ms = marker_sizes.base,
        markeralpha = 0.20,
        markerstrokewidth = 0,
        xlabel = "First half of bitstring",
        ylabel = "Second half of bitstring",
        title = title,
        label = "Landscape",
        legend = :outertopright,
        colorbar = :right,
        colorbar_title = fitness_label,
        colorbar_tickfontsize = 18,
        aspect_ratio = :equal,
        size = size,
        dpi = dpi,
        xticks = x_ticks,
        yticks = y_ticks,
        xlims = (-0.8, x_max + 0.8),
        ylims = (-0.8, y_max + 0.8),
        grid = true,
        gridalpha = 0.18,
        framestyle = :box,
        background_color = :white,
        tickfontsize = 18,
        guidefontsize = 24,
        titlefontsize = 28,
    )

    if !isempty(network_data.edge_counts)
        max_edge_count = maximum(Base.values(network_data.edge_counts))
        edge_label_used = false

        for ((from, to), count) in sort!(collect(network_data.edge_counts); by = last)
            from_node = node_lookup[from]
            to_node = node_lookup[to]
            edge_strength = count / max_edge_count

            plot!(
                plt,
                [Float64(from_node.x), Float64(to_node.x)],
                [Float64(from_node.y), Float64(to_node.y)];
                linewidth = 1.0 + 5.0 * sqrt(edge_strength),
                color = :black,
                alpha = 0.12 + 0.52 * edge_strength,
                label = edge_label_used ? "" : "Transitions",
            )
            edge_label_used = true
        end
    end

    if !isempty(network_data.visited_indices)
        max_visit_count = maximum(Base.values(network_data.visit_counts))
        visited_nodes = [node_lookup[index] for index in network_data.visited_indices]
        visited_sizes = [
            marker_sizes.base + 0.95 * sqrt(network_data.visit_counts[index] / max_visit_count) * marker_sizes.optimum
            for index in network_data.visited_indices
        ]
        scatter!(
            plt,
            Float64[node.x for node in visited_nodes],
            Float64[node.y for node in visited_nodes];
            ms = visited_sizes,
            markercolor = :black,
            markeralpha = 0.45,
            markerstrokecolor = :white,
            markerstrokewidth = 0.7,
            label = "Visited states",
        )
    end

    if !isempty(network_data.start_indices)
        max_start_count = maximum(Base.values(network_data.start_counts))
        start_nodes = [node_lookup[index] for index in network_data.start_indices]
        start_sizes = [
            marker_sizes.optimum + 0.7 * sqrt(network_data.start_counts[index] / max_start_count) * marker_sizes.base
            for index in network_data.start_indices
        ]
        scatter!(
            plt,
            Float64[node.x for node in start_nodes],
            Float64[node.y for node in start_nodes];
            ms = start_sizes,
            markershape = :diamond,
            markercolor = :white,
            markerstrokecolor = :black,
            markerstrokewidth = 1.4,
            label = "Initial population",
        )
    end

    if !isempty(network_data.end_indices)
        max_end_count = maximum(Base.values(network_data.end_counts))
        end_nodes = [node_lookup[index] for index in network_data.end_indices]
        end_sizes = [
            marker_sizes.optimum + 0.7 * sqrt(network_data.end_counts[index] / max_end_count) * marker_sizes.base
            for index in network_data.end_indices
        ]
        scatter!(
            plt,
            Float64[node.x for node in end_nodes],
            Float64[node.y for node in end_nodes];
            ms = end_sizes,
            markershape = :utriangle,
            markercolor = :forestgreen,
            markerstrokewidth = 0,
            label = "Final population",
        )
    end

    if !isempty(network_data.pareto_indices)
        pareto_nodes = [node_lookup[index] for index in network_data.pareto_indices]
        scatter!(
            plt,
            Float64[node.x for node in pareto_nodes],
            Float64[node.y for node in pareto_nodes];
            ms = marker_sizes.optimum + 2.0,
            markershape = :circle,
            markercolor = :darkorange3,
            markerstrokecolor = :black,
            markerstrokewidth = 0.7,
            label = "Final Pareto front",
        )
    end

    best_node = node_lookup[network_data.best_penalized_index]
    scatter!(
        plt,
        [Float64(best_node.x)],
        [Float64(best_node.y)];
        ms = marker_sizes.optimum + 6.0,
        markershape = :star5,
        markercolor = :gold3,
        markerstrokecolor = :black,
        markerstrokewidth = 0.9,
        label = "Best penalized summary",
    )

    return plt
end

function save_nsga2_search_trajectory_network_plot(landscape::Landscape,
                                                   result,
                                                   output_path::AbstractString;
                                                   values = penalized_fitness_values(landscape, getproperty(result, :epsilon)),
                                                   title::AbstractString = "$(landscape.name) NSGA-II Search Trajectory Network",
                                                   fitness_label::AbstractString = "Penalized fitness",
                                                   size::Tuple{Int, Int} = (2200, 1400),
                                                   dpi::Int = 300)
    plt = plot_nsga2_search_trajectory_network(
        landscape,
        result;
        values=values,
        title=title,
        fitness_label=fitness_label,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function compress_ea_path(feature_counts::AbstractVector{<:Integer},
                          fitness_values::AbstractVector{<:Real})
    length(feature_counts) == length(fitness_values) ||
        throw(ArgumentError("feature_counts and fitness_values must have the same length"))
    isempty(feature_counts) && return Int[]

    positions = Int[1]
    for i in 2:length(feature_counts)
        if feature_counts[i] != feature_counts[i - 1] || fitness_values[i] != fitness_values[i - 1]
            push!(positions, i)
        end
    end

    return positions
end

function compress_path_positions(x_values::AbstractVector,
                                 y_values::AbstractVector)
    length(x_values) == length(y_values) ||
        throw(ArgumentError("x_values and y_values must have the same length"))
    isempty(x_values) && return Int[]

    positions = Int[1]
    for i in 2:length(x_values)
        if x_values[i] != x_values[i - 1] || y_values[i] != y_values[i - 1]
            push!(positions, i)
        end
    end

    return positions
end

function plot_fitness_by_feature_count_with_ea(landscape::AbstractLandscape,
                                               result;
                                               title::AbstractString = "$(landscape.name) Fitness by Feature Count with EA Path",
                                               fitness_label::AbstractString = "Fitness",
                                               size::Tuple{Int, Int} = (1400, 900),
                                               dpi::Int = 200)
    plt = plot_fitness_by_feature_count(
        landscape;
        title=title,
        fitness_label=fitness_label,
        size=size,
        dpi=dpi,
    )

    trace_data = ea_trace_plot_data(result)
    path_positions = compress_ea_path(trace_data.current_num_selected, trace_data.current_fitness)
    path_counts = [trace_data.current_num_selected[i] for i in path_positions]
    path_fitness = [trace_data.current_fitness[i] for i in path_positions]

    plot!(
        plt,
        path_counts,
        path_fitness;
        linewidth = 2.5,
        color = :black,
        alpha = 0.65,
        label = "EA path",
    )

    scatter!(
        plt,
        path_counts,
        path_fitness;
        ms = 4.2,
        markercolor = :black,
        markeralpha = 0.45,
        markerstrokewidth = 0,
        label = "",
    )

    scatter!(
        plt,
        [path_counts[1]],
        [path_fitness[1]];
        ms = 8.0,
        markershape = :diamond,
        markercolor = :white,
        markerstrokecolor = :black,
        markerstrokewidth = 1.8,
        label = "EA start",
    )

    scatter!(
        plt,
        [path_counts[end]],
        [path_fitness[end]];
        ms = 9.0,
        markershape = :utriangle,
        markercolor = :forestgreen,
        markerstrokewidth = 0,
        label = "EA end",
    )

    scatter!(
        plt,
        [getproperty(result, :best_num_selected)],
        [getproperty(result, :best_penalized_fitness)];
        ms = 10.0,
        markershape = :star5,
        markercolor = :gold3,
        markerstrokecolor = :black,
        markerstrokewidth = 0.8,
        label = "EA best",
    )

    return plt
end

function save_fitness_by_feature_count_with_ea_plot(landscape::AbstractLandscape,
                                                    result,
                                                    output_path::AbstractString;
                                                    title::AbstractString = "$(landscape.name) Fitness by Feature Count with EA Path",
                                                    fitness_label::AbstractString = "Fitness",
                                                    size::Tuple{Int, Int} = (1400, 900),
                                                    dpi::Int = 200)
    plt = plot_fitness_by_feature_count_with_ea(
        landscape,
        result;
        title=title,
        fitness_label=fitness_label,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function swarm_iteration_state(result; iteration::Union{Nothing, Integer} = nothing)
    if isnothing(iteration)
        return (
            iteration = getproperty(result, :iterations),
            particle_indices = getproperty(result, :final_particle_indices),
            best_index = getproperty(result, :best_index),
            best_penalized_fitness = getproperty(result, :best_penalized_fitness),
        )
    end

    particle_index_history = getproperty(result, :particle_index_history)
    best_index_history = getproperty(result, :best_index_history)
    best_penalized_fitness_history = getproperty(result, :best_penalized_fitness_history)

    any(isnothing, (particle_index_history, best_index_history, best_penalized_fitness_history)) &&
        throw(ArgumentError("Swarm history is missing. Run run_swarm_ea(...; keep_history=true) before plotting iteration-specific swarm visualizations or animations."))

    0 <= iteration <= length(particle_index_history) - 1 ||
        throw(ArgumentError("iteration must be between 0 and $(length(particle_index_history) - 1)"))

    position = iteration + 1
    return (
        iteration = Int(iteration),
        particle_indices = particle_index_history[position],
        best_index = best_index_history[position],
        best_penalized_fitness = best_penalized_fitness_history[position],
    )
end

function swarm_snapshot_plot_data(landscape::Landscape,
                                  result;
                                  iteration::Union{Nothing, Integer} = nothing)
    state = swarm_iteration_state(result; iteration=iteration)
    counts_by_index = Dict{Int, Int}()

    for index in state.particle_indices
        counts_by_index[index] = get(counts_by_index, index, 0) + 1
    end

    unique_indices = sort!(collect(keys(counts_by_index)); by = index -> (count_ones(index), index))
    epsilon = getproperty(result, :epsilon)

    return (
        iteration = state.iteration,
        particle_indices = state.particle_indices,
        unique_indices = unique_indices,
        multiplicity = [counts_by_index[index] for index in unique_indices],
        feature_counts = [count_ones(index) for index in unique_indices],
        penalized_fitness = [candidate_state(landscape, index, epsilon).penalized_fitness for index in unique_indices],
        best_index = state.best_index,
        best_penalized_fitness = state.best_penalized_fitness,
    )
end

function mean_pairwise_hamming_distance(indices::AbstractVector{<:Integer})
    n = length(indices)
    n <= 1 && return 0.0

    distance_sum = 0
    pair_count = 0

    for i in 1:(n - 1)
        for j in (i + 1):n
            distance_sum += count_ones(xor(indices[i], indices[j]))
            pair_count += 1
        end
    end

    return distance_sum / pair_count
end

function swarm_trace_plot_data(landscape::Landscape, result)
    particle_index_history = getproperty(result, :particle_index_history)
    best_index_history = getproperty(result, :best_index_history)
    best_penalized_fitness_history = getproperty(result, :best_penalized_fitness_history)

    any(isnothing, (particle_index_history, best_index_history, best_penalized_fitness_history)) &&
        throw(ArgumentError("Swarm history is missing. Run run_swarm_ea(...; keep_history=true) before plotting."))

    epsilon = getproperty(result, :epsilon)
    fitness_lookup = Dict(
        index => candidate_state(landscape, index, epsilon).penalized_fitness
        for index in landscape.indices
    )

    mean_fitness = Float64[]
    median_fitness = Float64[]
    unique_subsets = Int[]
    mean_hamming_distance = Float64[]
    global_best_fraction = Float64[]

    for (position, snapshot) in pairs(particle_index_history)
        snapshot_fitness = Float64[fitness_lookup[index] for index in snapshot]
        push!(mean_fitness, mean(snapshot_fitness))
        push!(median_fitness, median(snapshot_fitness))
        push!(unique_subsets, length(unique(snapshot)))
        push!(mean_hamming_distance, mean_pairwise_hamming_distance(snapshot))
        push!(global_best_fraction, count(==(best_index_history[position]), snapshot) / length(snapshot))
    end

    return (
        iterations = collect(0:(length(particle_index_history) - 1)),
        best_fitness = best_penalized_fitness_history,
        mean_fitness = mean_fitness,
        median_fitness = median_fitness,
        unique_subsets = unique_subsets,
        mean_pairwise_hamming_distance = mean_hamming_distance,
        global_best_fraction = global_best_fraction,
    )
end

function swarm_best_path_plot_data(landscape::Landscape,
                                   result;
                                   iteration::Union{Nothing, Integer} = nothing)
    best_index_history = getproperty(result, :best_index_history)
    best_penalized_fitness_history = getproperty(result, :best_penalized_fitness_history)

    if any(isnothing, (best_index_history, best_penalized_fitness_history))
        isnothing(iteration) ||
            throw(ArgumentError("Swarm history is missing. Run run_swarm_ea(...; keep_history=true) before plotting iteration-specific paths."))
        return nothing
    end

    last_position = if isnothing(iteration)
        length(best_index_history)
    else
        0 <= iteration <= length(best_index_history) - 1 ||
            throw(ArgumentError("iteration must be between 0 and $(length(best_index_history) - 1)"))
        iteration + 1
    end

    path_indices = best_index_history[1:last_position]
    path_fitness = best_penalized_fitness_history[1:last_position]
    path_feature_counts = count_ones.(path_indices)
    feature_path_positions = compress_ea_path(path_feature_counts, path_fitness)
    nodes = build_hbm(landscape; values=fitness_values(landscape))
    node_lookup = Dict(node.index => node for node in nodes)
    path_x = Float64[node_lookup[index].x for index in path_indices]
    path_y = Float64[node_lookup[index].y for index in path_indices]
    hbm_path_positions = compress_path_positions(path_x, path_y)

    return (
        indices = path_indices,
        penalized_fitness = path_fitness,
        feature_counts = path_feature_counts,
        feature_path_positions = feature_path_positions,
        hbm_x = path_x,
        hbm_y = path_y,
        hbm_path_positions = hbm_path_positions,
    )
end

function plot_fitness_by_feature_count_with_swarm(landscape::Landscape,
                                                  result;
                                                  iteration::Union{Nothing, Integer} = nothing,
                                                  values = penalized_fitness_values(landscape, getproperty(result, :epsilon)),
                                                  title::AbstractString = "$(landscape.name) Fitness by Feature Count with Swarm",
                                                  fitness_label::AbstractString = "Penalized fitness",
                                                  size::Tuple{Int, Int} = (1400, 900),
                                                  dpi::Int = 200)
    plt = plot_fitness_by_feature_count(
        landscape;
        values=values,
        title=title,
        fitness_label=fitness_label,
        size=size,
        dpi=dpi,
    )

    snapshot = swarm_snapshot_plot_data(landscape, result; iteration=iteration)
    path_data = swarm_best_path_plot_data(landscape, result; iteration=iteration)
    particle_label = maximum(snapshot.multiplicity) > 1 ? "Particles (size = duplicates)" : "Particles"
    particle_size = 5.0 .+ 3.0 .* sqrt.(Float64.(snapshot.multiplicity))
    best_label = isnothing(iteration) ? "Swarm best" : "Best so far"

    if !isnothing(path_data)
        path_positions = path_data.feature_path_positions
        path_counts = [path_data.feature_counts[i] for i in path_positions]
        path_fitness = [path_data.penalized_fitness[i] for i in path_positions]

        plot!(
            plt,
            path_counts,
            path_fitness;
            linewidth = 2.5,
            color = :black,
            alpha = 0.55,
            label = "Best path",
        )

        scatter!(
            plt,
            path_counts,
            path_fitness;
            ms = 4.4,
            markercolor = :black,
            markeralpha = 0.42,
            markerstrokewidth = 0,
            label = "",
        )

        scatter!(
            plt,
            [path_counts[1]],
            [path_fitness[1]];
            ms = 8.0,
            markershape = :diamond,
            markercolor = :white,
            markerstrokecolor = :black,
            markerstrokewidth = 1.6,
            label = "Path start",
        )
    end

    scatter!(
        plt,
        snapshot.feature_counts,
        snapshot.penalized_fitness;
        ms = particle_size,
        markershape = :circle,
        markercolor = :black,
        markeralpha = 0.55,
        markerstrokecolor = :white,
        markerstrokewidth = 0.6,
        label = particle_label,
    )

    scatter!(
        plt,
        [count_ones(snapshot.best_index)],
        [snapshot.best_penalized_fitness];
        ms = 13.0,
        markershape = :star5,
        markercolor = :gold3,
        markerstrokecolor = :black,
        markerstrokewidth = 0.9,
        label = best_label,
    )

    return plt
end

function save_fitness_by_feature_count_with_swarm_plot(landscape::Landscape,
                                                       result,
                                                       output_path::AbstractString;
                                                       iteration::Union{Nothing, Integer} = nothing,
                                                       values = penalized_fitness_values(landscape, getproperty(result, :epsilon)),
                                                       title::AbstractString = "$(landscape.name) Fitness by Feature Count with Swarm",
                                                       fitness_label::AbstractString = "Penalized fitness",
                                                       size::Tuple{Int, Int} = (1400, 900),
                                                       dpi::Int = 200)
    plt = plot_fitness_by_feature_count_with_swarm(
        landscape,
        result;
        iteration=iteration,
        values=values,
        title=title,
        fitness_label=fitness_label,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function plot_hbm_with_swarm(landscape::Landscape,
                             result;
                             iteration::Union{Nothing, Integer} = nothing,
                             values = penalized_fitness_values(landscape, getproperty(result, :epsilon)),
                             title::AbstractString = "$(landscape.name) HBM with Swarm",
                             fitness_label::AbstractString = "Penalized fitness",
                             max_local_optima::Int = 50,
                             size::Tuple{Int, Int} = (2200, 1400),
                             dpi::Int = 300)
    nodes = build_hbm(landscape; values=values)
    plot_data = hbm_plot_data(nodes, landscape.num_features; allow_zero=landscape.allow_zero)
    isempty(nodes) && throw(ArgumentError("nodes must not be empty"))

    node_lookup = Dict(node.index => node for node in nodes)
    x_bits = ceil(Int, landscape.num_features / 2)
    y_bits = fld(landscape.num_features, 2)
    x_max = (1 << x_bits) - 1
    y_max = (1 << y_bits) - 1
    marker_sizes = hbm_marker_sizes(x_max + 1, y_max + 1)
    x_ticks = ([0.0, Float64(x_max)], ["2^0 - 1", "2^$(x_bits) - 1"])
    y_ticks = ([0.0, Float64(y_max)], ["2^0 - 1", "2^$(y_bits) - 1"])
    local_only = top_local_optima(
        nodes,
        plot_data.local_optima,
        plot_data.global_optima;
        max_count = max_local_optima,
    )
    snapshot = swarm_snapshot_plot_data(landscape, result; iteration=iteration)
    path_data = swarm_best_path_plot_data(landscape, result; iteration=iteration)
    particle_nodes = [node_lookup[index] for index in snapshot.unique_indices]
    particle_size = marker_sizes.base .+ 1.8 .* sqrt.(Float64.(snapshot.multiplicity))
    best_label = isnothing(iteration) ? "Swarm best" : "Best so far"

    base_plt = scatter(
        plot_data.x,
        plot_data.y;
        marker_z = plot_data.fitness,
        color = cgrad([:darkgreen, :ivory, :purple]),
        ms = marker_sizes.base,
        markerstrokewidth = 0,
        xlabel = "First half of bitstring",
        ylabel = "Second half of bitstring",
        title = title,
        label = "",
        legend = false,
        colorbar = :right,
        colorbar_title = "",
        colorbar_tickfontsize = 22,
        aspect_ratio = :equal,
        size = size,
        dpi = dpi,
        xticks = x_ticks,
        yticks = y_ticks,
        xlims = (-0.8, x_max + 0.8),
        ylims = (-0.8, y_max + 0.8),
        grid = true,
        gridalpha = 0.18,
        framestyle = :box,
        background_color = :white,
        tickfontsize = 24,
        guidefontsize = 34,
        titlefontsize = 54,
        left_margin = 14Plots.mm,
        right_margin = 12Plots.mm,
        top_margin = 10Plots.mm,
        bottom_margin = 18Plots.mm,
    )

    if !isempty(local_only)
        local_nodes = [node_lookup[index] for index in local_only]
        scatter!(
            base_plt,
            Float64[node.x for node in local_nodes],
            Float64[node.y for node in local_nodes];
            ms = marker_sizes.optimum,
            markershape = :circle,
            markercolor = :dodgerblue3,
            markeralpha = 1.0,
            markerstrokewidth = 0,
            label = "",
        )
    end

    if !isempty(plot_data.global_optima)
        global_nodes = [node_lookup[index] for index in plot_data.global_optima]
        scatter!(
            base_plt,
            Float64[node.x for node in global_nodes],
            Float64[node.y for node in global_nodes];
            ms = marker_sizes.optimum,
            markershape = :circle,
            markercolor = :red2,
            markeralpha = 0.98,
            markerstrokewidth = 0,
            label = "",
        )
    end

    if !isnothing(path_data)
        path_positions = path_data.hbm_path_positions
        path_x = [path_data.hbm_x[i] for i in path_positions]
        path_y = [path_data.hbm_y[i] for i in path_positions]

        plot!(
            base_plt,
            path_x,
            path_y;
            linewidth = 2.6,
            color = :black,
            alpha = 0.55,
            label = "",
        )

        scatter!(
            base_plt,
            path_x,
            path_y;
            ms = marker_sizes.base + 1.5,
            markercolor = :black,
            markeralpha = 0.38,
            markerstrokewidth = 0,
            label = "",
        )

        scatter!(
            base_plt,
            [path_x[1]],
            [path_y[1]];
            ms = marker_sizes.optimum,
            markershape = :diamond,
            markercolor = :white,
            markerstrokecolor = :black,
            markerstrokewidth = 1.4,
            label = "",
        )
    end

    scatter!(
        base_plt,
        Float64[node.x for node in particle_nodes],
        Float64[node.y for node in particle_nodes];
        ms = particle_size,
        markershape = :circle,
        markercolor = :black,
        markeralpha = 0.55,
        markerstrokecolor = :white,
        markerstrokewidth = 0.7,
        label = "",
    )

    best_node = node_lookup[snapshot.best_index]
    scatter!(
        base_plt,
        [Float64(best_node.x)],
        [Float64(best_node.y)];
        ms = marker_sizes.optimum + 5.0,
        markershape = :star5,
        markercolor = :gold3,
        markerstrokecolor = :black,
        markerstrokewidth = 0.9,
        label = "",
    )

    info_plt = plot(
        xlim = (0, 1),
        ylim = (0, 1);
        legend = :best,
        legend_column = 2,
        framestyle = :none,
        grid = false,
        showaxis = false,
        ticks = nothing,
        background_color = :white,
        legend_background_color = :white,
        legend_font_pointsize = 24,
        left_margin = 0Plots.mm,
        right_margin = 0Plots.mm,
        top_margin = 0Plots.mm,
        bottom_margin = 0Plots.mm,
    )

    scatter!(
        info_plt,
        [2.0],
        [2.0];
        ms = 26,
        markershape = :circle,
        markercolor = :dodgerblue3,
        markerstrokewidth = 0,
        label = "Local optimum",
    )

    scatter!(
        info_plt,
        [2.0],
        [2.0];
        ms = 26,
        markershape = :circle,
        markercolor = :red2,
        markerstrokewidth = 0,
        label = "Global optimum",
    )

    plot!(
        info_plt,
        [2.0, 2.1],
        [2.0, 2.0];
        linewidth = 3.0,
        color = :black,
        alpha = 0.55,
        label = "Best path",
    )

    scatter!(
        info_plt,
        [2.0],
        [2.0];
        ms = 24,
        markershape = :diamond,
        markercolor = :white,
        markerstrokecolor = :black,
        markerstrokewidth = 1.4,
        label = "Path start",
    )

    scatter!(
        info_plt,
        [2.0],
        [2.0];
        ms = 26,
        markershape = :circle,
        markercolor = :black,
        markeralpha = 0.55,
        markerstrokecolor = :white,
        markerstrokewidth = 0.7,
        label = maximum(snapshot.multiplicity) > 1 ? "Particles (size = duplicates)" : "Particles",
    )

    scatter!(
        info_plt,
        [2.0],
        [2.0];
        ms = 30,
        markershape = :star5,
        markercolor = :gold3,
        markerstrokecolor = :black,
        markerstrokewidth = 0.9,
        label = best_label,
    )

    label_plt = plot(
        xlim = (0, 1),
        ylim = (0, 1);
        framestyle = :none,
        grid = false,
        showaxis = false,
        ticks = nothing,
        background_color = :white,
        left_margin = 0Plots.mm,
        right_margin = 0Plots.mm,
        top_margin = 0Plots.mm,
        bottom_margin = 0Plots.mm,
        annotations = [(0.5, 0.5, text(fitness_label, 30, :black, rotation = 90))],
    )

    lower_plt = plot(
        base_plt,
        label_plt;
        layout = grid(1, 2, widths = [0.95, 0.05]),
        background_color = :white,
    )

    return plot(
        info_plt,
        lower_plt;
        layout = grid(2, 1, heights = [0.12, 0.88]),
        size = size,
        dpi = dpi,
        background_color = :white,
    )
end

function save_hbm_with_swarm_plot(landscape::Landscape,
                                  result,
                                  output_path::AbstractString;
                                  iteration::Union{Nothing, Integer} = nothing,
                                  values = penalized_fitness_values(landscape, getproperty(result, :epsilon)),
                                  title::AbstractString = "$(landscape.name) HBM with Swarm",
                                  fitness_label::AbstractString = "Penalized fitness",
                                  max_local_optima::Int = 50,
                                  size::Tuple{Int, Int} = (2200, 1400),
                                  dpi::Int = 300)
    plt = plot_hbm_with_swarm(
        landscape,
        result;
        iteration=iteration,
        values=values,
        title=title,
        fitness_label=fitness_label,
        max_local_optima=max_local_optima,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function plot_swarm_trace(landscape::Landscape,
                          result;
                          title::AbstractString = "Swarm Trace",
                          fitness_label::AbstractString = "Penalized fitness",
                          size::Tuple{Int, Int} = (1400, 1000),
                          dpi::Int = 200)
    plot_data = swarm_trace_plot_data(landscape, result)

    fitness_plt = plot(
        plot_data.iterations,
        plot_data.best_fitness;
        linewidth = 3.0,
        color = :darkorange3,
        label = "Best",
        ylabel = fitness_label,
        title = title,
        legend = :bottomright,
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        framestyle = :box,
        tickfontsize = 10,
        guidefontsize = 12,
        titlefontsize = 16,
    )

    plot!(
        fitness_plt,
        plot_data.iterations,
        plot_data.mean_fitness;
        linewidth = 2.6,
        color = :dodgerblue3,
        label = "Mean",
    )

    plot!(
        fitness_plt,
        plot_data.iterations,
        plot_data.median_fitness;
        linewidth = 2.4,
        linestyle = :dash,
        color = :purple3,
        label = "Median",
    )

    diversity_plt = plot(
        plot_data.iterations,
        plot_data.unique_subsets;
        linewidth = 2.7,
        color = :darkgreen,
        label = "Unique subsets",
        ylabel = "Diversity",
        legend = :topright,
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        framestyle = :box,
        tickfontsize = 10,
        guidefontsize = 12,
    )

    plot!(
        diversity_plt,
        plot_data.iterations,
        plot_data.mean_pairwise_hamming_distance;
        linewidth = 2.5,
        color = :firebrick3,
        label = "Mean Hamming distance",
    )

    collapse_plt = plot(
        plot_data.iterations,
        plot_data.global_best_fraction;
        linewidth = 2.7,
        color = :black,
        label = "Particles at global best",
        xlabel = "Iteration",
        ylabel = "Best fraction",
        legend = :bottomright,
        ylims = (-0.02, 1.02),
        grid = true,
        gridalpha = 0.22,
        background_color = :white,
        framestyle = :box,
        tickfontsize = 10,
        guidefontsize = 12,
    )

    return plot(
        fitness_plt,
        diversity_plt,
        collapse_plt;
        layout = grid(3, 1, heights = [0.42, 0.32, 0.26]),
        size = size,
        dpi = dpi,
        background_color = :white,
        link = :x,
    )
end

function save_swarm_trace_plot(landscape::Landscape,
                               result,
                               output_path::AbstractString;
                               title::AbstractString = "Swarm Trace",
                               fitness_label::AbstractString = "Penalized fitness",
                               size::Tuple{Int, Int} = (1400, 1000),
                               dpi::Int = 200)
    plt = plot_swarm_trace(
        landscape,
        result;
        title=title,
        fitness_label=fitness_label,
        size=size,
        dpi=dpi,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function swarm_animation_frame_title(title_prefix::AbstractString,
                                     generation::Integer,
                                     unique_subsets::Integer)
    return "$(title_prefix) (generation=$(generation), unique=$(unique_subsets))"
end

function append_animation_hold_frames!(anim::Plots.Animation, plt, hold_frames::Integer)
    hold = max(Int(hold_frames), 0)

    for _ in 1:hold
        Plots.frame(anim, plt)
    end

    return anim
end

function save_fitness_by_feature_count_swarm_animation(landscape::Landscape,
                                                       result,
                                                       output_path::AbstractString;
                                                       values = penalized_fitness_values(landscape, getproperty(result, :epsilon)),
                                                       title_prefix::AbstractString = "$(landscape.name) Fitness by Feature Count with Swarm",
                                                       fitness_label::AbstractString = "Penalized fitness",
                                                       fps::Int = 10,
                                                       final_hold_frames::Int = 0,
                                                       size::Tuple{Int, Int} = (1400, 900),
                                                       dpi::Int = 200)
    plot_data = swarm_trace_plot_data(landscape, result)
    anim = Plots.Animation()
    last_plot = nothing

    for (position, iteration) in pairs(plot_data.iterations)
        frame_title = swarm_animation_frame_title(title_prefix, iteration, plot_data.unique_subsets[position])
        plt = plot_fitness_by_feature_count_with_swarm(
            landscape,
            result;
            iteration=iteration,
            values=values,
            title=frame_title,
            fitness_label=fitness_label,
            size=size,
            dpi=dpi,
        )
        Plots.frame(anim, plt)
        last_plot = plt
    end

    isnothing(last_plot) || append_animation_hold_frames!(anim, last_plot, final_hold_frames)
    mkpath(dirname(output_path))
    Plots.gif(anim, output_path; fps=fps)
    return output_path
end

function save_hbm_swarm_animation(landscape::Landscape,
                                  result,
                                  output_path::AbstractString;
                                  values = penalized_fitness_values(landscape, getproperty(result, :epsilon)),
                                  title_prefix::AbstractString = "$(landscape.name) HBM with Swarm",
                                  fitness_label::AbstractString = "Penalized fitness",
                                  max_local_optima::Int = 50,
                                  fps::Int = 10,
                                  final_hold_frames::Int = 0,
                                  size::Tuple{Int, Int} = (2200, 1400),
                                  dpi::Int = 300)
    plot_data = swarm_trace_plot_data(landscape, result)
    anim = Plots.Animation()
    last_plot = nothing

    for (position, iteration) in pairs(plot_data.iterations)
        frame_title = swarm_animation_frame_title(title_prefix, iteration, plot_data.unique_subsets[position])
        plt = plot_hbm_with_swarm(
            landscape,
            result;
            iteration=iteration,
            values=values,
            title=frame_title,
            fitness_label=fitness_label,
            max_local_optima=max_local_optima,
            size=size,
            dpi=dpi,
        )
        Plots.frame(anim, plt)
        last_plot = plt
    end

    isnothing(last_plot) || append_animation_hold_frames!(anim, last_plot, final_hold_frames)
    mkpath(dirname(output_path))
    Plots.gif(anim, output_path; fps=fps)
    return output_path
end
