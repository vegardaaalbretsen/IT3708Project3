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
