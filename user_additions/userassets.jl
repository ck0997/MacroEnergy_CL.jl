struct Steelmaking <: AbstractAsset
    id::AssetId
    transform::Transformation
    crudesteel_edge::Union{Missing,AbstractEdge}
    dri_edge::Union{Missing,AbstractEdge}
    ironore_edge::Union{Missing,AbstractEdge}
    coal_edge::Union{Missing,AbstractEdge}
    natgas_edge::Union{Missing,AbstractEdge}
    hydrogen_edge::Union{Missing,AbstractEdge}
    steelscrap_edge::Union{Missing,AbstractEdge}
    elec_edge::Union{Missing,AbstractEdge}
    co2_edge::Union{Missing,AbstractEdge}
    co2_captured_edge::Union{Missing,AbstractEdge}
end

struct DRIMaking <: AbstractAsset
    id::AssetId
    transform::Transformation
    dri_edge::Union{Missing,AbstractEdge}
    ironore_edge::Union{Missing,AbstractEdge}
    coal_edge::Union{Missing,AbstractEdge}
    natgas_edge::Union{Missing,AbstractEdge}
    hydrogen_edge::Union{Missing,AbstractEdge}
    steelscrap_edge::Union{Missing,AbstractEdge}
    elec_edge::Union{Missing,AbstractEdge}
    co2_edge::Union{Missing,AbstractEdge}
    co2_captured_edge::Union{Missing,AbstractEdge}
end

function default_data(::Type{Steelmaking}, id=missing, style="full")
    return legacy_process_default_data(id, "CrudeSteel")
end

function default_data(::Type{DRIMaking}, id=missing, style="full")
    return legacy_process_default_data(id, "DRI")
end

function legacy_process_default_data(id, output_commodity)
    return Dict{Symbol,Any}(
        :id => id,
        :location => missing,
        :transforms => merge!(
            transform_default_data(),
            Dict{Symbol,Any}(
                :timedata => output_commodity,
                :constraints => Dict{Symbol,Bool}(:BalanceConstraint => true),
            ),
        ),
        :edges => Dict{Symbol,Any}(),
    )
end

function make(asset_type::Type{Steelmaking}, data::AbstractDict{Symbol,Any}, system::System)
    id, transform, edges = make_legacy_process(asset_type, data, system)
    return Steelmaking(
        id,
        transform,
        get(edges, :crudesteel_edge, missing),
        get(edges, :dri_edge, missing),
        get(edges, :ironore_edge, missing),
        get(edges, :coal_edge, missing),
        get(edges, :natgas_edge, missing),
        get(edges, :hydrogen_edge, missing),
        get(edges, :steelscrap_edge, missing),
        get(edges, :elec_edge, missing),
        get(edges, :co2_edge, missing),
        get(edges, :co2_captured_edge, missing),
    )
end

function make(asset_type::Type{DRIMaking}, data::AbstractDict{Symbol,Any}, system::System)
    id, transform, edges = make_legacy_process(asset_type, data, system)
    return DRIMaking(
        id,
        transform,
        get(edges, :dri_edge, missing),
        get(edges, :ironore_edge, missing),
        get(edges, :coal_edge, missing),
        get(edges, :natgas_edge, missing),
        get(edges, :hydrogen_edge, missing),
        get(edges, :steelscrap_edge, missing),
        get(edges, :elec_edge, missing),
        get(edges, :co2_edge, missing),
        get(edges, :co2_captured_edge, missing),
    )
end

function make_legacy_process(asset_type::Type{<:AbstractAsset}, data::AbstractDict{Symbol,Any}, system::System)
    id = AssetId(data[:id])
    location = as_symbol_or_missing(get(data, :location, missing))

    @setup_data(asset_type, data, id)

    transform_key = :transforms
    @process_data(
        transform_data,
        data[transform_key],
        [
            (data[transform_key], key),
            (data[transform_key], Symbol("transform_", key)),
            (data, Symbol("transform_", key)),
            (data, key),
        ],
    )
    transform = Transformation(;
        id = Symbol(id, "_", transform_key),
        timedata = system.time_data[Symbol(transform_data[:timedata])],
        location = location,
        constraints = get(transform_data, :constraints, [BalanceConstraint()]),
    )

    edges = Dict{Symbol,AbstractEdge}()
    for edge_key in keys(data[:edges])
        edge = make_legacy_process_edge(asset_type, edge_key, id, data, transform, system)
        if !ismissing(edge)
            edges[edge_key] = edge
        end
    end

    output_edge_key = legacy_output_edge_key(asset_type)
    output_edge = edges[output_edge_key]
    input_edge_keys = setdiff(
        collect(keys(edges)),
        [output_edge_key, :co2_edge, :co2_captured_edge],
    )

    transform.balance_data = Dict{Symbol,Dict{Symbol,Float64}}()
    for edge_key in input_edge_keys
        rate_key = legacy_consumption_key(edge_key)
        transform.balance_data[rate_key] = Dict(
            output_edge.id => get(transform_data, rate_key, 0.0),
            edges[edge_key].id => 1.0,
        )
    end

    if haskey(edges, :co2_edge)
        transform.balance_data[:emissions] = Dict(
            output_edge.id => get(transform_data, :emission_rate, 0.0),
            edges[:co2_edge].id => -1.0,
        )
    end

    if haskey(edges, :co2_captured_edge)
        transform.balance_data[:capture] = Dict(
            output_edge.id => get(transform_data, :capture_rate, 0.0),
            edges[:co2_captured_edge].id => -1.0,
        )
    end

    return id, transform, edges
end

legacy_output_edge_key(::Type{Steelmaking}) = :crudesteel_edge
legacy_output_edge_key(::Type{DRIMaking}) = :dri_edge

function legacy_consumption_key(edge_key::Symbol)
    edge_name = String(edge_key)
    return Symbol(replace(edge_name, r"_edge$" => "_consumption"))
end

function make_legacy_process_edge(
    asset_type::Type{<:AbstractAsset},
    edge_key::Symbol,
    asset_id::AssetId,
    data::AbstractDict{Symbol,Any},
    transform::Transformation,
    system::System,
)
    edge_data = recursive_merge(edge_default_data(), data[:edges][edge_key])
    edge_prefix = replace(String(edge_key), r"_edge$" => "")
    for key in keys(edge_default_data())
        loaded_value = get_from(
            [
                (data[:edges][edge_key], key),
                (data[:edges][edge_key], Symbol(edge_prefix, "_", key)),
                (data, Symbol(edge_prefix, "_", key)),
                (data, key),
            ],
            missing,
            false,
        )
        if !ismissing(loaded_value)
            edge_data[key] = loaded_value
        end
    end
    edge_data = process_data(edge_data)

    commodity_symbol = Symbol(edge_data[:commodity])
    commodity = commodity_types()[commodity_symbol]
    is_output = edge_key in (legacy_output_edge_key(asset_type), :co2_edge, :co2_captured_edge)

    if is_output
        start_vertex = transform
        vertex = get_from([(edge_data, :end_vertex), (data, :location)], missing, false)
        if ismissing(vertex)
            if edge_key in (:co2_edge, :co2_captured_edge)
                return missing
            end
            error("Missing end_vertex for required edge $(asset_id)_$(edge_key)")
        end
        edge_data[:end_vertex] = vertex
        end_vertex = find_node(system, Symbol(vertex), commodity)
    else
        vertex = get_from([(edge_data, :start_vertex), (data, :location)], missing, false)
        if ismissing(vertex)
            return missing
        end
        edge_data[:start_vertex] = vertex
        start_vertex = find_node(system, Symbol(vertex), commodity)
        end_vertex = transform
    end

    edge_constructor = get(edge_data, :unidirectional, true) ? Edge : BidirectionalEdge
    edge = edge_constructor(
        Symbol(asset_id, "_", edge_key),
        edge_data,
        system.time_data[commodity_symbol],
        commodity,
        start_vertex,
        end_vertex,
    )

    if edge_key in (:co2_edge, :co2_captured_edge)
        edge.constraints = Vector{AbstractTypeConstraint}()
    end

    return edge
end
