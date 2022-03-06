using CSV
using DataFrames
using MetaGraphs
using Graphs
using GraphDataFrameBridge
using DataFrames
using Random
using DifferentialEquations
using Plots
using DiffEqFlux
using OrdinaryDiffEq
using Flux
using Optim
using SymbolicRegression
using LinearAlgebra
using Arpack

# Function that takes the edge dataframe and the original edge dataframe and returns a dynamical graph in the form of an array of graphs
function get_dynamical_graph(edge_df)

    # create metagraph
    full_mg = MetaGraph(edge_df, :Bird1, :Bird2, edge_attributes = [:Day])

    full_mg.graph
    typeof(full_mg)

    edges

    # Make a copy of the full graph that we can remove all edges from - this is the base with the entire vertex set
    empty_graph = copy(full_mg)
    # remove all edges from this graph
    for edge in collect(Graphs.edges(empty_graph))
        Graphs.rem_edge!(empty_graph, edge)
    end
    empty_graph

    # Now, create separate dataframe for each time t, from whoich we will use the entries to fill the edges

    df_list = DataFrame[]
    for i in 1:6
        day_df = filter(:Day => ==(i), edge_df)
        println(typeof(day_df))
        push!(df_list,day_df)
    end
    df_list

    # Now we fill the edges in for each time graph, and store them in the time series array of graphs
    time_graphs = MetaGraph{Int64, Float64}[]
    for i in 1:6
        day_graph = copy(empty_graph)
        for j in 1:nrow(df_list[i])
            Graphs.add_edge!(day_graph, df_list[i][j, :].Bird1, df_list[i][j, :].Bird2)
        end
        push!(time_graphs, day_graph)
    end

    return time_graphs
end



# function that does svd decomposition
function do_the_rdpg(A,d)
    L,Σ,R = svds(A; nsv=d)[1]
    L̂ = L * diagm(.√Σ)
    R̂ = R * diagm(.√Σ)
    return (L̂ = L̂, R̂ = R̂)
end



# Create function do give neural net an input.

# Function that takes take vertex embedding, one vertex and returns all distances between that vertex and the k closest vertices
function vertex_distances(M, v, k)
    distances = []  # preallocate sizes
    vertex_norm_distances = []
    for i::Int8 in 1:n                                                 # find closest_vertices to v 
        if i != v
            distance = norm(M[:,v] - M[:,i], 1)
            push!(vertex_norm_distances, (distance, i))
      
        end 
    end

    closest_vertices_to_v = partialsort(vertex_norm_distances,1:k, by=x->x[1])

    for i in 1:length(closest_vertices_to_v)
        vertex_index = closest_vertices_to_v[i][2]           
        for j in 1:size(M)[1]                                          # iterating over all columns in matrix
            push!(distances, M[j, v] - M[j, vertex_index])
        end
    end
    return distances
end



function vertex_distances_2(M, v, k)
    distances = []  # preallocate sizes
    vertex_norm_distances = []
    for i::Int8 in 1:n                                                 # find closest_vertices to v 
        if i != v
            distance = norm(M[:,v] - M[:,i], 1)
            push!(vertex_norm_distances, (distance, i))
      
        end 
    end

    closest_vertices_to_v = partialsort(vertex_norm_distances,1:k, by=x->x[1])

    for i in 1:length(closest_vertices_to_v)
        vertex_index = closest_vertices_to_v[i][2]           
        for j in 1:size(M)[1]                                          # iterating over all columns in matrix
            push!(distances, M[j, v] - M[j, vertex_index])
        end
    end
    return distances
end

