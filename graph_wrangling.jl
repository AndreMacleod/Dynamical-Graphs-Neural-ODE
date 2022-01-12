# Libraries
using CSV
using DataFrames
using LightGraphs
using MetaGraphs
using GraphDataFrameBridge
using DataFrames
using Random

cd("C:/Users/andre_viking96/Desktop/JuliaSciML")

# Loading dataframe
df = CSV.read("Video_games_5_clean.csv", DataFrame)

# Improving colname
rename!(df, :asin => :productID)

describe(df)

# Getting only columns we need to create graph
edge_csv = df[:, [:reviewerID, :productID, :reviewYear]]
edge_csv

# Create graph
mg = MetaGraph(edge_csv, :reviewerID, :productID, edge_attributes = [:reviewYear])
mg.graph


## Getting a smaller graph to work with for our first model
# use egonet, to make sure we have a connected graph to work with
Random.seed!(6);

egraph = egonet(mg, rand(vertices(mg)), 2)

# save this graph object for loading and drawing later on
savegraph("egraph", egraph.graph)

# saving this to upload to github and replicate with this (file size limits)
savegraph("egraph.mg", egraph)



## GIULIO REPLICATE FROM HERE
## Creating a 'time-series' of graphs to represent a dynamic graph over the same set of nodes

mg = loadgraph("egraph.mg", MGFormat())

# Function to filter edges by year attribute
function edge_filter_fn(g::AbstractMetaGraph, e, year)
    return get_prop(g, e, :reviewYear) == year
end

# Now for each year we have a set of edges for that year, and we create a list with all of these objects
year_edges = Vector{LightGraphs.SimpleGraphs.SimpleEdge{Int64}}[]
for year in 1999:2018
    push!(year_edges, collect(filter_edges(mg, (g, e) -> edge_filter_fn(g,e, year))))
end


# make a copy of egraph that we can remove all edges from
empty_graph = copy(mg)
# remove all edges from this graph
for edge in collect(edges(empty_graph))
    rem_edge!(empty_graph, edge)
end
empty_graph

# Crete empty array we fill with series of graphs
time_graphs = []

for edge_set in year_edges
    year_graph = copy(empty_graph)
    println(year_graph)
    for edge in collect(edge_set)
        add_edge!(year_graph, edge)
    end
    push!(time_graphs, year_graph)
end

time_graphs


## Doing the embedding
using LinearAlgebra
using Arpack

# function that does svd decomposition
function do_the_rdpg(A,d)
    L,Σ,R = svds(A; nsv=d)[1]
    L̂ = L * diagm(.√Σ)
    R̂ = R * diagm(.√Σ)
    return (L̂ = L̂, R̂ = R̂)
end

time_embeddings = Matrix{Float64}[]

for graph in time_graphs
    L, R = do_the_rdpg(adjacency_matrix(graph), 4)
    push!(time_embeddings, L)
end


time_embeddings[11]
length(time_embeddings)

# Testing with two of the graphs
A1 = adjacency_matrix(time_graphs[11])

# check it is a symmetric matrix
@assert A1 == A1'
L1, R1 = do_the_rdpg(A1, 4)

L1
R1

A2 = adjacency_matrix(time_graphs[8])
L2, R2 = do_the_rdpg(A2, 4)
