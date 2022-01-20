# packages
using CSV
using DataFrames
using LightGraphs
using MetaGraphs
using GraphDataFrameBridge
using DataFrames
using Random


cd("C:/Users/andre_viking96/Desktop/New folder (2)/")

# Loading df
df = CSV.read("aves-wildbird-network.csv", DataFrame)

# Improve df column names
rename!(df,[:Bird1,:Bird2,:Weight, :Day])

df
describe(df)

# Create graph from dataframe
edge_df = df[:, [:Bird1, :Bird2, :Day]]
edge_df

full_mg = MetaGraph(edge_df, :Bird1, :Bird2, edge_attributes = [:Day])
full_mg.graph
typeof(full_mg)

# save this graph object for leading and drawing in different file
savegraph("birdgraph", full_mg.graph)


## We now need a time series of graphs, 6 - one per each day

# make a copy of ful graph that we can remove all edges from - this is the base with the entire vertex set
empty_graph = copy(full_mg)
# remove all edges from this graph
for edge in collect(edges(empty_graph))
    rem_edge!(empty_graph, edge)
end
empty_graph

# now, create separate dataframe for each time t, from whoich we will use th entries to fill the edges
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
        add_edge!(day_graph, df_list[i][j, :].Bird1, df_list[i][j, :].Bird2)
    end
    push!(time_graphs, day_graph)
end

time_graphs


# Testing - looks good
t5 = time_graphs[5]
collect(edges(t5))[150:180]
filter(:Bird2 => ==(4), df_list[5])

# use adjacency matrix function
A1 = adjacency_matrix(time_graphs[1])


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

# check it is diagonal matrix
@assert A1 == A1'

time_embeddings = Matrix{Float64}[]

# Create a list of embeddings that will contain our time series of graph embeddings
for graph in time_graphs
    L, R = do_the_rdpg(adjacency_matrix(graph), 3)
    push!(time_embeddings, L)
end

time_embeddings[1]
length(time_embeddings)

# Having a look at our L and R matrices
L1, R1 = do_the_rdpg(A1, 4)


## Solving neural ODE

using DifferentialEquations
using Plots
using DiffEqFlux
using OrdinaryDiffEq
using Flux
using Optim
using ProgressMeter
using ProgressLogging


# Create training set
train = time_embeddings[1:5]

# Define initial parameters
u0 = train[1]
datasize = length(train)
tspan = (1, 5)
tsteps = range(tspan[1], tspan[2], length = datasize)


# Create function do give neural net an input.
# In this case, the imput will be the distance from one vertex to each other vertex, for every vertex in the graph, repeated for every vertex

# Function that takes take vertex embedding, one vertex and returns all distances between that vertex and all other vertices
function vertex_distances(M::Matrix{Float64}, v::Int64)
    distances = []
    for i in 1:size(M)[1]-1              # iterating over all rows in matrix except the one selected
        for j in 1:size(M)[2]            # iterating over all columns in matrix
            push!(distances, M[v, j] - M[Not(v), :][i, j])
        end
    end
    return distances
end

vertex_distances(u0, 2)

# Defining dimension of matrix - n is n. of vertices, d is the dimension of each vertex
n = 202
d = 3

# create model architecture
ann = Chain(Dense((n-1)*d,640,tanh), Dense(640,3))
pinit,re = Flux.destructure(ann)

# check how many params in the nn - 388483
pinit

# create initial function
function dudt1_(du,u,p,t)
    for i in 1:n
        for j in 1:d
            du[(i-1)*d + j]= re(p)(vertex_distances(u, i))[j]
        end
    @info "foo-bar" progress=i/n  # this will show us that we are iterating through the matrix
    end
end

# Define problem
prob = ODEProblem(dudt1_, u0, tspan, pinit)

# Solving problem

# Reshaping train so it can be fitted aginst loss function
loss_train = Array{Float64}(undef,n,d,datasize)
train[1]
for i in 1:datasize
     loss_train[:,:,i] = train[i]
end

# flatening loss_train to conform with shape during nn training
loss_train = reshape(loss_train, (n*d,datasize))

# Create loss function
function loss(p)
    tmp_prob = remake(prob, p = p)
    tmp_sol = solve(tmp_prob, Tsit5(), saveat = tsteps, progress = true)
    sol_array = reshape(Array(tmp_sol), (n*d,datasize))
    println("TRAINING")
    println(loss_train)
    println("NNVALS")
    println(sol_array)
    sum(abs2, sol_array - loss_train)

end

# Testing loss function by finding loss of initial condition
loss(pinit)

# Neural callback function used for nn training
function neuralode_callback(p,l)
    @show l
    @show p
    tmp_prob = remake(prob, p = p)
    tmp_sol = solve(tmp_prob, Tsit5(), saveat = tsteps)
    fig = plot(tmp_sol)
    display(fig)
    false
end


#Use ADAM for finding initial local minima
@time res = DiffEqFlux.sciml_train(loss, pinit, ADAM(0.05), cb = neuralode_callback, maxiters = 5)

# res.minimzers gives best parameters from first search, now use these to continue with BFGS which finishes local minima better
# bgfs runs until finds a fit - finalizes optimization - bfgs good for stiff conditions
# bfgs looks for convergences criteria
res2 = DiffEqFlux.sciml_train(loss, res.minimizer, BFGS(initial_stepnorm=0.01), cb = neuralode_callback, maxiters = 5)

