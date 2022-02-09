using Pkg
Pkg.activate(".")


Pkg.instantiate()

# Attaching packages
using CSV
using DataFrames
using LightGraphs
using MetaGraphs
using GraphDataFrameBridge
using DataFrames
using Random


#cd("C:/Users/andre_viking96/Desktop/New folder (2)/")

# Loading df
df = CSV.read("aves-wildbird-network.csv", DataFrame)

# Improve df column names
rename!(df,[:Bird1,:Bird2,:Weight, :Day])

mini_df = filter([:Bird1, :Bird2] => (x, y) -> x < 9 && y < 9, df)

mini_df
describe(mini_df)



gdf = groupby(mini_df, [:Day])
combine(gdf, nrow)


# Create graph from dataframe
edge_df = mini_df[:, [:Bird1, :Bird2, :Day]]
edge_df

full_mg = MetaGraph(edge_df, :Bird1, :Bird2, edge_attributes = [:Day])

full_mg.graph
typeof(full_mg)

# save this graph object for leading and drawing in different file
savegraph("minibirdgraph", full_mg.graph)

## We now need a time series of graphs, 6 - one per each day

# Make a copy of the full graph that we can remove all edges from - this is the base with the entire vertex set
empty_graph = copy(full_mg)
# remove all edges from this graph
for edge in collect(edges(empty_graph))
    rem_edge!(empty_graph, edge)
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
        add_edge!(day_graph, df_list[i][j, :].Bird1, df_list[i][j, :].Bird2)
    end
    push!(time_graphs, day_graph)
end

time_graphs

props(empty_graph, 25)

## Testing to make sure vetrex labels match index for when we use add_edge()

a = 0
for i in 1:10
    if get_prop(empty_graph, i, :name) != i
        print("incorrect name")
        break
    else
        a += 1
    end        
end
print(a)

time_graphs[5]

## Doing the embedding

using LinearAlgebra
using Arpack

n = 8
d = 2

# function that does svd decomposition
function do_the_rdpg(A,d)
    L,Σ,R = svds(A; nsv=d)[1]
    L̂ = L * diagm(.√Σ)
    R̂ = R * diagm(.√Σ)
    return (L̂ = L̂, R̂ = R̂)
end


time_embeddings = Matrix{Float64}[]

# Create a list of embeddings that will contain our time series of graph embeddings
for graph in time_graphs
    L, R = do_the_rdpg(adjacency_matrix(graph), d)
    push!(time_embeddings, L)
end

time_embeddings[5]
length(time_embeddings)

# Having a look at our L and R matrices
#L1, R1 = do_the_rdpg(A1, 4)


## Solving neural ODE

using DifferentialEquations
using Plots
using DiffEqFlux
using OrdinaryDiffEq
using Flux
using Optim



# Create training set
train = time_embeddings[1:5]

# Define initial parameters
u0 = train[1]
datasize = length(train)
tspan = (1, 5)
tsteps = range(tspan[1], tspan[2], length = datasize)


# Function that returns 3 closest vertices by 1-norm to each vertex i in a matrix (each row in matrix corresponds to a vertex), as an array of tuples, where the first entry in the tuple is the target vertex

# #function closest_vertices(M::Matrix{Float64})
#     vertex_norm_distances = []
#     for i in 1:n
#         vertex_norm_distances = []
#         for j in 1:n
#             if i != j 
#                 distance = norm(M[i,:] - M[j,:], 1)
#                 push!(vertex_norm_distances, (distance, j))           
#             end 
#         end
#         sort!(vertex_norm_distances) 
#         closest_vertices_to_i = (i, vertex_norm_distances[1][2], vertex_norm_distances[2][2], vertex_norm_distances[3][2])
#         push!(closest_vertices_sets, closest_vertices_to_i)
#     end
#     return closest_vertices_sets
# #end

L1 = [2.0 2; 3 2; 1 1.4; 2 2; 2 1.9]

#x = closest_vertices(L1) 



# Create function do give neural net an input.
# In this case, the imput will be the distance from one vertex to each other vertex, for every vertex in the graph, repeated for every vertex

# Function that takes take vertex embedding, one vertex and returns all distances between that vertex and the 3 closest vertices
function vertex_distances(M, v)
    distances = []
    vertex_norm_distances = []
    for i::Int8 in 1:n                                                 # find closest_vertices to v 
        if i != v
            distance = norm(M[v,:] - M[i,:], 1)
            push!(vertex_norm_distances, (distance, i))           
        end 
    end
    sort!(vertex_norm_distances) 
    closest_vertices_to_v = [vertex_norm_distances[1][2]]
    for i in 1:length(closest_vertices_to_v)
        vertex_index = closest_vertices_to_v[i]           
        for j in 1:size(M)[2]                             # iterating over all columns in matrix
            push!(distances, M[v, j] - M[vertex_index, j])
        end
    end
    return distances
end

L1 = [2.0 2; 3 2; 1 1.4; 2 2; 2 1.9; 4 4; 4 4]
vertex_distances(u0, 2)


u0
# time series of matrices

L2 = [2.0 3; 4 1; 1 2;   2 2; 2 1.9]
L3 = [3.0 2; 4 3; 1 3.4; 2 2; 2 1.8]
L4 = [4.0 2; 4 3; 2 3;   2 2; 2 1.7]
L5 = [5.0 3; 3 1; 2 2;   2 2; 2 1.5]

test = [3.0 3; 4 2; 2 3; 2 2; 2 1]

norm(L1[1,:] - L1[2,:], 1)

L1[1,:]




L1
#u0 = train[1]
#datasize = length(train)
#tspan = (1, 5)
#tsteps = range(tspan[1], tspan[2], length = datasize)
#train = [L1,L2,L3,L4,L5]
# n=3 
# d=2





# create model architecture
ann = Chain(Dense((1)*d,4,tanh), Dense(4,d))
pinit,re = Flux.destructure(ann)

# check how many params in the nn - 388483
pinit



# time series of matrices
#L1 = [2.0 2; 3 2; 1 1]
#L2 = [2.0 3; 4 1; 1 2]
#L3 = [3.0 2; 4 3; 1 3]
#L4 = [4.0 2; 4 3; 2 3]
#L5 = [5.0 3; 3 1; 2 2]

#test = [3.0 3; 4 2; 2 3]

#u0 = train[1]
#datasize = length(train)
#tspan = (1, 5)
#tsteps = range(tspan[1], tspan[2], length = datasize)
#train = [L1,L2,L3,L4,L5]
# n=3 
# d=2

# create initial function
function dudt1_(du,u,p,t)
    for i in 1:n
        for j in 1:d
            du[(i-1)*d + j]= re(p)(vertex_distances(u, i))[j]  #put this outside of j loop
        end
    #@info "foo-bar" progress=i/n  # this will show us that we are iterating through the matrix
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
loss_train

# flatening loss_train to conform with shape during nn training
loss_train = reshape(loss_train, (n*d,datasize))

#plot(loss_train', linestyle = :dot, linealpha = 0.8, linewidth = 1.5)
# Create loss function
function loss(p)
    tmp_prob = remake(prob, p = p)
    tmp_sol = solve(tmp_prob, Tsit5(), saveat = tsteps)
    sol_array = reshape(Array(tmp_sol), (n*d,datasize))
    sum(abs2, sol_array - loss_train)

end

# Testing loss function by finding loss of initial condition
loss(pinit)

# Neural callback function used for nn training
function neuralode_callback(p,l)
    @show l
    tmp_prob = remake(prob, p = p)
    tmp_sol = solve(tmp_prob, Tsit5(), saveat = tsteps)
    plot(tmp_sol)
    data_plot = plot!(loss_train', linestyle = :dot, linealpha = 0.8, linewidth = 1.5)
    display(data_plot)
    false
end


#Use ADAM for finding initial local minima
res = DiffEqFlux.sciml_train(loss, pinit, ADAM(0.02), cb = neuralode_callback, maxiters = 10)

# res.minimzers gives best parameters from first search, now use these to continue with BFGS which finishes local minima better
# bgfs runs until finds a fit - finalizes optimization - bfgs good for stiff conditions
# bfgs looks for convergences criteria
res2 = DiffEqFlux.sciml_train(loss, res.minimizer, BFGS(initial_stepnorm=0.01), cb = neuralode_callback, maxiters = 10)

res2.minimum

pred_prob = ODEProblem(dudt1_, u0, tspan, res2.minimizer)
pred_data = solve(pred_prob, Tsit5(), saveat = tsteps)

reshape(Array(pred_data),  (n*d,datasize))
loss_train


using ModelingToolkit
using DataDrivenDiffEq
using OrdinaryDiffEq
using Sundials

X = (reshape(Array(pred_data),  (n*d,datasize)))
DX = (reshape(Array(pred_data(pred_data.t, Val{1})), (n*d,datasize)))

DX
X

# create basis
@variables t, u[1:16](t)
# Generate the basis functions, multivariate polynomials up to deg 5
# and sine
b = [polynomial_basis(u, 3); sin.(u)]
basis = Basis(b, u, iv = t)
# Create the thresholds which should be used in the search process
λ = Float32.(exp10.(-8:0.1:0))
# Create an optimizer for the SINDy problem
opt = STLSQ(λ)

# Define problem with data and differentiated data 
ddprob = ContinuousDataDrivenProblem(X, DX)

# find solution within basis
ddsol = solve(ddprob, basis, opt, normalize = true, maxiter = 10000)

system= result(ddsol)
print(system)

params = parameters(ddsol);

print(ddsol, Val{false})

equations(system)
print(system)
@named sys = ODESystem(equations(system), independent_variable(system), states(system), parameters(system))

dudt = ODEFunction(sys)

pred_tspan = (1,6)
pred_tsteps = range(pred_tspan[1], pred_tspan[2], length = datasize+1)


prob = ODEProblem(dudt, u0, pred_tspan, params)
pred_sol = solve(prob,  Vern7(), saveat = pred_tsteps, maxiters = 100000)

Array(pred_sol)

pred_embed1 = pred_sol[:, :, 6]

using JLD
save("pred_embed1.jld", "array", pred_embed)


pred = load("pred_embed.jld")["array"]

convert(DataFrame, pred')
pred_df = DataFrame(pred, :auto)

(eachrow(pred_df))

# Adding vertex number to column so we know wchich vertex it is
pred_df[!, :vertex] = 1:n

# reordering columns 
pred_df = pred_df[:, [:vertex, :x1, :x2]]

edge_prob_df = crossjoin(pred_df, pred_df, makeunique = true)

edge_prob_df = edge_prob_df[edge_prob_df[:, :vertex] .!= edge_prob_df[:, :vertex_1], :]

edge_prob_df[!, :edge_prob] = edge_prob_df[!, :x1] .* edge_prob_df[!, :x1_1] .+ edge_prob_df[!, :x2] .* edge_prob_df[!, :x2_1]

edge_prob_df

a = edge_prob_df[edge_prob_df[:,:edge_prob] .> 0.5, :]

print(a)

collect(edges(time_graphs[6]))
