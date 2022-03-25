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

# Define some parameters
# Number of vertices we take from graph - n
n = 10
# Dimension of svd - d
d = 2
# number of closest vertices for nn input function - k
k = 3


# Loading df
df = CSV.read("aves-wildbird-network.csv", DataFrame)

# Improve df column names
rename!(df,[:Bird1,:Bird2,:Weight, :Day])

# create smaller graph with on 10 vertices
mini_df = filter([:Bird1, :Bird2] => (x, y) -> x <= n && y <= n, df)
describe(mini_df)


# Create graph from dataframe
edge_df = mini_df[:, [:Bird1, :Bird2, :Day]]
edge_df

# Get time series of graphs 
time_graphs = get_dynamical_graph(edge_df)


# Convert this to time series of embeddings

time_embeddings = Matrix{Float64}[]

for graph in time_graphs
    L, R = do_the_rdpg(adjacency_matrix(graph), d)
    push!(time_embeddings, L)
end




## Solving neural ODE


# Create training set
train = time_embeddings[1:5]

# Define initial parameters
# We transpose matrices as julia deals with reshaping column-wise
u0 = train[1]'
datasize = length(train)
tspan = (1, 5)
tsteps = range(tspan[1], tspan[2], length = datasize)


# Create model architecture
ann = Chain(Dense((k)*d,8,tanh), Dense(8,d))
pinit,re = Flux.destructure(ann)

# check how many params in the nn - 388483
pinit

# ODE function
function dudt1_(du,u,p,t)
    for i in 1:n
        nn = re(p)(vertex_distances(u, i, k))
        for j in 1:d
            du[(i-1)*d + j] = nn[j]  
        end
    end
end


# Defining problem
prob = ODEProblem(dudt1_, u0, tspan, pinit)

# Reshaping train so it can be fitted aginst loss function
loss_train = Array{Float64}(undef,d,n,datasize)
train[2]
for i in 1:datasize
     loss_train[:,:,i] = train[i]'
end
loss_train


# Flattening loss_train to conform with shape during nn training
loss_train = transpose(reshape(loss_train, (n*d,datasize)))'

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
    display(data_plot)
    false
end

# Training NN
# Use ADAM for finding initial local minima
res = DiffEqFlux.sciml_train(loss, pinit, ADAM(0.03), cb = neuralode_callback, maxiters = 75)

# res.minimzers gives best parameters from first search, now use these to continue with BFGS which finishes local minima better
res2 = DiffEqFlux.sciml_train(loss, res.minimizer, BFGS(initial_stepnorm=0.01), cb = neuralode_callback, maxiters = 75)


# EquationSearch to find equations that fit NN data

# random inputs X
X = rand(k*d, 10)

# Y as output of nn
Y = re(par)(X)

# Separate y into d (in this case manually d=2) different vectors to find equations separately
y1 = Y[1,:]
y2 = Y[2,:]

options = SymbolicRegression.Options(
    binary_operators=(+, *, /, -),
    unary_operators=(cos, sin, exp),
    npopulations=20
)

# y1 EquationSearch
hof1 = EquationSearch(X, y1, niterations=5, options=options)
dominating = calculateParetoFrontier(X, y1, hof1, options)
eqn = node_to_symbolic(dominating[end].tree, options)
println("Complexity\tMSE\tEquation")

for member in dominating
    size = countNodes(member.tree)
    score = member.score
    string = stringTree(member.tree, options)

    println("$(size)\t$(score)\t$(string)")
end

# y2 EquationSearch
hof2 = EquationSearch(X, y2, niterations=5, options=options)
dominating = calculateParetoFrontier(X, y2, hof2, options)
eqn = node_to_symbolic(dominating[end].tree, options)

println("Complexity\tMSE\tEquation")

for member in dominating
    size = countNodes(member.tree)
    score = member.score
    string = stringTree(member.tree, options)
    println("$(size)\t$(score)\t$(string)")
end
