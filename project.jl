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

# Reshaping train so it can be fitted aginst loss function
loss_train = Array{Float64}(undef,d,n,datasize)
for i in 1:datasize
     loss_train[:,:,i] = train[i]'
end
loss_train

# Flattening loss_train to conform with shape during nn training
loss_train = transpose(reshape(loss_train, (n*d,datasize)))'



# Create model architecture
ann = Chain(Dense(d,8,tanh), Dense(8,d))
pinit,re = Flux.destructure(ann)

# check how many params in the nn - 388483
pinit
u0


nv = nearest_vertices(u0, 2, 3)[1][2]
for i in 1:k
    ab = vertex_distances(u0, 2, nv[i][2])
end

vertex_distances(u0, 2, nv[3][2])

vertex_distances_1(u0, 2, 3)


# Different way of doing ODE function - first way slower but appears more accurate

# function 1
function dudt1_(du,u,p,t)
    for i in 1:n
        nn = re(p)(vertex_distances(u, i, k))
        for j in 1:d
            du[(i-1)*d + j] = nn[j]  
        end
    end
end


# function 2 - can think of 'convolutional'
function dudt2_(du,u,p,t)

    for i in 1:n
        dv_i = [0.0,0.0]
        nearest_vertices_indices = nearest_vertices(u, i, k)
        for j in 1:k
            nv_index = nearest_vertices_indices[j][2]
            dv_i .+= re(p)(vertex_distances(u, i, nv_index))
        end
        du[:,i] = dv_i'
    end

end




# Defining problem
prob = ODEProblem(dudt2_, u0, tspan, pinit)



# Create loss function
function loss(p)
    tmp_prob = remake(prob, p = p)
    tmp_sol = solve(tmp_prob, Tsit5(), saveat = tsteps)
    sol_array = reshape(Array(tmp_sol), (n*d,datasize))
    sum(abs2, sol_array - loss_train)
    #print(sol_array)
    #print(size(sol_array))
end

# Testing loss function by finding loss of initial condition
loss(pinit)

# Neural callback function used for nn training

function neuralode_callback(p,l)
    @show l
    tmp_prob = remake(prob, p = p)
    tmp_sol = solve(tmp_prob, Tsit5(), saveat = tsteps)
    display(plot(tmp_sol))
    false
end

# Training NN
# Use ADAM for finding initial local minima
res = DiffEqFlux.sciml_train(loss, pinit, ADAM(0.02), cb = neuralode_callback, maxiters = 40)



## EquationSearch to find equations that fit NN data

# random inputs X
X = rand(d, 8)

# Y as output of nn
par = res2.minimizer
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
dominating1 = calculateParetoFrontier(X, y1, hof1, options)
eqn = node_to_symbolic(dominating1[end].tree, options)
println("Complexity\tMSE\tEquation")

for member in dominating1
    size = countNodes(member.tree)
    score = member.score
    string = stringTree(member.tree, options)

    println("$(size)\t$(score)\t$(string)")
end


#15      0.003311639176085496    ((sin(x2) / (-0.2227361 - (x1 + exp(sin(x1))))) - (sin(x1) * 0.9973934))

# y2 EquationSearch
hof2 = EquationSearch(X, y2, niterations=5, options=options)
dominating2 = calculateParetoFrontier(X, y2, hof2, options)
eqn = node_to_symbolic(dominating2[end].tree, options)

println("Complexity\tMSE\tEquation")

for member in dominating2
    size = countNodes(member.tree)
    score = member.score
    string = stringTree(member.tree, options)
    println("$(size)\t$(score)\t$(string)")
end

#19      0.0028572787857697474   ((x1 + x2) / (-1.5959082 - (x1 + ((cos(x1 - (x2 - 0.7570081)) * 0.07968318) / sin(x2)))))


function dudt_(du,u,p,t)
    for i in 1:n
        nearest_vertices_indices = nearest_vertices(u, i, k)

        for j in 1:k
            nv_index = nearest_vertices_indices[j][2]
            x1 =  vertex_distances(u, i, nv_index)[1]
            x2 =  vertex_distances(u, i, nv_index)[2]

            du[1,i] += ((sin(x2) / (-0.2227361 - (x1 + exp(sin(x1))))) - (sin(x1) * 0.9973934))
            du[2,i] += ((x1 + x2) / (-1.5959082 - (x1 + ((cos(x1 - (x2 - 0.7570081)) * 0.07968318) / sin(x2)))))
        end
    end
end


# solve ODE with best params from training
prob = ODEProblem(dudt2_, u0, tspan, res2.minimizer)

sol = solve(prob, Tsit5(), saveat = tsteps)
sol[1]




# Trying to find equations that fit NN data with SINDy technique - not working yet
using DataDrivenDiffEq
dd_prob = ContinuousDataDrivenProblem(sol)
@parameters t
@variables u[1:3](t)
Ψ = Basis([u; u[1]*u[2]], u, independent_variable = t)

res_sindy = solve(dd_prob, Ψ, STLSQ(),digits=1)
sys_sindy = result(res_sindy);


