# Andre Macleod-Hungar DATA601project


project.jl is all the code for the project so far, with csv to graph embedding time series, and using these to train neural ODE. It loads in the aves file as the dataset.

Training of NN so far is extremely slow to run on my CPU - running loss(pinit) currently a percentage bar for rough time estimate - after 20mins, still 0.0%. I was way off in my initial time estimates. About half a mill NN params, with 202 vertices and 3 dimensions.
