using Random, Distributions, DataFrames, CSV
include("quantile_expectile.jl")

Random.seed!(0)

N = 400000

# generate probability P~(0,1)
a = rand(Uniform(1, 100), N)

# parameters: sigma_e
sigma_e = 0.7   

# calculates the inverse of the Cumulative Normal Distribution Function  
v = quantile(Normal(0, sigma_e), a)

# generate the inverse CDF (F_1(tau) & E_1(tau))
probs=collect(0.0001:0.0001:0.9999)
E_1 = expectile(x, probs)
Q_1 = quantile(x, probs)

Q_1_E_1 = DataFrame(hcat(Q_1, E_1));
CSV.write("Q_1_E_1.dat", Q_1_E_1, header = false)