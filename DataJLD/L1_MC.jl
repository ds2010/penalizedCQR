using Distributions, LinearAlgebra, Gurobi

include("DGP.jl")
include("CrossValidation.jl")
include("L1_CQER.jl")
include("findtau.jl")
const GRB_ENV = Gurobi.Env()

function  L1_testing(n, d, k, SNR, tau, method, fun, seed)
    
    # input and output data
    X, y, x_true, epsilon, y_true = DGP(n, d, k, SNR, seed)

    if method == "expectile"
        q_inv, tau_e = tau_expectile(epsilon, tau)
    elseif (method == "quantile")
        q_inv, ~ = tau_expectile(epsilon, tau)
        tau_e = tau
    end 

    tol = 0.01
    max_cuts = 100
    # find the optimal eta
    if method == "expectile"
        eta_all = range(0.1, stop = 3.5, length = 100)
    elseif (method == "quantile")
        eta_all = range(0.1, stop = 1.3, length = 100)
    end 
    MSE = zeros(length(eta_all), 1)
    for i in 1:length(eta_all)
        MSE[i, :] = CV_L1norm(X, y, tau_e, eta_all[i], max_cuts, tol, method, fun, seed)
    end
    eta = eta_all[argmin(MSE)[1]]

    theta_opt, Xi_opt, ~, ~, ~, ~, ~, ~, ~ = L1_CQER(X, y, tau_e, eta, max_cuts, tol, method, fun, seed)

    risk= norm(theta_opt - y_true .- q_inv )^2/norm(y_true .+ q_inv)^2

    s_xi = zeros(d)
    for i in 1:d
        if count(x->x==0, round.(Xi_opt[:, i]; digits=4)) == n
            s_xi[i] = 1
        else 
            s_xi[i] = 0
        end
    end
    
    s_opt = copy(findall(s_xi .>= 0.5))
    s_opt = sort(s_opt)
    accuracy = length(intersect(s_opt, x_true))/k*100
    
    println(eta)
    flush(stdout)
    return accuracy, risk

end