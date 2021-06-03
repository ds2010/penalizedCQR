using Distributions, LinearAlgebra, Gurobi

include("DGP.jl")
include("CrossValidation.jl")
include("L0_CQER.jl")
include("findtau.jl")
const GRB_ENV = Gurobi.Env()

function L0_testing(n, d, k, SNR, tau, method, fun, seed)

    # input and output dataW
    X, y, x_true, epsilon, y_true = DGP(n, d, k, SNR, seed)

    # find the optimal k and bigM
    k_all = collect(1:1:(d-1))
    if method == "expectile"
        q_inv, tau_e = tau_expectile(epsilon, tau)
        bigM_all = [1.5, 2, 3, 4, 5, 6]
    elseif (method == "quantile")
        q_inv, ~ = tau_expectile(epsilon, tau)
        tau_e = tau
        bigM_all = [0.1, 0.5, 1, 1.5, 2, 3, 4, 5]
    end 

    MSE = zeros(length(k_all), length(bigM_all))
    for i in 1:length(k_all)
        for j in 1:length(bigM_all)
            MSE[i, j] = CV_L0norm(X, y, k_all[i], tau_e, bigM_all[j], method, fun, seed)[1]
        end
    end
    k = k_all[argmin(MSE)[1]]
    bigM = bigM_all[argmin(MSE)[2]]

    theta_ub, ~, s_opt, ~, ~ = L0_CQER(X, y, k, tau_e, bigM, method, fun, seed)

    risk= norm(theta_ub - y_true .- q_inv )^2/norm(y_true .+ q_inv)^2
    accuracy = length(intersect(s_opt, x_true))/k*100

    println(k)
    println(bigM)

    flush(stdout)
    
    return accuracy, risk

end
