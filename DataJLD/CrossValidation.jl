using Distributions, Random, LinearAlgebra, ScikitLearn
include("L1_CQER.jl")
include("L0_CQER.jl")


function CV_L1norm(X, y, tau, eta, max_cuts, tol, method, fun, seed)
    # compute the MSE for selecting the turning parameter eta

    n = size(X)[1]
    d = size(X)[2]
    n_fold = 5  

    # standardize data X, y
    y_mean = mean(y, dims=1)
    y = y - repeat(y_mean, n, 1)
    y = y/norm(y, 2)
    
    X_mean = zeros(d)
    for i in 1:d
      X_mean[i] = sum([X[j,i] for j in 1:n])/n
    end
    X = X - ones(n,1)*X_mean'
    
    for i in 1:d
      X[:,i] = X[:,i]/norm(X[:,i],2)
    end

    # resample the data
    kfold = CrossValidation.KFold(n, n_folds=n_fold, random_state=1, shuffle=true)

    theta_array = zeros(Int(n-n/n_fold), n_fold) 
    beta_array = zeros(Int(n-n/n_fold), d, n_fold) 
    fhat_temp = zeros(Int(n/n_fold), Int(n-n/n_fold))
    fhat = zeros(Int(n/n_fold), n_fold)
    MSE = zeros(n_fold)

    for kk in 1:n_fold

        # estimate the L1 norm model
        theta_array[:, kk], beta_array[:, :, kk], ~, ~, ~, ~, ~, ~, ~ = L1_CQER(X[kfold[kk][1], :], y[kfold[kk][1], :], tau, eta, max_cuts, tol, method, fun, seed)

        for i in 1:Int(n/n_fold)
            for j in 1:Int(n-n/n_fold)
                # calcute the yhat for testing obs. using the Eq. (47) Chen et al. 2020
                fhat_temp[i, j] =  theta_array[j, kk] + (X[kfold[kk][2], :][i, :] - X[kfold[kk][1], :][j, :])' * beta_array[j, :, kk]
            end
        end

        if (fun == "prod")
            fhat[:, kk] = minimum(fhat_temp, dims=2)
        elseif (fun == "cost")
            fhat[:, kk] = maximum(fhat_temp, dims=2) 
        end

        # calculate MSE for testing data set 
        MSE[kk] = norm(fhat[:, kk] - y[kfold[kk][2], :])^2/Int(n/n_fold)
    end

    return mean(MSE, dims=1)

end



function CV_L0norm(X, y, k, tau, bigM, method, fun, seed)
    # compute the MSE for selecting the turning parameter k and bigM

    n = size(X)[1]
    d = size(X)[2]
    n_fold = 5    

    # standardize data X, y
    y_mean = mean(y, dims=1)
    y = y - repeat(y_mean, n, 1)
    y = y/norm(y, 2)
    
    X_mean = zeros(d)
    for i in 1:d
      X_mean[i] = sum([X[j,i] for j in 1:n])/n
    end
    X = X - ones(n,1)*X_mean'
    
    for i in 1:d
      X[:,i] = X[:,i]/norm(X[:,i],2)
    end

    # resample the data
    kfold = CrossValidation.KFold(n, n_folds=n_fold, random_state=1, shuffle=true)

    theta_array = zeros(Int(n-n/n_fold), n_fold) 
    beta_array = zeros(Int(n-n/n_fold), d, n_fold) 
    fhat_temp = zeros(Int(n/n_fold), Int(n-n/n_fold))
    fhat = zeros(Int(n/n_fold), n_fold)
    MSE = zeros(n_fold)

    for kk in 1:n_fold

        # estimate the L1 norm model
        theta_array[:, kk], beta_array[:, :, kk], ~, ~ = L0_CQER(X[kfold[kk][1], :], y[kfold[kk][1], :], k, tau, bigM, method, fun, seed)

        for i in 1:Int(n/n_fold)
            for j in 1:Int(n-n/n_fold)
                # calcute the yhat for testing obs. using the Eq. (47) Chen et al. 2020
                fhat_temp[i, j] =  theta_array[j, kk] + (X[kfold[kk][2], :][i, :] - X[kfold[kk][1], :][j, :])' * beta_array[j, :, kk]
            end
        end

        if (fun == "prod")
            fhat[:, kk] = minimum(fhat_temp, dims=2)
        elseif (fun == "cost")
            fhat[:, kk] = maximum(fhat_temp, dims=2) 
        end

        # calculate MSE for testing data set 
        MSE[kk] = norm(fhat[:, kk] - y[kfold[kk][2], :])^2/Int(n/n_fold)
    end

    return mean(MSE, dims=1)

end