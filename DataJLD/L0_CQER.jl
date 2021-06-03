using JuMP, Gurobi
include("toolbox.jl")


function L0_CQER(X, y, k, tau, bigM, method, fun, seed)
    ## Solving the CQER with L0 norm by ACP Algorithm ##

    n = size(X)[1]
    max_cuts_sparse = 0
    tol = 0.01
    max_iter_sparse = 1
    max_cuts = 1000
    final_gap = 0.1
    iter = 1

    UB_array = zeros(max_iter_sparse)
    LB_array = zeros(max_iter_sparse)
    runtime_LB = zeros(max_iter_sparse)
    S_opt = zeros(Int64,k)
    S_lb = zeros(Int64,k)
    gap = 2*final_gap
    
    theta_ub = []
    Xi_ub=[]
    z_lb = []
    rand_idxs = zeros(Int64,1,2)

    t1 = time() 
    while(gap > final_gap && iter <= max_iter_sparse)

        # Solve for lb.
        t3 = time()
        theta_lb, Xi_lb, z_lb, LB_array[iter],~, ~  = CQER_MIO(X, y, k, tau, max_cuts_sparse, tol, bigM, z_lb, theta_ub, Xi_ub, method, fun, seed)
        t4 = time()
        runtime_LB[iter] = t4-t3 
        S_lb = copy(findall(z_lb .>= 0.5))
        S_lb = sort(S_lb)
    
        # Find violated constraints.
        # Add them to the UB problem as the reduced master.
        if(max_iter_sparse > 0)
            viol_idxs =  zeros(Int64, n, 2)
            for i in 1:n 
                XiX_i = dot(vec(Xi_lb[i,:]),vec(X[i,:]));
                separation = (theta_lb[i]-XiX_i)*ones(n,1) - theta_lb + X*Xi_lb[i,:]

                if (fun == "cost")
                    # calculate infeasibility [Eq. (34)] in Bertsimas and Mundru (2020).
                    separation_positive = max.(separation, 0)
                    # find the maximum j(i) [Eq. (6)].
                    max_value = maximum(separation)
                    if(max_value > tol)
                        max_index = argmax(separation)[1]
                        viol_idxs[i,1] = copy(i);
                        viol_idxs[i,2] = copy(max_index);
                    end
                elseif(fun == "prod")
                    # calculate infeasibility
                    separation_positive = min.(separation, 0)
                    # find the minimum j(i)
                    max_value = minimum(separation_positive)
                    if(max_value < -tol)
                        max_index = argmin(separation)[1]
                        viol_idxs[i,1] = copy(i);
                        viol_idxs[i,2] = copy(max_index);
                    end
                end
            end
            rand_idxs = vcat(rand_idxs,viol_idxs)
        end
        
        # Polish S_lb solution to get correpsonding UB.
        theta_ub, Xi_ub_reduced,~,~,~,~,~, UB_array[iter], ~= CQER(X[:,S_lb], y, tau, max_cuts, tol, bigM, 1, method, fun, seed)
        Xi_ub = zeros(n,d);
        for i in 1:length(S_lb)
            Xi_ub[:,S_lb[i]] = Xi_ub_reduced[:,i];
        end
        # Pass them as warm-starts for lower bound problem.
        gap = (UB_array[iter] - LB_array[iter])/LB_array[iter]

        iter +=1
    end
    t2 = time()
    runtime = t2-t1
    S_opt = sort(S_lb)

    return theta_ub, Xi_ub, S_opt, runtime, gap
end