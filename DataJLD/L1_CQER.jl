using JuMP, Gurobi
include("toolbox.jl")


function L1_CQER(X, y, tau, eta, max_cuts, tol, method, fun, seed)

    # number of observations and variables.
    n = length(y)
    p = size(X,2)

    # Define the reduced master problem.
    m_outer = direct_model(Gurobi.Optimizer(GRB_ENV))
    MOI.set(m_outer, MOI.Silent(), true)
    
    # Define the variables
    @variable(m_outer, theta[1:n])
    @variable(m_outer, Xi[1:n,1:p] >=0 )
    @variable(m_outer, eminus[1:n] >=0 )
    @variable(m_outer, eplus[1:n] >=0 )
    
    # Solve the reduced problem.
    @constraint(m_outer, reg[i=1:n], y[i] - theta[i] == eplus[i] - eminus[i])

    full_violation_pair=[]
    numcon = n;
    r=1;
    for i in 1:numcon
        ind1 = i
        for j in 1:r
            ind2 = Distributions.sample(seed, setdiff(collect(1:n),i),1,replace=false)[1]
            if(ind2 > 0)
                if (fun == "prod")
                    @constraint(m_outer, theta[ind1] + sum((X[ind2,ind] - X[ind1,ind])*Xi[ind1,ind] for ind=1:p) >= theta[ind2])
                elseif (fun == "cost")
                    @constraint(m_outer, theta[ind1] + sum((X[ind2,ind] - X[ind1,ind])*Xi[ind1,ind] for ind=1:p) <= theta[ind2])
                end
                push!(full_violation_pair, [copy(ind1), copy(ind2)])
            end
        end
    end

    #println("Finished the constraints")
    if (method == "quantile")
        @objective(m_outer, Min, (1-tau)*sum(eminus[i] for i=1:n) + tau*sum(eplus[i] for i=1:n) + eta*sum(sum(Xi[i, j] for j=1:p) for i=1:n))
    elseif(method == "expectile")
        @objective(m_outer, Min, (1-tau)*sum((eminus[i])^2 for i=1:n) + tau*sum((eplus[i])^2 for i=1:n) + eta*sum(sum(Xi[i, j] for j=1:p) for i=1:n))
    end
    t1 = time()
    optimize!(m_outer)
    t2 = time()
    actual_time = t2 - t1
    #println("Solved the reduced master problem")
    theta_opt = value.(theta)
    Xi_opt = value.(Xi)
    eplus_opt = value.(eplus)
    eminus_opt = value.(eminus)

    cuts_added = 0
    opt_reached = 0
    separation = zeros(n)
    separation_positive = zeros(n)
    max_index = zeros(Int,n)

    initial_numconstr = copy(num_linear_constraints(m_outer)) 
    cuts_array = Array{Int}(undef, 0)
    Pinf_array = Array{Float64}(undef, 0)
    MaxV_array = Array{Float64}(undef, 0)
    alpha = zeros(n, 1) 

    # Adding cuts sequentially.
    while(cuts_added <= max_cuts-1 && opt_reached == 0)
        # Define the separation problem.
        # Sorting.
        # If no violation, then opt_reached == 1.
        # For each i, find a cut.

        MaxV = 0
        norm_infeasibility = 0
        violation_pair = []
        for i in 1:n
            XiX_i = dot(vec(Xi_opt[i,:]), vec(X[i,:]))
            separation = (theta_opt[i]-XiX_i)*ones(n,1) - theta_opt + X*Xi_opt[i,:]

            if (fun == "cost")
                # calculate infeasibility [Eq. (34)] in Bertsimas and Mundru (2020).
                separation_positive = max.(separation, 0)
                norm_infeasibility = norm_infeasibility + norm(separation_positive)^2
                # find the maximum j(i) [Eq. (6)].
                max_value = maximum(separation)
                if(max_value > tol)
                    max_index[i] = copy(argmax(separation)[1])
                    push!(violation_pair, [copy(i), copy(max_index[i])])
                else
                    max_index[i] = 0
                end
                if(max_value > MaxV)
                    MaxV = copy(max_value)
                end
            elseif(fun == "prod")
                # calculate infeasibility
                separation_positive = min.(separation,0)
                norm_infeasibility = norm_infeasibility + norm(separation_positive)^2
                # find the minimum j(i)
                max_value = minimum(separation_positive)
                if(max_value < -tol)
                    max_index[i] = copy(argmin(separation)[1])
                    push!(violation_pair, [copy(i), copy(max_index[i])])
                else
                    max_index[i] = 0
                end
                if(max_value < MaxV)
                    MaxV = copy(max_value)
                end
            end
        end

        norm_infeasibility = sqrt(norm_infeasibility)/n
        push!(Pinf_array, norm_infeasibility)
        push!(MaxV_array, MaxV)

        if(sum(max_index) == 0)
            opt_reached = 1
            # Else, add constraint, and re-solve.
        else
            # 1. Add one for each i.
            # 2. Add the j^*(i) for each i.
            # 3. j^*(i) = max_index[i].
            for i in 1:length(violation_pair)
                i_star = violation_pair[i][1]
                j_star = violation_pair[i][2]
                if (j_star > 0)
                    if (fun == "prod")
                        @constraint(m_outer, theta[i_star] + sum((X[j_star,ind] - X[i_star,ind])*Xi[i_star,ind] for ind=1:p) >= theta[j_star])
                    elseif (fun == "cost")
                        @constraint(m_outer, theta[i_star] + sum((X[j_star,ind] - X[i_star,ind])*Xi[i_star,ind] for ind=1:p) <= theta[j_star])
                    end
                end
            end

            # Update full_violation_pair.
            full_violation_pair = [full_violation_pair; violation_pair]

            # Set warm starts.
            set_start_value.(theta, theta_opt)
            set_start_value.(Xi, Xi_opt)
            set_start_value.(eplus, eplus_opt)
            set_start_value.(eminus, eminus_opt)
            if (method == "quantile")
                @objective(m_outer, Min, (1-tau)*sum(eminus[i] for i=1:n) + tau*sum(eplus[i] for i=1:n) + eta*sum(sum(Xi[i, j] for j=1:p) for i=1:n))
            elseif(method == "expectile")
                @objective(m_outer, Min, (1-tau)*sum((eminus[i])^2 for i=1:n) + tau*sum((eplus[i])^2 for i=1:n) + eta*sum(sum(Xi[i, j] for j=1:p) for i=1:n))
            end
            t1 = time()
            optimize!(m_outer)
            t2 = time()
            actual_time += t2 - t1

            theta_opt = value.(theta)
            Xi_opt = value.(Xi)
            eplus_opt = value.(eplus)
            eminus_opt = value.(eminus)

            # Loop counter.
            cuts_added = cuts_added + 1

            if (cuts_added == 1)
                push!(cuts_array,initial_numconstr)
                push!(cuts_array, num_linear_constraints(m_outer) - initial_numconstr)
            else
                sum_so_far = sum([cuts_array[j] for j in 1:cuts_added])
                push!(cuts_array, num_linear_constraints(m_outer) - sum_so_far)
            end
        end # if - else - violation found.
    end # cuts_added loop.
    #println("Cutting planes ends")
    #@show cuts_added
    #@show length(full_violation_pair)

    # calculate alpha
    for i = 1:n
        alpha[i] = theta_opt[i] - sum(Xi_opt[i, ind] * X[i, ind] for ind=1:p)
    end

    return theta_opt, Xi_opt, opt_reached, cuts_array, Pinf_array, MaxV_array, actual_time, getobjectivevalue(m_outer), alpha
end
