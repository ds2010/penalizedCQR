# The first version of Cutting-plane Algorithm was created by Bertsimas and Mundru (2020).  
# What is presented here is a heavily adapted version to estiamte large scale CQR/CER models.
#
# Sheng Dai (sheng.dai@aalto.fi)
# Aalto University School of Business, Finland
# Feb 21, 2021


using JuMP, Gurobi


function CQER_full(X, y, tau, method, fun)
        # compute the full CQR/CER model 

        n = size(y,1)
        p = size(X,2)
    
        # Define the model
        model = direct_model(Gurobi.Optimizer(GRB_ENV))
        MOI.set(model, MOI.Silent(), true)

        # Define the variables
        @variable(model, theta[1:n])
        @variable(model, Xi[1:n,1:p] >= 0 )
        @variable(model, eminus[1:n] >= 0 )
        @variable(model, eplus[1:n] >= 0 )
    
        @constraint(model, reg[i=1:n], y[i] - theta[i] == eplus[i] - eminus[i])

        for i in 1:n
            for j in 1:n
                if( j != i)
                    if (fun == "prod")
                        @constraint(model, theta[i] + sum((X[j,ind] - X[i,ind])*Xi[i,ind] for ind=1:p) >= theta[j])
                    elseif (fun == "cost")
                        @constraint(model, theta[i] + sum((X[j,ind] - X[i,ind])*Xi[i,ind] for ind=1:p) <= theta[j])
                    end
                end
            end
        end
    
        if (method == "quantile")
            @objective(model, Min, (1-tau)*sum(eminus[i] for i=1:n) + tau*sum(eplus[i] for i=1:n) )
        elseif(method == "expectile")
            @objective(model, Min, (1-tau)*sum((eminus[i])^2 for i=1:n) + tau*sum((eplus[i])^2 for i=1:n) )
        end
        t1 = time()
        optimize!(model)
        t2 = time()-t1
    
        # calculate residual values
        theta_opt = value.(theta)
        beta = value.(Xi)
        residual_plus = value.(eplus)
        residual_minus = value.(eminus)

        # calculate alpha values
        alpha = zeros(n, 1)
            for i = 1:n,
                alpha[i] = theta_opt[i] - sum(beta[i, ind] * X[i, ind] for ind=1:p)
            end
        return theta_opt, alpha, beta, residual_plus, residual_minus, t2
    
end



function CQER(X, y, tau, max_cuts, tol, bigM, bounds, method, fun, seed)
    ## Solving the CQR/CER by the adapted cutting-plane Algorithm ##
    ## Using the SP to form the initial constraints ##

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
    
    if(bounds == 1)
        for i in 1:p 
            @constraint(m_outer, Xi[:,i] .<= bigM)
            @constraint(m_outer, Xi[:,i] .>= -bigM)
        end
    end

    @constraint(m_outer, reg[i=1:n], y[i] - theta[i] == eplus[i] - eminus[i])

    # Solve the reduced problem.
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
        @objective(m_outer, Min, (1-tau)*sum(eminus[i] for i=1:n) + tau*sum(eplus[i] for i=1:n) )
    elseif(method == "expectile")
        @objective(m_outer, Min, (1-tau)*sum((eminus[i])^2 for i=1:n) + tau*sum((eplus[i])^2 for i=1:n) )
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
                separation_positive = min.(separation, 0)
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
                @objective(m_outer, Min, (1-tau)*sum(eminus[i] for i=1:n) + tau*sum(eplus[i] for i=1:n) )
            elseif(method == "expectile")
                @objective(m_outer, Min, (1-tau)*sum((eminus[i])^2 for i=1:n) + tau*sum((eplus[i])^2 for i=1:n) )
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

    return theta_opt, Xi_opt, opt_reached, cuts_array, Pinf_array, MaxV_array, actual_time, getobjectivevalue(m_outer), length(full_violation_pair)
end


function CQER_NEW(X, y, tau, max_cuts, tol, idxs, bigM, bounds, method, fun)
    ## Solving the CQR/CER by the adapted cutting-plane Algorithm ##
    ## CQER_NEW uses the external initial constraint pair by argument indxs ##
    ## if use the SP to form the inital constaints, the results in CQER_NEW are equal to those in CQER ##

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
    
    if(bounds == 1)
        for i in 1:p 
            @constraint(m_outer, Xi[:,i] .<= bigM)
            @constraint(m_outer, Xi[:,i] .>= -bigM)
        end
    end

    @constraint(m_outer, reg[i=1:n], y[i] - theta[i] == eplus[i] - eminus[i])

    # Solve the reduced problem.
    full_violation_pair=[]
    r = size(idxs,2)-1
    numcon = size(idxs,1)
    for i in 1:numcon
        ind1 = idxs[i,1]
        for j in 1:r
            ind2 = idxs[i,j+1]
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
        @objective(m_outer, Min, (1-tau)*sum(eminus[i] for i=1:n) + tau*sum(eplus[i] for i=1:n) )
    elseif(method == "expectile")
        @objective(m_outer, Min, (1-tau)*sum((eminus[i])^2 for i=1:n) + tau*sum((eplus[i])^2 for i=1:n) )
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
                separation_positive = min.(separation, 0)
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
                @objective(m_outer, Min, (1-tau)*sum(eminus[i] for i=1:n) + tau*sum(eplus[i] for i=1:n) )
            elseif(method == "expectile")
                @objective(m_outer, Min, (1-tau)*sum((eminus[i])^2 for i=1:n) + tau*sum((eplus[i])^2 for i=1:n) )
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

    return theta_opt, Xi_opt, opt_reached, cuts_array, Pinf_array, MaxV_array, actual_time, getobjectivevalue(m_outer)
end


function CQER_MIO(X, y, k, tau, max_cuts, tol, bigM, z_0, theta_0, Xi_0, method, fun, seed)
    ## Solving the CQR/CER with L0 norm by the adapted cutting-plane Algorithm ##
    ## Using the SP to form the initial constraints ##

    n = length(y)
    p = size(X,2)
    z_opt_path = [];
    
    # Define the reduced master problem.
    m_outer = direct_model(Gurobi.Optimizer(GRB_ENV))
    MOI.set(m_outer, MOI.Silent(), true)

    @variable(m_outer,z[1:p],Bin)
    @variable(m_outer, theta[1:n])
    @variable(m_outer, Xi[1:n,1:p] >=0 )
    @variable(m_outer, eminus[1:n] >=0 )
    @variable(m_outer, eplus[1:n] >=0 )
    
    @constraint(m_outer,UB_constr[i=1:p,j=1:n], Xi[j,i] <= bigM*z[i]);
    @constraint(m_outer,LB_constr[i=1:p,j=1:n], Xi[j,i] >= -bigM*z[i]);
    
    @constraint(m_outer, Cardinality_constr, sum(z[i] for i in 1:p) <= k)

    @constraint(m_outer, reg[i=1:n], y[i] - theta[i] == eplus[i] - eminus[i])

    # Solve the reduced problem.
    #println("Adding the constraints")
    idxs = zeros(Int64,n,2);
    for i in 1:n
        ind2 = Distributions.sample(seed, setdiff(collect(1:n),i), 1, replace=false)[1];
        if (fun == "prod")
            @constraint(m_outer, theta[i] + sum((X[ind2,j] - X[i,j])*Xi[i,j] for j =1:p) >= theta[ind2])
        elseif (fun == "cost")
            @constraint(m_outer, theta[i] + sum((X[ind2,j] - X[i,j])*Xi[i,j] for j =1:p) <= theta[ind2])
        end
        idxs[i,1] = copy(i);
        idxs[i,2] = copy(ind2);
    end
    
    if (method == "quantile")
        @objective(m_outer, Min, (1-tau)*sum(eminus[i] for i=1:n) + tau*sum(eplus[i] for i=1:n) )
    elseif(method == "expectile")
        @objective(m_outer, Min, (1-tau)*sum((eminus[i])^2 for i=1:n) + tau*sum((eplus[i])^2 for i=1:n) )
    end

    if(length(z_0) > 0)
        set_start_value.(z, z_0)
        set_start_value.(theta, theta_0)
        set_start_value.(Xi, Xi_0)
    end
    #println("Solving the model")
    optimize!(m_outer)
    
    theta_opt = value.(theta)
    Xi_opt = value.(Xi)
    eplus_opt = value.(eplus)
    eminus_opt = value.(eminus)
    z_opt = value.(z)
    push!(z_opt_path, sort(findall(z_opt .> 0.5)))
    #@show findall(z_opt .== 1)

    cuts_added = 0;
    opt_reached = 0;

    separation = zeros(n);
    separation_positive = zeros(n);
    max_index = zeros(Int,n)
    new_idxs = zeros(Int64,n,2)

    # Adding cuts sequentially.
    while(cuts_added <= max_cuts-1 && opt_reached == 0)
        # Define the separation problem.
        # Sorting.
        # If no violation, then opt_reached == 1.
        # For each i, find a cut.
        violation_pairs=[];
        for i in 1:n
            XiX_i = dot(vec(Xi_opt[i,:]), vec(X[i,:]))
            separation = (theta_opt[i]-XiX_i)*ones(n,1) - theta_opt + X*Xi_opt[i,:]

            if (fun == "cost")
                # calculate infeasibility [Eq. (34)] in Bertsimas and Mundru (2020).
                separation_positive = max.(separation, 0)
                # find the maximum j(i) [Eq. (6)].
                max_value = maximum(separation)
                if(max_value > tol)
                    max_index[i] = copy(argmax(separation)[1])
                    push!(violation_pair, [copy(i), copy(max_index[i])])
                else
                    max_index[i] = 0
                end
            elseif(fun == "prod")
                # calculate infeasibility
                separation_positive = min.(separation, 0)
                norm_infeasibility = norm_infeasibility + norm(separation_positive)^2
                # find the minimum j(i)
                max_value = minimum(separation_positive)
                if(max_value < -tol)
                    max_index[i] = copy(argmin(separation)[1])
                    push!(violation_pair, [copy(i), copy(max_index[i])])
                else
                    max_index[i] = 0
                end
            end
        end

        if(sum(max_index) == 0)
            opt_reached = 1;
            #print("Cutting plane converges \n")
            # Else, add constraint, and re-solve.
        else
            # 1. Add one for each i.
            # 2. Add the j^*(i) for each i.
            # 3. j^*(i) = max_index[i].
            numcon = length(findall(max_index .> 0))
            new_idxs = zeros(Int64,numcon,2)
            numcon_count = 1
            for i in 1:length(violation_pairs)
                i_star = violation_pairs[i][1];
                j_star = violation_pairs[i][2];
                if(j_star > 0 && i_star > 0 )
                    if (fun == "prod")
                        @constraint(m_outer, theta[i_star] + sum((X[j_star,ind] - X[i_star,ind])*Xi[i_star,ind] for ind =1:p) >= theta[j_star] )
                    elseif (fun == "cost")
                        @constraint(m_outer, theta[i_star] + sum((X[j_star,ind] - X[i_star,ind])*Xi[i_star,ind] for ind =1:p) <= theta[j_star] )
                    end
                    new_idxs[numcon_count,1] = copy(i_star) 
                    new_idxs[numcon_count,2] = copy(j_star)
                    numcon_count += 1
                end
            end
    
            # Given current support, polish the feasible warm starts.
            # Solve using Gurobi with the current constraints.
            # Use CQER_NEW
            S_current = findall(z_opt .>= 0.5)
            #@show S_current
            # Construct new_idxs
            # Solve with max_cuts = 0 -> only the current problem with the added constraints so far.
            idxs = vcat(idxs, new_idxs)
            theta_opt, Xi_opt_1,~, ~, ~,~,~,~ =  CQER_NEW(X[:,S_current], y, tau, 0, tol, idxs, bigM, 1, method, fun)
            Xi_opt = zeros(n,d)
            count_ind = 1
            for i in 1:length(S_current)
                Xi_opt[:,S_current[i]] = Xi_opt_1[:,i]
            end
        
            # Set warm starts.
            set_start_value.(theta, theta_opt)
            set_start_value.(Xi, Xi_opt)
            set_start_value.(z, z_opt)
            set_start_value.(eplus, eplus_opt)
            set_start_value.(eminus, eminus_opt)
        
            # Re-solve the problem.
            if (method == "quantile")
                @objective(m_outer, Min, (1-tau)*sum(eminus[i] for i=1:n) + tau*sum(eplus[i] for i=1:n) )
            elseif(method == "expectile")
                @objective(m_outer, Min, (1-tau)*sum((eminus[i])^2 for i=1:n) + tau*sum((eplus[i])^2 for i=1:n) )
            end
            optimize!(m_outer)

            push!(z_opt_path, sort(findall(z_opt .> 0.5)))
            theta_opt = value.(theta)
            Xi_opt = value.(Xi)
            eplus_opt = value.(eplus)
            eminus_opt = value.(eminus)
            z_opt = value.(z)
            #@show findall(z_opt.== 1)
        end
        # Loop counter.
        cuts_added = cuts_added+1;
    end # end while loop.
    #@show cuts_added

    return theta_opt, Xi_opt, z_opt, getobjectivevalue(m_outer), cuts_added, z_opt_path
end


function CNLS(X, y, max_cuts, tol, bigM, bounds, fun, seed)
    ## Solving the CNLS by the adapted cutting-plane Algorithm ##
    ## Using the SP to form the initial constraints ##

    # number of observations and variables.
    n = length(y)
    p = size(X,2)

    # Define the reduced master problem.
    m_outer = direct_model(Gurobi.Optimizer(GRB_ENV))
    MOI.set(m_outer, MOI.Silent(), true)

    # Define the variables
    @variable(m_outer, theta[1:n])
    @variable(m_outer, Xi[1:n,1:p] >=0 )
    
    if(bounds == 1)
        for i in 1:p 
            @constraint(m_outer, Xi[:,i] .<= bigM)
            @constraint(m_outer, Xi[:,i] .>= -bigM)
        end
    end

    # Solve the reduced problem.
    full_violation_pair=[]
    numcon = n;
    r=1;
    for i in 1:numcon
        ind1 = i
        for j in 1:r
            ind2 = Distributions.sample(seed, setdiff(collect(1:n),i),1,replace=false)[1];
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
    @objective(m_outer, Min, sum((y[i]-theta[i])^2 for i=1:n))
    t1 = time()
    optimize!(m_outer)
    t2 = time()
    actual_time = t2 - t1
    #println("Solved the reduced master problem")
    theta_opt = value.(theta)
    Xi_opt = value.(Xi)

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
                separation_positive = min.(separation, 0)
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
            @objective(m_outer, Min, sum((y[i]-theta[i])^2 for i=1:n))
            t1 = time()
            optimize!(m_outer)
            t2 = time()
            actual_time += t2 - t1

            theta_opt = value.(theta)
            Xi_opt = value.(Xi)

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
        alpha[i] = theta_opt[i]  - sum(Xi_opt[i, ind] * X[i, ind] for ind=1:p)
    end

    return theta_opt, Xi_opt, opt_reached, cuts_array, Pinf_array, MaxV_array, actual_time, getobjectivevalue(m_outer)
end


function CNLS_NEW(X, y, max_cuts, tol, idxs, bigM, bounds, fun)
    ## Solving the CNLS by the adapted cutting-plane Algorithm ##
    ## CNLS_NEW uses the external initial constraint pair by argument indxs ##
    ## if use the SP to form the inital constaints, the results in CNLS_NEW are equal to those in CNLS ##

    # number of observations and variables.
    n = length(y)
    p = size(X,2)

    # Define the reduced master problem.
    m_outer = direct_model(Gurobi.Optimizer(GRB_ENV))
    MOI.set(m_outer, MOI.Silent(), true)

    # Define the variables
    @variable(m_outer, theta[1:n])
    @variable(m_outer, Xi[1:n,1:p] >=0 )
    
    if(bounds == 1)
        for i in 1:p 
            @constraint(m_outer, Xi[:,i] .<= bigM)
            @constraint(m_outer, Xi[:,i] .>= -bigM)
        end
    end

    # Solve the reduced problem.
    full_violation_pair=[]
    r = size(idxs,2)-1
    numcon = size(idxs,1)
    for i in 1:numcon
        ind1 = idxs[i,1]
        for j in 1:r
            ind2 = idxs[i,j+1]
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
    @objective(m_outer, Min, sum((y[i]-theta[i])^2 for i=1:n))
    t1 = time()
    optimize!(m_outer)
    t2 = time()
    actual_time = t2 - t1
    #println("Solved the reduced master problem")
    theta_opt = value.(theta)
    Xi_opt = value.(Xi)

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
                separation_positive = min.(separation, 0)
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
            @objective(m_outer, Min, sum((y[i]-theta[i])^2 for i=1:n))
            t1 = time()
            optimize!(m_outer)
            t2 = time()
            actual_time += t2 - t1

            theta_opt = value.(theta)
            Xi_opt = value.(Xi)

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

    return theta_opt, Xi_opt, opt_reached, cuts_array, Pinf_array, MaxV_array, actual_time, getobjectivevalue(m_outer)
end


function CNLS_MIO(X, y, k, max_cuts, tol, bigM, z_0, theta_0, Xi_0, fun, seed)
    ## Solving the CNLS with L0 norm by the adapted cutting-plane Algorithm ##
    ## Using the SP to form the initial constraints ##

    n = length(y)
    p = size(X,2)
    z_opt_path = [];
    
    # Define the reduced master problem.
    m_outer = direct_model(Gurobi.Optimizer(GRB_ENV))
    MOI.set(m_outer, MOI.Silent(), true)

    @variable(m_outer,z[1:p],Bin)
    @variable(m_outer, theta[1:n])
    @variable(m_outer, Xi[1:n,1:p] >=0 )
    
    @constraint(m_outer,UB_constr[i=1:p,j=1:n], Xi[j,i] <= bigM*z[i]);
    @constraint(m_outer,LB_constr[i=1:p,j=1:n], Xi[j,i] >= -bigM*z[i]);
    
    @constraint(m_outer, Cardinality_constr, sum(z[i] for i in 1:p) <= k)

    # Solve the reduced problem.
    #println("Adding the constraints")
    idxs = zeros(Int64,n,2);
    for i in 1:n
        ind2 = Distributions.sample(seed, setdiff(collect(1:n),i), 1, replace=false)[1];
        if (fun == "prod")
            @constraint(m_outer, theta[i] + sum((X[ind2,j] - X[i,j])*Xi[i,j] for j =1:p) >= theta[ind2])
        elseif (fun == "cost")
            @constraint(m_outer, theta[i] + sum((X[ind2,j] - X[i,j])*Xi[i,j] for j =1:p) <= theta[ind2])
        end
        idxs[i,1] = copy(i);
        idxs[i,2] = copy(ind2);
    end
    
    @objective(m_outer, Min, sum((y[i]-theta[i])^2 for i=1:n))
    if(length(z_0) > 0)
        set_start_value.(z, z_0)
        set_start_value.(theta, theta_0)
        set_start_value.(Xi, Xi_0)
    end
    #println("Solving the model")
    optimize!(m_outer)
    
    theta_opt = value.(theta);
    Xi_opt = value.(Xi);
    z_opt = value.(z)
    push!(z_opt_path, sort(findall(z_opt .> 0.5)))
    #@show findall(z_opt .== 1)

    cuts_added = 0;
    opt_reached = 0;

    separation = zeros(n);
    separation_positive = zeros(n);
    max_index = zeros(Int,n)
    new_idxs = zeros(Int64,n,2)

    # Adding cuts sequentially.
    while(cuts_added <= max_cuts-1 && opt_reached == 0)
        # Define the separation problem.
        # Sorting.
        # If no violation, then opt_reached == 1.
        # For each i, find a cut.
        violation_pairs=[];
        for i in 1:n
            XiX_i = dot(vec(Xi_opt[i,:]), vec(X[i,:]))
            separation = (theta_opt[i]-XiX_i)*ones(n,1) - theta_opt + X*Xi_opt[i,:]

            if (fun == "cost")
                # calculate infeasibility [Eq. (34)] in Bertsimas and Mundru (2020).
                separation_positive = max.(separation, 0)
                # find the maximum j(i) [Eq. (6)].
                max_value = maximum(separation)
                if(max_value > tol)
                    max_index[i] = copy(argmax(separation)[1])
                    push!(violation_pair, [copy(i), copy(max_index[i])])
                else
                    max_index[i] = 0
                end
            elseif(fun == "prod")
                # calculate infeasibility
                separation_positive = min.(separation, 0)
                norm_infeasibility = norm_infeasibility + norm(separation_positive)^2
                # find the minimum j(i)
                max_value = minimum(separation_positive)
                if(max_value < -tol)
                    max_index[i] = copy(argmin(separation)[1])
                    push!(violation_pair, [copy(i), copy(max_index[i])])
                else
                    max_index[i] = 0
                end
            end
        end

        if(sum(max_index) == 0)
            opt_reached = 1;
            #print("Cutting plane converges \n")
            # Else, add constraint, and re-solve.
        else
            # 1. Add one for each i.
            # 2. Add the j^*(i) for each i.
            # 3. j^*(i) = max_index[i].
            numcon = length(findall(max_index .> 0))
            new_idxs = zeros(Int64,numcon,2)
            numcon_count = 1
            for i in 1:length(violation_pairs)
                i_star = violation_pairs[i][1];
                j_star = violation_pairs[i][2];
                if(j_star > 0 && i_star > 0 )
                    if (fun == "prod")
                        @constraint(m_outer, theta[i_star] + sum((X[j_star,ind] - X[i_star,ind])*Xi[i_star,ind] for ind =1:p) >= theta[j_star] )
                    elseif (fun == "cost")
                        @constraint(m_outer, theta[i_star] + sum((X[j_star,ind] - X[i_star,ind])*Xi[i_star,ind] for ind =1:p) <= theta[j_star] )
                    end
                    new_idxs[numcon_count,1] = copy(i_star) 
                    new_idxs[numcon_count,2] = copy(j_star)
                    numcon_count += 1
                end
            end
    
            # Given current support, polish the feasible warm starts.
            # Solve using Gurobi with the current constraints.
            # Use CNLS_NEW
            S_current = findall(z_opt .>= 0.5)
            #@show S_current
            # Construct new_idxs
            # Solve with max_cuts = 0 -> only the current problem with the added constraints so far.
            idxs = vcat(idxs, new_idxs)
            theta_opt, Xi_opt_1,~, ~, ~,~,~,~ =  CNLS_NEW(X[:,S_current], y, 0, tol, idxs, bigM, 1, fun)
            Xi_opt = zeros(n,d)
            count_ind = 1
            for i in 1:length(S_current)
                Xi_opt[:,S_current[i]] = Xi_opt_1[:,i]
            end
        
            # Set warm starts.
            set_start_value.(theta, theta_opt)
            set_start_value.(Xi, Xi_opt)
            set_start_value.(z, z_opt)
        
            # Re-solve the problem.
            @objective(m_outer, Min, sum((y[i]-theta[i])^2 for i=1:n))
            optimize!(m_outer)

            push!(z_opt_path, sort(findall(z_opt .> 0.5)))
            theta_opt = value.(theta)
            Xi_opt = value.(Xi)
            z_opt = value.(z)
            #@show findall(z_opt.== 1)
        end
        # Loop counter.
        cuts_added = cuts_added+1;
    end # end while loop.
    #@show cuts_added

    return theta_opt, Xi_opt, z_opt, getobjectivevalue(m_outer), cuts_added, z_opt_path
end



function num_linear_constraints(model) 
    ## counting the number of linear constraints ##
    
    return num_constraints(model, AffExpr, MOI.LessThan{Float64}) +
            num_constraints(model, AffExpr, MOI.GreaterThan{Float64}) +
            num_constraints(model, AffExpr, MOI.EqualTo{Float64})
end