include("quantile_expectile.jl")


function tau_expectile(epsilon, tau)
    
    dec = 2
    probs=collect(0.0001:0.0001:0.9999)
    
    # generate the inverse CDF (F_1(tau) & E_1(tau))
    F_1 = quantile(epsilon, probs, dec);
    E_1 = expectile(epsilon, probs, dec);

    # calculates the inverse of the quantile function
    index = floor(Int64, 1000*tau*10)
    q_inv = F_1[index, 2]
    indexArray = findall( x -> x == q_inv, E_1[:, 2])
    indexArray = floor(Int64, (first(indexArray)+last(indexArray))/2)

    sch2 = E_1[indexArray]
    indexArray2 = findall( x -> x == sch2, E_1)    
    index_e = getindex(indexArray2, 1)[1]
    
    # find the corresponding tau in expectile distribution
    tau_e = round(index_e/(1000*10); digits=3)

    return q_inv, tau_e

end