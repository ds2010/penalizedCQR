function expectile(x, probs, dec)
    # calculate the sample expectiles; see expectile.R in expectreg package

    e = mean(x)
    ee = 0*probs
    g = maximum(abs.(x)) * 1e-06

    for k in 1:length(probs)
        p = probs[k]
        if (p == 0)
            ee[k] = minimum(x)
        elseif (p == 1)
            ee[k] = maximum(x)
        else
            for it in 1:20
                w = ifelse.(x .< e, 1 - p, p)
                enew = (w'*x)/sum(w)
                de = max(abs(enew-e))
                e = enew
                if (de < g)
                    break
                end
            end
            ee[k] = e
        end
    end

    ee = round.(ee; digits=dec)

    return hcat(probs, ee)

end


function quantile(x, probs, dec)
    # calculate the sample quantiles; see https://gist.github.com/sikli/f1775feb9736073cefee97ec81f6b193

    n = length(x)
    index = 1 .+ (n-1) .* probs
    
    lo = Int.(floor.(index))
    hi = Int.(ceil.(index))

    x = sort(x)
    qs = x[lo]
    h = index - lo
    qs = (1 .- h) .* qs + h .* x[hi]

    qs = round.(qs; digits=dec)

    return hcat(probs, qs)

end
