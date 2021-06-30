using Distributions, Random, LinearAlgebra


function DGP(n, d, k, SNR, seed)
  # Data generate processing (DGP)

  # the true X index
  x_true = Distributions.sample(seed, collect(1:d),k,replace=false)

  # input X 
  X = rand(seed, Uniform(1, 10), n, d)

  # true y
  y_true = zeros(n)
  for i in 1:n
      y_true[i] = cumprod([X[i, index].^(0.8/k) for index in x_true], dims=1)[k]
  end

  # epsilon
  sigma_sq = var(y_true)/SNR
  epsilon = rand(seed, Normal(0, sqrt(sigma_sq)), n)

  # observed y 
  y = y_true + epsilon

  return X, y, sort(x_true), epsilon, y_true

end