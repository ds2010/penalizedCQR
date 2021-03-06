# penalized CQR/CER

This repository presents the main codes to calcualte the L1- and L0-penalized CQR and CER models using the CNLS-Adapted cutting plane (CNLS-A) algorithm. See more discussions in the following working paper:

*Variable selection in convex quantile regression: L1-norm or L0-norm regularization?* available on [arXiv](https://arxiv.org/abs/2107.03119).

## functions
- L1_CQER.jl: solve the L1-norm penalized CQR/CER model
- L0_CQER.jl: solve the L0-norm penalized CQR/CER model
- toolbox.jl: some auxiliary functions to support L1- and L0- CQR/CER calculation

