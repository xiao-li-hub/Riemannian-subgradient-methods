# Riemannian-subgradient-methods
This package implements Riemannian subgradient methods for solving nonsmooth optimization problems over Stiefel manifold.

For orthogonal dictionary learning (ODL) problem, 
a) run ODL_RieSub_linear.m which solves ODL using Riemannian subgradient-type methods with geometrically diminishing stepsizes and generates Figure 3b in the paper. 

b) run ODL_RieSub_polynomial.m which solves ODL using Riemannian subgradient-type methods with polynomially diminishing stepsizes and generates Figure 3a in the paper.


For dual principal component pursuit (DPCP) problem, 
a) run DPCP_RieSub_linear.m which solves DPCP using Riemannian subgradient-type methods with geometrically diminishing stepsizes and generates Figure 2b in the paper;

b) run DPCP_RieSub_polynomial.m which solves DPCP using Riemannian subgradient-type methods with polynomially diminishing stepsizes and generates Figure 2a in the paper.


Reference: 

Xiao Li et. al., ''Weakly Convex Optimization over Stiefel Manifold Using Riemannian Subgradient-Type Methods’’, arXiv:1911.05047. 
