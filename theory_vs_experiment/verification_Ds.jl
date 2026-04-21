using CairoMakie
using HDF5

include("../module/experiment_simplified_linear_attention.jl")
include("../module/theory_ridgeless.jl")


# parameters
Ds = [100, 200, 400, 800]
alpha = 4.0
tau = 3.0
sigma = 0.01
tmax = 30
n_trials = 5

prec = 2048
n_test = 2000
lambda = 1e-5
seed_base = 42

# theory
generalization_error_theory = zeros(tmax + 1)
Et, b, h, c = theory_Et_direct(alpha, tau, sigma, tmax; prec=prec)
generalization_error_theory .= Float64.(Et)

# experiment
generalization_error_experiment = zeros(length(Ds), n_trials, tmax + 1)
indexes = CartesianIndices((eachindex(Ds), 1:n_trials))
@time Threads.@threads for I in indexes
    D_idx, n_trial_idx = Tuple(I)
    D = Ds[D_idx]
    seed = seed_base + 100 * D_idx + n_trial_idx
    result = run_experiment(D=D, α=alpha, τ=tau, σ=sigma, λ=lambda, n_test=n_test, t_max=tmax, seed=seed)   
    generalization_error_experiment[D_idx, n_trial_idx, :] .= result.gen_error
end


# save results
save_path = joinpath(@__DIR__, "simplified_linear_dynamics_theory_experiment_alpha=$(alpha)_tau=$(tau).h5")
mkpath(dirname(save_path))
h5open(save_path, "w") do file
    # params 
    write(file, "Ds", Ds)
    write(file, "alpha", alpha)
    write(file, "tau", tau)
    write(file, "sigma", sigma)
    write(file, "tmax", tmax)
    write(file, "n_trials", n_trials)
    write(file, "prec", prec)
    write(file, "n_test", n_test)
    write(file, "lambda", lambda)
    write(file, "seed_base", seed_base)

    # results
    write(file, "generalization_error_theory", generalization_error_theory)
    write(file, "generalization_error_experiment", generalization_error_experiment)
end
