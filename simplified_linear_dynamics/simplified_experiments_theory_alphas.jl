using HDF5

include("../module/experiment_simplified_linear_attention.jl")
include("../module/theory_ridgeless.jl")


# parameters
alphas = [0.5, 1.0, 2.0, 4.0]
tau = 4.0
sigma = 0.01
tmax = 60

# experiment
D = 500
lambda = 1e-5
seed = 42
n_test = 2000
n_trials = 5

prec = 1512



# theory
generalization_error_theory = zeros(length(alphas), tmax + 1)
@time for (i, alpha) in enumerate(alphas)
    Et, b, h, c = theory_Et_direct(alpha, tau, sigma, tmax; prec=prec)
    generalization_error_theory[i, :] .= Float64.(Et)
end

# experiment
generalization_error_experiment = zeros(length(alphas), n_trials, tmax + 1)
indexes = CartesianIndices((eachindex(alphas), 1:n_trials))
@time Threads.@threads for I in indexes
    alpha_idx, n_trial_idx = Tuple(I)
    alpha = alphas[alpha_idx]
    result = run_experiment(D=D, α=alpha, τ=tau, σ=sigma, λ=lambda, n_test=n_test, t_max=tmax, seed=seed + 100 * alpha_idx + n_trial_idx)   
    generalization_error_experiment[alpha_idx, n_trial_idx, :] .= result.gen_error
end



# save results
save_path = joinpath(@__DIR__, "simplified_linear_dynamics_theory_experiment_tau4.0.h5")
mkpath(dirname(save_path))
h5open(save_path, "w") do file
    # params 
    write(file, "alphas", alphas)
    write(file, "tau", tau)
    write(file, "sigma", sigma)
    write(file, "tmax", tmax)
    write(file, "D", D)
    write(file, "lambda", lambda)
    write(file, "seed", seed)
    write(file, "n_test", n_test)
    write(file, "n_trials", n_trials)
    write(file, "prec", prec)

    # results
    write(file, "generalization_error_theory", generalization_error_theory)
    write(file, "generalization_error_experiment", generalization_error_experiment)
end

