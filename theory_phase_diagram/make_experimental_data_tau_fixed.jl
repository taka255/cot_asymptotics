using HDF5

include("../module/experiment_simplified_linear_attention.jl")
include("../module/theory_ridgeless.jl")

# fixed parameters
D = 1200
tau = 4.0
alphas = [0.2, 1.0, 1.5, 2.0]
sigma = 0.0
tmax = 52
n_trials = 4

prec = 2048
n_test = 2000
lambda = 1e-5
seed_base = 42

# theory
generalization_error_theory = zeros(length(alphas), tmax + 1)
for (alpha_idx, alpha) in enumerate(alphas)
    Et, _, _, _ = theory_Et_direct(alpha, tau, sigma, tmax; prec=prec)
    generalization_error_theory[alpha_idx, :] .= Float64.(Et)
end

# experiment
generalization_error_experiment = zeros(length(alphas), n_trials, tmax + 1)
indexes = CartesianIndices((eachindex(alphas), 1:n_trials))
@time Threads.@threads for I in indexes
    alpha_idx, trial_idx = Tuple(I)
    alpha = alphas[alpha_idx]
    seed = seed_base + 100 * alpha_idx + trial_idx
    result = run_experiment(
        D=D,
        α=alpha,
        τ=tau,
        σ=sigma,
        λ=lambda,
        n_test=n_test,
        t_max=tmax,
        seed=seed,
    )
    generalization_error_experiment[alpha_idx, trial_idx, :] .= result.gen_error
end

# save results
save_path = joinpath(
    @__DIR__,
    "data",
    "simplified_linear_dynamics_theory_experiment_D=$(D)_tau=$(tau)_alpha_sweep.h5",
)
mkpath(dirname(save_path))
h5open(save_path, "w") do file
    # params
    write(file, "D", D)
    write(file, "tau", tau)
    write(file, "alphas", alphas)
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
