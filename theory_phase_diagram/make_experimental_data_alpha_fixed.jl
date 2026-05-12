using HDF5

include("../module/experiment_simplified_linear_attention.jl")
include("../module/theory_ridgeless.jl")

# fixed parameters
D = 1200
alpha = 3.0
taus = [1.5, 2.0]
sigma = 0.0
tmax = 52
n_trials = 5

prec = 2048
n_test = 2000
lambda = 1e-5
seed_base = 42

# theory
generalization_error_theory = zeros(length(taus), tmax + 1)
for (tau_idx, tau) in enumerate(taus)
    Et, _, _, _ = theory_Et_direct(alpha, tau, sigma, tmax; prec=prec)
    generalization_error_theory[tau_idx, :] .= Float64.(Et)
end

# experiment
generalization_error_experiment = zeros(length(taus), n_trials, tmax + 1)
indexes = CartesianIndices((eachindex(taus), 1:n_trials))
@time Threads.@threads for I in indexes
    println("I: ", I)
    tau_idx, trial_idx = Tuple(I)
    tau = taus[tau_idx]
    seed = seed_base + 100 * tau_idx + trial_idx
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
    generalization_error_experiment[tau_idx, trial_idx, :] .= result.gen_error
    println("result: ", result.gen_error)
end

# save results
save_path = joinpath(
    @__DIR__,
    "data",
    "simplified_linear_dynamics_theory_experiment_D=$(D)_alpha=$(alpha)_tau_sweep.h5",
)
mkpath(dirname(save_path))
h5open(save_path, "w") do file
    # params
    write(file, "D", D)
    write(file, "alpha", alpha)
    write(file, "taus", taus)
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
