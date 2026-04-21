using HDF5

include("../module/experiment_simplified_linear_attention.jl")
include("../module/theory_ridgeless.jl")


# parameters
alpha = 3.0
taus = [1.5, 2.0, 2.5, 3.0]
sigma = 0.01
tmax = 40

# experiment
D = 500
lambda = 1e-5
seed = 42
n_test = 2000
n_trials = 5

prec = 1512



# theory
generalization_error_theory = zeros(length(taus), tmax + 1)
@time for (i, tau) in enumerate(taus)
    Et, b, h, c = theory_Et_direct(alpha, taus[i], sigma, tmax; prec=prec)
    generalization_error_theory[i, :] .= Float64.(Et)
end

# experiment
generalization_error_experiment = zeros(length(taus), n_trials, tmax + 1)
indexes = CartesianIndices((eachindex(taus), 1:n_trials))
@time Threads.@threads for I in indexes
    tau_idx, n_trial_idx = Tuple(I)
    tau = taus[tau_idx]
    result = run_experiment(D=D, α=alpha, τ=tau, σ=sigma, λ=lambda, n_test=n_test, t_max=tmax, seed=seed + 100 * tau_idx + n_trial_idx)   
    generalization_error_experiment[tau_idx, n_trial_idx, :] .= result.gen_error
end



# save results
save_path = joinpath(@__DIR__, "simplified_linear_dynamics_theory_experiment_alpha3.0.h5")
mkpath(dirname(save_path))
h5open(save_path, "w") do file
    # params 
    write(file, "alpha", alpha)
    write(file, "taus", taus)
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

