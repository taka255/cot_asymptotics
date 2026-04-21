using Random
using LinearAlgebra
using Statistics

# ============================================================
# Utility
# ============================================================
function make_sizes(D::Int, τ::Float64)
    M = round(Int, τ * D)

    if M <= 0
        error("M must be positive.")
    end

    return M
end

function sample_X(D::Int, L::Int, rng::AbstractRNG)
    return randn(rng, D, L) / sqrt(D)
end

# ============================================================
# Pretraining data generation
# ============================================================
function generate_pretraining_data(
    D::Int,
    α::Float64,
    τ::Float64,
    σ::Float64;
    rng::AbstractRNG = Random.default_rng()
)
    L = round(Int, α * D)
    M = make_sizes(D, τ)

    task_weights = Matrix{Float64}(undef, D, M)
    Xs = Vector{Matrix{Float64}}(undef, M)
    ys = Vector{Vector{Float64}}(undef, M)

    for μ in 1:M
        w = randn(rng, D)
        X = sample_X(D, L, rng)
        ε = σ .* randn(rng, L)
        y = X' * w + ε

        task_weights[:, μ] = w
        Xs[μ] = X
        ys[μ] = y
    end

    return (
        D = D,
        L = L,
        M = M,
        task_weights = task_weights,
        Xs = Xs,
        ys = ys,
    )
end

# ============================================================
# Learn A*
# ============================================================

"""
    learn_A_star(pretrain; λ)

For each task μ, define
    g_μ = X_μ X_μ' ŵ - X_μ y_μ = - X_μ y_μ,
    ŵ^{pred}_μ = ŵ - (D/L) A g_μ = -(D/L) A g_μ.

The minimizer of
    (1/M) Σ ||ŵ^{pred}_μ - w_μ||^2 + (λ/2)||A||_F^2
is
    A* = [Σ w_μ b_μ'] [Σ b_μ b_μ' + (Mλ/2)I]^{-1},
where
    b_μ = -(D/L) g_μ = (D/L) X_μ y_μ.

Returns:
    A
"""
function learn_A_star(pretrain; λ::Float64)
    D = pretrain.D
    L = pretrain.L
    M = pretrain.M

    S_wb = zeros(Float64, D, D)
    S_bb = zeros(Float64, D, D)

    scale = D / L

    for μ in 1:M
        X = pretrain.Xs[μ]
        y = pretrain.ys[μ]
        w_true = pretrain.task_weights[:, μ]

        b = scale .* (X * y)
        S_wb .+= w_true * b'
        S_bb .+= b * b'
    end

    A = S_wb * inv(S_bb + (M * λ / 2) * I(D))
    return A
end

# ============================================================
# Test-time generalization
# ============================================================

"""
    one_test_trajectory(A, D, L, t_max; rng)

Generate one fresh test task:
    w ~ N(0, I_D)
    X with columns x_l ~ N(0, I_D / D)
    y = X' w   (noise-free, matching the user's formula)

Then run:
    ŵ_{t+1} = ŵ_t - (D/L) A (X X' ŵ_t - X X' w),  ŵ_0 = 0

Returns:
    errs::Vector{Float64} of length t_max+1
where errs[t+1] = ||w - ŵ_t||^2 / D
"""
function one_test_trajectory(
    A::Matrix{Float64},
    D::Int,
    L::Int,
    t_max::Int;
    rng::AbstractRNG = Random.default_rng()
)
    w = randn(rng, D)
    X = sample_X(D, L, rng)

    XXtw = X * (X' * w)
    scale = D / L

    w_hat = zeros(Float64, D)
    errs = zeros(Float64, t_max + 1)

    errs[1] = sum(abs2, w - w_hat) / D   # t = 0

    for t in 1:t_max
        residual = X * (X' * w_hat) - XXtw
        w_hat .-= scale .* (A * residual)
        errs[t + 1] = sum(abs2, w - w_hat) / D
    end

    return errs
end

"""
    estimate_generalization_curve(A, D, α, n_test, t_max; rng)

Average the generalization error over n_test independent fresh tasks.
Returns mean_errs of length t_max+1.
"""
function estimate_generalization_curve(
    A::Matrix{Float64},
    D::Int,
    α::Float64,
    n_test::Int,
    t_max::Int;
    rng::AbstractRNG = Random.default_rng()
)
    L = max(1, round(Int, α * D))
    mean_errs = zeros(Float64, t_max + 1)

    for _ in 1:n_test
        mean_errs .+= one_test_trajectory(A, D, L, t_max; rng=rng)
    end
    mean_errs ./= n_test

    return mean_errs
end

# ============================================================
# End-to-end experiment
# ============================================================

"""
    run_experiment(; kwargs...)

Run:
1. one pretraining stage
2. learn fixed A*
3. evaluate mean test generalization error curve

Keyword arguments:
    D::Int
    α::Float64
    τ::Float64
    σ::Float64       # pretraining noise std
    λ::Float64       # ridge regularization for A
    n_test::Int
    t_max::Int
    seed::Int

Returns a named tuple with:
    A, gen_error, pretrain_info
"""
function run_experiment(;
    D::Int = 600,
    α::Float64 = 2.0,
    τ::Float64 = 4.0,
    σ::Float64 = 0.1,
    λ::Float64 = 1e-4,
    n_test::Int = 100,
    t_max::Int = 20,
    seed::Int = 42,
)
    rng = MersenneTwister(seed)

    # 1. pretraining data
    pretrain = generate_pretraining_data(D, α, τ, σ; rng=rng)

    # 2. learn A*
    A = learn_A_star(pretrain; λ=λ)

    # 3. test generalization
    gen_error = estimate_generalization_curve(A, D, α, n_test, t_max; rng=rng)

    return (
        A = A,
        gen_error = gen_error,
        pretrain_info = (
            D = pretrain.D,
            L = pretrain.L,
            M = pretrain.M,
        )
    )
end


# ============================================================
# Example
# ============================================================
#=
@time result = run_experiment(
    D = 200,
    α = 2.0,
    τ = 4.0,
    σ = 0.1,
    λ = 1e-2,
    n_test = 200,
    t_max = 20,
    seed = 42,
)

print(result.gen_error)

# result.A         : learned A*
# result.gen_error : vector [E_0, E_1, ..., E_tmax]
=#