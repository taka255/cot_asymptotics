include("../module/theory_ridgeless.jl")
using CairoMakie
using HDF5


function tau_c(alpha, sigma)
    α = promote(alpha, sigma)[1]
    σ = promote(alpha, sigma)[2]

    α <= 0 && throw(ArgumentError("alpha must be positive"))
    σ < 0  && throw(ArgumentError("sigma must be nonnegative"))

    Δ = (α + one(α) + 2σ^2)^2 - 4α
    sqrtΔ = sqrt(Δ)

    return one(α) + ((one(α) + σ^2) *
                     (α + one(α) + 2σ^2 - sqrtΔ) *
                     (2α + 2one(α) + 2σ^2 - sqrtΔ)) / (2α * sqrtΔ)
end

alphas = range(0.01, 10.0, length=200)
taus_critical = [tau_c(alpha, 0.01) for alpha in alphas]

alphas_trial = range(0.01, 4.0, length=40)
taus_trial = range(1.01, 5.0, length=40)

errors_at_fixed_t = zeros(length(alphas_trial), length(taus_trial))
t_max = 80
sigma = 0.0

Threads.@threads for I in CartesianIndices((length(alphas_trial), length(taus_trial)))
    i, j = Tuple(I)
    alpha = alphas_trial[i]
    tau = taus_trial[j]
    Et, _, _, _ = theory_Et_direct(alpha, tau, sigma, t_max; prec=214)
    errors_at_fixed_t[i, j] = Et[end]
end



# save data
h5open(joinpath(@__DIR__, "data", "phase_diagram_theory.h5"), "w") do file
    write(file, "alphas_trial", collect(alphas_trial))
    write(file, "taus_trial", collect(taus_trial))
    write(file, "errors_at_fixed_t", errors_at_fixed_t)
    write(file, "t_max", t_max)
    write(file, "sigma", sigma)
    write(file, "alphas", collect(alphas))
    write(file, "taus_critical", collect(taus_critical))
end



fig = Figure()
ax = Axis(fig[1, 1], xlabel="α", ylabel="τ", limits=(0.0, 4.0, 1.0, 5.0))
Label(fig[0, :], L"The generalization error at $t =$%$(t_max)", fontsize=16)

hm = heatmap!(ax, alphas_trial, taus_trial, errors_at_fixed_t, colorscale=log10,
            colorrange=(10^-5, 10^3))
Colorbar(fig[1, 2], hm)
# 臨界曲線: α<1 は点線、α≥1 は実線（α=1 で分割）
let αc = collect(alphas), τc = collect(taus_critical), n = length(alphas)
    j = searchsortedfirst(αc, 1.0)
    τ1 = if j == 1
        τc[1]
    elseif j > n
        τc[end]
    elseif αc[j] == 1.0
        τc[j]
    else
        a0, a1 = αc[j-1], αc[j]
        τc[j-1] + (1.0 - a0) / (a1 - a0) * (τc[j] - τc[j-1])
    end
    has_below = any(<(1.0), αc)
    has_above = any(>=(1.0), αc)
    if has_below
        idx_b = findlast(<(1.0), αc)
        α_d = αc[1:idx_b]
        τ_d = τc[1:idx_b]
        if has_above
            push!(α_d, 1.0)
            push!(τ_d, τ1)
        end
        lines!(ax, α_d, τ_d, color=:white, linewidth=2, linestyle=:dash)
    end
    if has_above
        idx_a = findfirst(>=(1.0), αc)
        α_s = αc[idx_a:n]
        τ_s = τc[idx_a:n]
        if has_below && α_s[1] > 1.0
            α_s = [1.0; α_s]
            τ_s = [τ1; τ_s]
        end
        lines!(ax, α_s, τ_s, color=:white, linewidth=2)
    end
end

display(fig)