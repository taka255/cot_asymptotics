# ------------------------------------------------------------
# 1D truncated convolution
# ------------------------------------------------------------
function conv_trunc(a::Vector{BigFloat}, b::Vector{BigFloat}, N::Int)
    out = zeros(BigFloat, N + 1)
    for n in 0:N
        s = big"0"
        for k in 0:n
            s += a[k + 1] * b[n - k + 1]
        end
        out[n + 1] = s
    end
    return out
end

# ------------------------------------------------------------
# h(x) = g(u(x)) coefficients via recurrence
# h(x) = sum_{n>=0} h_n x^n
# ------------------------------------------------------------
function h_coeffs(alpha::Real, sigma::Real, N::Int; prec::Int=256)
    setprecision(BigFloat, prec)

    α = BigFloat(alpha)
    σ = BigFloat(sigma)
    A = α + big"1" + σ^2
    B = big"2" + σ^2

    h = zeros(BigFloat, N + 1)  # h[n+1] = h_n
    h[1] = big"1"

    for n in 1:N
        s = big"0"
        # coefficient of x^n in x*h(x)^2
        for k in 0:n-1
            s += h[k + 1] * h[n - k]
        end
        # A*h_n = B*h_{n-1} - s - A*1_{n=1}
        h[n + 1] = (B * h[n] - s - (n == 1 ? A : big"0")) / A
    end

    c = big"1" + (big"1" + σ^2) / α
    return h, c
end

# ------------------------------------------------------------
# Direct coefficients of B(x,y)
#
# Write h_x = h(x), h_y = h(y), r = h - 1, p = h/(1-x).
# Then
#   D(h_x,h_y) * B(x,y) = alpha * p(x) p(y),
# where
#   D(h_x,h_y) = alpha
#                 - (1 + beta(alpha+1)) r_x r_y
#                 - beta r_x^2 r_y
#                 - beta r_x r_y^2.
#
# Since the constant term of D is alpha, the coefficients b_{m,n}
# are determined triangularly by coefficient comparison.
# ------------------------------------------------------------
function direct_B_coeffs(alpha::Real, tau::Real, sigma::Real, N::Int; prec::Int=256)
    setprecision(BigFloat, prec)

    α = BigFloat(alpha)
    τ = BigFloat(tau)
    σ = BigFloat(sigma)

    h, c = h_coeffs(α, σ, N; prec=prec)
    β = (c - big"1") / (τ - big"1")

    # r(x) = h(x) - 1
    r = copy(h)
    r[1] = big"0"

    # r(x)^2
    r2 = conv_trunc(r, r, N)

    # p(x) = h(x)/(1-x) so p_n = sum_{k=0}^n h_k
    p = zeros(BigFloat, N + 1)
    run = big"0"
    for n in 0:N
        run += h[n + 1]
        p[n + 1] = run
    end

    # D(x,y) coefficients and B(x,y) coefficients
    d = zeros(BigFloat, N + 1, N + 1)
    b = zeros(BigFloat, N + 1, N + 1)

    pref = big"1" + β * (α + big"1")
    d[1, 1] = α

    for m in 0:N, n in 0:N
        if m == 0 && n == 0
            continue
        end
        d[m + 1, n + 1] = -pref * r[m + 1] * r[n + 1] - β * r2[m + 1] * r[n + 1] - β * r[m + 1] * r2[n + 1]
    end

    # Solve D * B = alpha * p(x)p(y) coefficient-by-coefficient.
    for totaldeg in 0:(2N)
        mmin = max(0, totaldeg - N)
        mmax = min(N, totaldeg)
        for m in mmin:mmax
            n = totaldeg - m
            if n < 0 || n > N
                continue
            end

            rhs = α * p[m + 1] * p[n + 1]
            acc = rhs
            for i in 0:m, j in 0:n
                if i == 0 && j == 0
                    continue
                end
                acc -= d[i + 1, j + 1] * b[m - i + 1, n - j + 1]
            end
            b[m + 1, n + 1] = acc / α
        end
    end

    return b, h, c
end

# ------------------------------------------------------------
# Diagonal coefficients E_t = [x^t y^t] B(x,y) = b_{t,t}
# ------------------------------------------------------------
function theory_Et_direct(alpha::Real, tau::Real, sigma::Real, tmax::Int; prec::Int=256)
    b, h, c = direct_B_coeffs(alpha, tau, sigma, tmax; prec=prec)
    Et = [b[t + 1, t + 1] for t in 0:tmax]
    return Et, b, h, c
end

# ------------------------------------------------------------
# Theorem-predicted limit in the subcritical convergent regime
# 0 < alpha < 1 and tau > 2 + sigma^2
# ------------------------------------------------------------
function theorem_limit(alpha::Real, tau::Real, sigma::Real; prec::Int=256)
    setprecision(BigFloat, prec)
    α = BigFloat(alpha)
    τ = BigFloat(tau)
    σ = BigFloat(sigma)
    return (big"1" - α) * (τ - big"1") / (τ - big"2" - σ^2)
end


using CairoMakie

"""
    tau_c(alpha, sigma)

Critical value τ_c(α, σ^2) under the symmetric-critical-point ansatz:

    τ_c = 1 +
          ((1+σ²)(α+1+2σ²-√Δ)(2α+2+2σ²-√Δ)) / (2α√Δ),

where

    Δ = (α+1+2σ²)^2 - 4α.

Arguments:
- `alpha` : α > 0
- `sigma` : σ ≥ 0   (note: this is σ, not σ²)

Returns:
- `Float64` by default, or `BigFloat` if inputs are `BigFloat`.
"""
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

# ------------------------------------------------------------
# Example usage when run as a script:
#   julia direct_B_recurrence.jl
# ------------------------------------------------------------
#=
if abspath(PROGRAM_FILE) == @__FILE__
    using .DirectBRecurrence

    alpha = 0.6
    tau = 4.9
    sigma = 0.01
    tmax = 30
    prec = 512

    Et, b, h, c = theory_Et_direct(alpha, tau, sigma, tmax; prec=prec)
    println("Predicted limit = ", theorem_limit(alpha, tau, sigma; prec=prec))
    println("Last few E_t values:")
    for t in max(0, tmax - 10):tmax
        println("t = ", t, ", E_t = ", Et[t + 1])
    end
end
=#