# σ = n^(-1) * sum(λ[i]), M = T - σ * Iₙ, tol = u
# μ = ||y||, where y solves (I - |N|)y = e and N is strivtly 
# upper triangular part of T.

function AtomicBlock{TT}(f::Function, T::Matrix{TT}, tol::TT, λ::Array{TT,1})
    n  = LinAlg.checksquare(T)
    # λ  = eigvals(T)
    σ  = n^(-1) * sum(λ)
    M  = T - σ * eye(n)
    Fₛ = f(σ) * eye(n)
    μ  = norm(T, Inf)
    P  = M
    s  = 1
    while true
        #TODO Use an effective way to take the derivative of the function f. 
        inclement = f⁽ˢ⁾(σ) * P
        Fₛ+= inclement
        P = P * M / (s+1)
        if (norm(inclement, Inf) ≤ tol * norm(Fₛ, Inf))
            ω(sr) = max( [abs( f⁽ˢ⁺ʳ⁾(λᵢ) ) for λᵢ in λ]  )
            Δ = max([ω(s+r)/factorial(r) for r in 0:n-1])
            if μ*Δ*norm(P, Inf) ≤ tol*norm(Fₛ, Inf)
                break
            end
        end
    end
    Fₛ
end

function BlockPattern{TT}(T::Matrix{TT}, λ::Array{TT,1}, δ::TT=0.1)
    p = 1
    Sₚ = Set{TT}
    n  = LinAlg.checksquare(T)
    for i in 1:n
        λᵢ = λ[i]
        if λᵢ ∉ Sₚ
            push!(Sₚ, λᵢ)
        end
        for j = i + 1:n
            λⱼ = λ[j]
            if λⱼ ∉ Sₚ
                if abs(λᵢ - λⱼ) ≤ δ
                    if λⱼ 
