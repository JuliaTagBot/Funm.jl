using  ForwardDiff
# σ = n^(-1) * sum(λ[i]), M = T - σ * Iₙ, tol = u
# μ = ||y||, where y solves (I - |N|)y = e and N is strivtly 
# upper triangular part of T.

function _diff(f::Function, order::Int)
    if order == 1
        df = t -> ForwardDiff.derivative(f, t)
    else
        dff = t -> ForwardDiff.derivative(_diff(f, order-1), t)
    end
end

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
        inclement = _diff(f, s)(σ) * P
        Fₛ+= inclement
        P = P * M / (s+1)
        if (norm(inclement, Inf) ≤ tol * norm(Fₛ, Inf))
            ω(sr) = max( [abs( f⁽ˢ⁺ʳ⁾(λᵢ) ) for λᵢ in λ]  )
            Δ = max([ω(s+r)/factorial(r) for r in 0:n-1])
            if (μ*Δ*norm(P, Inf) ≤ tol*norm(Fₛ, Inf))
                break
            end
        end
        s+=1
    end
    Fₛ
end

function BlockPattern{TT}(T::Matrix{TT}, λ::Array{TT,1}, δ::TT=0.1)
    p  = 1
    Sps = Set{TT}
    n  = LinAlg.checksquare(T)
    Sp = TT[]
    Sps = Array(Array{TT, 1}, 1)
    Sqs = Array(Array{TT, 1}, n)
    push!(Sps, Sp)
    for i in 1:n
        λᵢ = λ[i]
        if (λᵢ ∉  [Sp for Sp in Sps[1:p-1]])
            push!(Sp, λᵢ)
            p += 1
            Sp = TT[]
        end
        Sqs[i] = Sps[find(λᵢ ∈ Sp for Sp in Sps)]
        for j = i + 1:n
            λⱼ = λ[j]
            if (λⱼ ∉  [S for S in Sq])
                if (abs(λᵢ - λⱼ) ≤ δ)
                    if (λⱼ∉ [S for S in Sqs[1:p-1]])
                        push!(Sqs[i], λⱼ)
                    else
                        # Move the element of Sₘₐₓ(qᵢ,qⱼ) to Sₘᵢₓ(qᵢ,qⱼ)
                        
                        # Reduce by 1 the indices of sets Sq for q > max(qᵢ,qⱼ)
                        p -= 1
                    end
                end
            end
        end
    end
    Sps, Sps
end
