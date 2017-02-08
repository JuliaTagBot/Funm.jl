using  ForwardDiff
# σ = n^(-1) * sum(λ[i]), M = T - σ * Iₙ, tol = u
# μ = ||y||, where y solves (I - |N|)y = e and N is strivtly 
# upper triangular part of T.

macro nderivs(f, order)
   dfs = [Symbol(string(:df, i)) for i in 1:order]
   block = Expr(:block)
   push!(block.args, :($(dfs[1]) = x -> ForwardDiff.derivative($f, x)))
   for i in 2:length(dfs)
       push!(block.args, :($(dfs[i]) = x -> ForwardDiff.derivative($(dfs[i-1]), x)))
   end
   ret_stmt = Expr(:tuple, [:($(f)(x)), [:($(df)(x)) for df in dfs]...]...)
   return quote
       x -> begin
           $block
           $ret_stmt
       end
   end
end

function AtomicBlock{TT}(f::Function, T::Matrix{TT}, tol::TT, λ::Array{TT,1})
    n  = LinAlg.checksquare(T)
    # TODO: Eigenvalues
    # λ  = eigvals(T)
    σ  = n^(-1) * sum(λ)
    M  = T - σ * eye(n)
    Fₛ = f(σ) * eye(n)
    μ  = norm(T, Inf)
    P  = M
    s  = 1
    # diff(n) = _diff(f, n)(σ)
    diff = @nderivs f 7
    ds  = diff(σ)
    while true
        inclement = ds[s+1] * P
        Fₛ+= inclement
        P = P * M / (s+1)
        if (norm(inclement, Inf) ≤ tol * norm(Fₛ, Inf))
            ω(sr) = max( [abs( ds[ sr ] ) for λᵢ in λ]  )
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
    n  = LinAlg.checksquare(T)
    Sp = TT[]
    q  = Array(Int, n)
    Sps= Array(Array{TT, 1}, 0)
    Sqs= Array(Array{TT, 1}, n)
    push!(Sps, Sp)
    for i in 1:n
        λᵢ = λ[i]
        if (λᵢ ∉  [Sp for Sp in Sps[1:p-1]])
            push!(Sp,  λᵢ)
            push!(Sps, Sp)
            p += 1
            Sp = TT[]
        end
        # Denote by Sqᵢ the set that contains λᵢ
        q[i] = find(λᵢ ∈ [Sp for Sp in Sps])
        Sqi  = Sps[qi]
        for j = i + 1:n
            λⱼ = λ[j]
            if (λⱼ ∉  Sqi)
                if (abs(λᵢ - λⱼ) ≤ δ)
                    if (λⱼ∉ [S for S in Sqs[1:p-1]])
                        push!(Sqi, λⱼ)
                    else
                        # Move the element of Sₘₐₓ(qᵢ,qⱼ) to Sₘᵢₙ(qᵢ,qⱼ)
                        Sᵤ = Sps[max(q[i], q[j])]
                        push!(Sqs[min(q[i], q[j])], Sᵤ)
                        # Reduce by 1 the indices of sets Sq for q > max(qᵢ,qⱼ)
                        Sqs = [Sqs[1:Sᵤ-1];Sqs[Sᵤ+1:end]]
                        p -= 1
                    end
                end
            end
        end
    end
    Sqs, Sqs
end

function ObtainingPermutation(q::Vector{Int})
    pre(j) = find(λ==j, q)
    ϕ(j) = length(pre(j))
    k = length(q)
    β = 1
    g = Array(Float64, k)
    for i in 1:k
        #FIXME
        g[i] = sum(j)/ϕ(i)
    end
    y = sortperm(g, rev=true)
    for i in y
        if q[β:β+ϕ(i)-1] .≠ i
            f = pre(i)
            g = β:β+ϕ(i)-1
            # Concatenate g(f~=g) and f(f~=g) to the end of ILST and IFST, respectively.
            # Let v = β:f[end] and delete all elements of v that are elements f.
            v = β:f[end]
            v = setdiff(v, f)
            q[g[end]+1:f[end]] = q[v]
            q[g] = ones(eltype(q), length(g))*i
            β = β + ϕ(i)
        end
    end
    ILST, IFST
end

function funm{T}(f::Function, A::Matrix{T})
    schur = schurfact(A)
    if isdiag(schur[:Schur])
        return schur[:vectors] * schur[:Schur] * transpose(schur[:Schur])
    end
    
end

