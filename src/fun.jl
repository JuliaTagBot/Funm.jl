using  ForwardDiff
# σ = n^(-1) * sum(λ[i]), M = T - σ * Iₙ, tol = u
# μ = ||y||, where y solves (I - |N|)y = e and N is strivtly 
# upper triangular part of T.

# function _diff(f::Function, order::Int)
#     if order == 1
#         df = t -> ForwardDiff.derivative(f, t)
#     else
#         dff = t -> ForwardDiff.derivative(_diff(f, order-1), t)
#     end
# end

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

# function ConstructDual{T}(x::T, n)
#    d = Dual(x, one(x))
#    for i = 1:n-1
#        d = Dual(d, one(x))
#    end
#    d
# end

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
    Sps, Sqs
end

# Nested Dual Numbers
# Dual(
# 	Dual(
# 		Dual(
# 			Dual(0.8414709848078965,0.5403023058681398),
# 			Dual(0.5403023058681398,-0.8414709848078965)),
# 		Dual(
# 			Dual(0.5403023058681398,-0.8414709848078965),
# 			Dual(-0.8414709848078965,-0.5403023058681398))),
# 	Dual(
# 		Dual(
# 			Dual(0.5403023058681398,-0.8414709848078965),
# 			Dual(-0.8414709848078965,-0.5403023058681398)),
# 		Dual(
# 			Dual(-0.8414709848078965,-0.5403023058681398),
# 			Dual(-0.5403023058681398,0.8414709848078965))))
