# using  ForwardDiff
# σ = n^(-1) * sum(λ[i]), M = T - σ * Iₙ, tol = u
# μ = ||y||, where y solves (I - |N|)y = e and N is strivtly 
# upper triangular part of T.

# Testing mysin
# Derivative of sin
mysin(x, a) = mysin(x, Val{a%4})
mysin(x, ::Type{Val{0}}) = sin(x)
mysin(x, ::Type{Val{1}}) = cos(x)
mysin(x, ::Type{Val{2}}) = -sin(x)
mysin(x, ::Type{Val{3}}) = -cos(x)


@generated function __diff__{N}(f, ::Type{Val{N}})
    if N == 1
        Expr = :(t -> ForwardDiff.derivative(f, t))
    else
        Expr = :(t -> ForwardDiff.derivative($(__diff__(f, Val{N-1})), t))
    end
    Expr
end

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

function AtomicBlock{TT}(fun::Function, T::UpperTriangular{TT,Array{TT,2}}, tol=eps(), maxIt=500)
    n  = LinAlg.checksquare(T)
    if n==1
        return (fun(T, 0), 1)
    end
    # TODO: Eigenvalues?
    # λ  = eigvals(T)
    σ  = trace(T)/n
    M  = T - σ * eye(n)
    Fₛ = fun(σ, 0) * eye(n)
    μ  = norm(T, Inf)
    P  = M
    max_der = 1
    # diff(n) = _diff(f, n)(σ)
    # diff_ = @nderivs f 7
    # ds  = diff_(σ)
    for s = 1:maxIt
        @show s
        inclement = fun(σ, s) * P
        rel_diff = norm(inclement, Inf)/(tol+norm(Fₛ, Inf))
        Fₛ+= inclement
        P = P * M / (s+1)
        if rel_diff <= tol
            # Approximate the maximum of derivatives in convex set containing
            # eigenvalues by maximum of derivatives at eigenvalues.
            f_der_max = [norm(fun.(diag(T), j), Inf) for j in max_der:s+n-1]
            max_der = s+n
            Ω = 0;
            metaΩ = maximum([f_der_max[s+j]/factorial(j) for j in 0:n-1])
            Ω = max(Ω, metaΩ)
            # ω(sr) = max( [abs( ds[ sr ] ) for λᵢ in λ]  )
            # Δ = max([ω(s+r)/factorial(r) for r in 0:n-1])
            if (μ*Ω*norm(P, Inf) <= tol*norm(Fₛ, Inf))
                return (Fₛ, s)
            end
        end
    end
    println("ERROR!")
end


function blocking{T<:Number, F<:AbstractFloat}(A::UpperTriangular{T, Matrix{T}}; delta::F=0.1)
    a    = diag(A)
    n    = length(a)
    m    = zeros(Int, n)
    maxM = 0

    for i = 1:n

        if m[i] == 0
            m[i] = maxM + 1
            maxM += 1
        end

        for j = i+1:n
            if m[i] .!= m[j]
                if abs(a[i]-a[j]) <= delta
                    if m[j] == 0
                        m[j] = m[i]
                    else
                        p = max(m[i], m[j])
                        q = min(m[i], m[j])
                        if m==p
                            m[1] = q
                        elseif m>p
                            m[1] -= 1
                        end
                    end
                end
            end
        end
    end
    m
end

function reshape_M(M)
    len = length(M)
    A = Array(Int, len, 2)
    for i in 1:len
        A[i,:] = M[i]
    end
    A
end

function swapping(m)
    mmax = maximum(m)
    M = Array(Array{Int,1},0)
    ind = Array(Array{Int,1},0)
    h = zeros(Int, mmax)
    g = zeros(Int, mmax)

    for i = 1:mmax
        p = find(x -> x==i, m)
        h[i] = length(p)
        g[i] = sum(p)/h[i]
    end
    
    y = sort(g)
    mdone = 1
    for i = y
        if any(x->x!=i, m[mdone:mdone+h[i]-1])
            f = find(x -> x==i, m)
            g = mdone:mdone+h[i]-1
            ff = setdiff(f, g)
            gg = setdiff(g, f)
            v = mdone-1 + find(x -> x!=i, m[mdone:f[end]])
            push!(M, vcat(ff, gg))
            m[g[end]+1:f[end]] = m[v]
            m[g] = i*ones(Int, 1,h[i])
            @show push!(ind, collect(mdone:mdone+h[i]-1))
            mdone += h[i]
        else
            push!(ind, collect(mdone:mdone+h[i]-1))
        end
    end
    A = reshape_M(M)
    A, ind, sum(abs(diff(A)))
end

funm{T<:Number}(A::UpperTriangular{T, Matrix{T}}, fun::Function; kwarg...) = funm(eye(A), A, fun; kwarg...)
function funm{T<:Number}(A::Matrix{T}, fun::Function; kwarg...)
    Schur, U = schur(A)
    funm(U, Schur, fun; kwarg...)
end

function funm(U, Schur, fun; delta=0.1, tol=eps(), m=Int[], prnt=false)
    n = LinAlg.checksquare(A)
    if isequal(Schur, triu(Schur))
        F = U*fun.(Diagonal(Schur))*U'
        # n_swaps = 0; n_call = 0; terms = 0; ind = collect(1:n)
        return F
    end

    if isempty(m)
        m = blocking(Schur, delta)
    end

    M, ind, n_swaps = swapping(m)
    m = length(ind)
    F = zeros(n, n)
    n_calls = size(M, 1)
    if n_calls > 0
        U, T = LAPACK.trexc!('V', M[1], M[2], Schur, U)
    end
    #TODO: Add atom evaluation
    for col=1:m
        j = ind[col]
        F[j, j], n_terms = AtomicBlock(fun, F[j,j], tol)
        terms[col] = n_terms

        for row=cow-1:-1:1
            i = ind[row]
            if length(i) == 1 && length(j) == 1
                # Scalar case
                k = i+1:j-1
                meta = T[i,j] * (F[i,i] - F[j,j]) + F[i,k]*T[k,j] - T[i,k]*F[k,j]
                F[i, j] = meta/(T[i,i]-T[j,j])
            else
                k = vcat(ind[row+1:col-1])
                rhs = F[i,i]*T[i,j] - T[i,j]*F[j,j] + F[i,k]*T[k,j] - T[i,k]*F[k,j]
                F[i,j] = sylvester(T[i,i], -T[j,j], rhs)
            end
        end
    end
    F = U*F*U'
end


#=

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
    diff_ = @nderivs f 7
    ds  = diff_(σ)
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

function fun_atom{TT<:Number}(T::Matrix{TT}, fun; tol=eps(), prnt=false)
    itmax = 500
    n = LinAlg.checksquare(A)
    n==1 && (return fun(T), 1)

    lambda = trace(T)/n
    F = eye(n)*fun(lambda)
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
=#
