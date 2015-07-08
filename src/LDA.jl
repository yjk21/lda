function updatePhi!(Q, d, params)
    Elogθ = ElogDir(Q.γ[:,d])
    temp = (Q.lλ) .+ Elogθ'
    Q.lϕ[:,:,d] = logNmz(temp, 2)'
#    Q.ϕ[:,:, d] = exp( Q.lϕ[:,:,d])
end


function updateGamma!(Q,d,Y,  params)
    Q.γ[:,d] = params.α + exp(Q.lϕ[:,:,d]) * Y[:,d]              
end


function updateLambda!(Q,Y,params,new)
    V,M = size(Y)

    for k = 1:nTopics(params)

        for d = 1:M

            for o in (Y.colptr[d]: Y.colptr[d+1]-1)
                j = Y.rowval[o]
                ndj = Y.nzval[o]
                lndj = log(ndj)

                Q.lλ[j,k] += log1p(exp(lndj + Q.lϕ[k,j,d]- Q.lλ[j,k]))
            end
        end

        Q.lλ[:,k] += log1p(exp(log(params.η)- Q.lλ[:,k]))
    end
    Q.lλ[:] = logNmz(Q.lλ,1)
end


function updateLambda!(Q,Y,params)
    V,M = size(Y)

    for k = 1:nTopics(params)
        for j = 1:V
            Q.λ[j,k] = params.η
            for d = 1:M
                Q.λ[j,k] += Y[j,d] * Q.ϕ[k,j,d]
            end
        end
    end
    Q.λ[:] = nmzLambda(Q.λ)
end

function getELBo(Y::SparseMatrixCSC, Q, params)
    #Compute evidence lower bound
    V,M = size(Y)
    K = nTopics(params)
    α = params.α
    η = params.η

    elboCorpus = 0.0
    Σα = sum(α)

    for d in 1:M
        Elogθ = ElogDir(Q.γ[:,d])

        elbo =  (lgamma(Σα) - sum(lgamma(α)))
        elbo += dot(α - Q.γ[:,d], Elogθ)
        elbo += sum(lgamma(Q.γ[:,d])) - lgamma(sum(Q.γ[:,d]))

        for o in (Y.colptr[d]: Y.colptr[d+1]-1)
            j = Y.rowval[o]
            ndj = Y.nzval[o]
            lndj = log(ndj)

            for k in 1:K
                elbo += exp(lndj + Q.lϕ[k,j,d]) * ( (Q.lλ[j,k]) + Elogθ[k] - (Q.lϕ[k,j,d]) )
            end
        end
        elboCorpus += elbo
    end

    return elboCorpus
end

function lda!(Q, Y, params, options)
    #Variational Inference for LDA
    V,M = size(Y)

    for it = 1:options.maxItEM
        #E-Step
        for d = 1:M
            updatePhi!(Q,d,params)
            updateGamma!(Q,d,Y, params)
        end
        println("Likelihood: ", getELBo(Y, Q, params))

        #M-Step
        updateLambda!(Q, Y, params,1)
        updateAlpha!(Q, params)
    end
end

function objectiveAlpha(α, Q )
    #updateAlpha is implemented using the linear time newton method from Bleis paper
    K = length(α)
    M = size(Q.γ,2)

    Σα = sum(α)

    ∇ = M*(digamma(Σα) - digamma(α))

    f = 0.0
    for d in 1:M
        Elogθ = ElogDir(Q.γ[:,d])

        f+=  (lgamma(Σα) - sum(lgamma(α))) + dot(α,  Elogθ)
        ∇ += Elogθ
    end

    c = -M * trigamma(Σα)
    u = ones(K)
    A = Diagonal(M * trigamma(α))
    H = Woodbury(A, u, c, u')

    return -f, -∇, H

end

function updateAlpha!(Q, params)
    a = params.α
    for it = 1:3
        f,∇,H = objectiveAlpha(a, Q)
        a = a - H \ ∇
        #    println("F: $(f), opt.cond: $(norm(∇))")
    end
    params.α[:] = a 
end


function fn(a, Q) 
    b,c,d = objectiveAlpha(a, Q)
    return b
end
