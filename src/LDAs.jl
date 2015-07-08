function updatePhi!(Q, d, params, ElogBeta)
    Elogθ = ElogDir(Q.γ[:,d])
    temp = ElogBeta.+ Elogθ'
    Q.lϕ[:,:,d] = logNmz(temp, 2)'
end

function getELBo(Y::SparseMatrixCSC, Q, params, ElogBeta)
    #Compute evidence lower bound
    V,M = size(Y)
    K = nTopics(params)
    α = params.α
    η = params.η

    elboCorpus = 0.0

    Σα = sum(α)
    Σλ = exp(logSum(Q.lλ,2))

    for k = 1:K
        for j = 1:M
            elboCorpus += lgamma(V * η) - V*lgamma(η) + (η - 1) * ElogBeta[j,k]
            elboCorpus -= lgamma(Σλ[k]) + lgamma(exp(Q.lλ[j,k]))
        end
    end

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
                elbo += exp(lndj + Q.lϕ[k,j,d]) * ( ElogBeta[j,k] + Elogθ[k] - (Q.lϕ[k,j,d]) )
            end
        end
        elboCorpus += elbo
    end

    return elboCorpus
end

function ldas!(Q, Y, params, options)
    #Variational Inference for LDA
    V,M = size(Y)
    K = nTopics(params)

    ElogBeta = zeros(V,K)

    llIter = [Inf]


    for it = 1:options.maxItEM
        #E-Step
        for k = 1:K
            ElogBeta[:,k] = ElogDir(exp(Q.lλ[:,k]) )
        end

        for d = 1:M
            updatePhi!(Q,d,params,ElogBeta)
            updateGamma!(Q,d,Y, params)
        end
        push!(llIter, getELBo(Y, Q, params,ElogBeta))
        println("Likelihood: ",llIter[end])

        #M-Step
        updateLambda!(Q, Y, params,1)
        updateAlpha!(Q, params)
    end
end
