function nmzPhi(ϕ) 
    #obsolete
    return ϕ ./ sum(ϕ,1)
end

function nmzLambda(λ)
    #obsolte
    return λ ./ sum(λ,1)
end

nTopics(params) = length(params.α )

function logSum(logmat,dim)
    #compute log( sum(exp(logmat), dim) )
    logmax = maximum(logmat,dim)
    return logmax + log( sum( exp(logmat .- logmax), dim))
end

function logNmz(logmat,dim)
    return logmat .- logSum(logmat,dim)
end

function ElogDir(params)
    # compute E[ log x | params ] where theta ~ Dirichlet( x, params )
    return digamma(params) .- digamma(sum(params,1))
end

function mat2ldac(Y, filename)
    #convert a matrix to lda-c format, i.e text file with format (SLOW)
    #N_d word1:count word2:count ...
    f = open(filename, "w")
    V, M = size(Y)

    for m = 1:M
        I = find(Y[:,m])
        V = similar(I)
        V[:] = (Y[I,m])
        I -=1 #zero based
        Nd = length(V)

        docline = ( join([join( pair, ":") for pair in zip(I, V)], " "))
        write(f, "$(Nd) ")
        write(f, docline)
        write(f, "\n")
        #println(docline)

    end
    close(f)
end
