function runTests()
    println("Running Unit Tests")
    testElbo()
end

function testElbo()
    V = 40 # voc size
    M = 5 # docs 
    K = 2 # topics

    srand(1)

    # generate random documents
    Y = sparse(round(rand(V,M) * 5.))

    Q = VBposterior(Y, K)

    params = VBparams(K)
    η = params.η
    α = params.α

    #init Variational parameters

    #Initialize ϕ, γ like LDA-C
    init!(Q, Y, α, η) 
    #init beta/lambda
    Q.λ = nmzLambda(rand(V,K))

    #precomputations
    ElogBeta = zeros(V,K)
    ElogTheta = zeros(K,M)
    for d = 1:M
        ElogTheta[:,d] = ElogDir(Q.γ[:,d])
    end

    for k = 1:K
        ElogBeta[:,k] = ElogDir(Q.λ[:,k] )
    end

    elboAll = 0.0 
    #TODO add P(β) contributions
    @time  for d in 1:M
        elbo =  (lgamma(sum(α)) - sum(lgamma(α)))
        elbo += dot(α - Q.γ[:,d], ElogTheta[:, d])
        elbo += sum(lgamma(Q.γ[:,d])) - lgamma(sum(Q.γ[:,d]))

        break
        for j in 1:V
            ndj = Y[j,d]
            for k in 1:K
                #elbo += ndj * Q.ϕ[k,j,d] * ( ElogBeta[j,k] + ElogTheta[k,d] - log(Q.ϕ[k,j,d]) )
                elbo += ndj * Q.ϕ[k,j,d] * ( log(Q.λ[j,k]) + ElogTheta[k,d] - log(Q.ϕ[k,j,d]) )
            end
        end

        elboAll += elbo
        #        println("Doc $(d); elbo $(elbo); total $(elboAll)")
    end
    mat2ldac(Y, "/home/90days/ko/test.txt")


    d = 1
    Elogθ = ElogDir(Q.γ[:,d])
    temp = exp(log(Q.λ) .+ Elogθ')
    temp =Q.λ .* exp(Elogθ')


    #    updatePhi!(Q, d, params)

    #    @show a,b,c = objectiveAlpha(ones(K), Q )
    #   r = rand(K)
    #   @show c \ r
    #   @show d \ r

    #    @show methods(Calculus.hessian)
    #    ff(x) = fn(x,Q)
    #    gn = Calculus.gradient(ff)
    #    Hn = Calculus.hessian(ff, ones(K))
    #    @show b
    #    @show gn(ones(K))

    #    updateAlpha!(Q, params)
    options = LDAoptions()
    options.maxItEM = 500000
    params.η=0.1
    @time ldas!(Q, Y, params, options)
    #    @time lda!(Q, Y, params, options)


end

function newtonTest()


end



